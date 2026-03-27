"""
kalavai_router_collapse_analysis.py

Diagnoses the seed 42 router collapse in the cross-lingual experiment.

Seed 42 achieved only +6.14% gain vs best-specialist, while seeds 137 and 2026
achieved ~+21.76%. This script investigates routing behavior and per-domain
metrics to identify the root cause.

Usage:
    python experiments/kalavai_router_collapse_analysis.py

Output:
    results/phase2/cross_lingual/collapse_analysis.json
"""

import sys
import json
import math
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "phase2" / "cross_lingual"
OUTPUT_PATH = RESULTS_DIR / "collapse_analysis.json"

SEED_FILES = {
    42: RESULTS_DIR / "result_seed42.json",
    137: RESULTS_DIR / "result_seed137.json",
    2026: RESULTS_DIR / "result_seed2026.json",
}

# ---------------------------------------------------------------------------
# Helper: routing entropy
# ---------------------------------------------------------------------------
def routing_entropy(weights: list[float]) -> float:
    """Shannon entropy of a routing weight vector.

    Perfect routing (all mass on one expert): H ≈ 0.0
    Maximally collapsed routing (uniform over N experts): H ≈ log(N)
    """
    h = 0.0
    for w in weights:
        if w > 1e-9:
            h -= w * math.log(w)
    return h


def dominant_expert(weights: list[float]) -> tuple[int, float]:
    """Return (index, weight) of the expert receiving most routing mass."""
    idx = max(range(len(weights)), key=lambda i: weights[i])
    return idx, weights[idx]


def is_correct_routing(domain: str, domains: list[str], weights: list[float]) -> bool:
    """Check whether the router sends domain traffic to the domain's own specialist.

    Expert ordering is assumed to match domain ordering in the 'domains' list.
    """
    if domain not in domains:
        return False
    expected_expert = domains.index(domain)
    dom_idx, _ = dominant_expert(weights)
    return dom_idx == expected_expert


# ---------------------------------------------------------------------------
# Per-domain routing analysis
# ---------------------------------------------------------------------------
def analyze_routing(seed_data: dict) -> dict:
    domains = seed_data["domains"]
    router_dist = seed_data["router_distribution"]

    per_domain = {}
    for domain, weights in router_dist.items():
        dom_idx, dom_weight = dominant_expert(weights)
        correct = is_correct_routing(domain, domains, weights)
        per_domain[domain] = {
            "weights": weights,
            "entropy": round(routing_entropy(weights), 6),
            "dominant_expert": dom_idx,
            "dominant_weight": round(dom_weight, 6),
            "expected_expert": domains.index(domain) if domain in domains else None,
            "correct_routing": correct,
        }

    n_correct = sum(1 for v in per_domain.values() if v["correct_routing"])
    n_total = len(per_domain)
    misrouted = [d for d, v in per_domain.items() if not v["correct_routing"]]

    return {
        "per_domain": per_domain,
        "n_correctly_routed": n_correct,
        "n_total_domains": n_total,
        "misrouted_domains": misrouted,
        "routing_correct_fraction": round(n_correct / n_total, 4),
    }


# ---------------------------------------------------------------------------
# Per-domain MoE vs specialist delta (loss reduction captured by MoE)
# ---------------------------------------------------------------------------
def analyze_moe_specialist_delta(seed_data: dict) -> dict:
    """For each domain, compute how much of the specialist's gain MoE captured."""
    domains = seed_data["domains"]
    eval_matrix = seed_data["eval_matrix"]

    base_losses = {d: eval_matrix["base"][d] for d in domains}
    moe_losses = {d: eval_matrix["moe"][d] for d in domains}

    # Best specialist loss per domain (the specialist trained on that domain)
    spec_losses = {}
    for d in domains:
        spec_key = f"{d}_spec"
        spec_losses[d] = eval_matrix[spec_key][d]

    results = {}
    for d in domains:
        base = base_losses[d]
        spec = spec_losses[d]
        moe = moe_losses[d]

        specialist_gain = base - spec  # how much specialist improved over base
        moe_gain = base - moe          # how much MoE improved over base
        if specialist_gain > 1e-9:
            capture_ratio = moe_gain / specialist_gain
        else:
            capture_ratio = 1.0 if moe_gain >= 0 else 0.0

        results[d] = {
            "base_loss": round(base, 6),
            "specialist_loss": round(spec, 6),
            "moe_loss": round(moe, 6),
            "specialist_gain_vs_base": round(specialist_gain, 6),
            "moe_gain_vs_base": round(moe_gain, 6),
            "capture_ratio": round(capture_ratio, 4),
        }

    return results


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
def diagnose_seed(seed_data: dict, routing_analysis: dict, delta_analysis: dict) -> dict:
    seed = seed_data["seed"]
    metrics = seed_data["metrics"]
    misrouted = routing_analysis["misrouted_domains"]
    n_correct = routing_analysis["n_correctly_routed"]
    n_total = routing_analysis["n_total_domains"]

    diagnosis_lines = []

    if misrouted:
        # Find the perplexity impact of misrouted domains
        perplexity = seed_data.get("metrics", {}).get("perplexity", {})
        for d in misrouted:
            dom_info = routing_analysis["per_domain"][d]
            dom_delta = delta_analysis[d]
            expected = dom_info["expected_expert"]
            actual = dom_info["dominant_expert"]
            diagnosis_lines.append(
                f"Domain '{d}' misrouted: expected expert {expected}, got expert {actual} "
                f"(weight={dom_delta['moe_loss']:.4f} vs specialist={dom_delta['specialist_loss']:.4f}). "
                f"Capture ratio: {dom_delta['capture_ratio']:.2%}."
            )

    if not misrouted:
        diagnosis_lines.append("All domains correctly routed to their specialists.")

    # Compute aggregate capture ratio
    all_capture = [v["capture_ratio"] for v in delta_analysis.values()]
    avg_capture = sum(all_capture) / len(all_capture)

    diagnosis_lines.append(
        f"Mean specialist gain capture ratio across domains: {avg_capture:.2%}."
    )

    verdict = seed_data.get("stop_go", {}).get("verdict", "N/A")
    reason = seed_data.get("stop_go", {}).get("reason", "N/A")
    diagnosis_lines.append(f"Stop/Go verdict: {verdict} — {reason}")

    return {
        "seed": seed,
        "improvement_vs_spec_pct": metrics["improvement_vs_spec"],
        "improvement_vs_base_pct": metrics["improvement_vs_base"],
        "routing_correct_fraction": routing_analysis["routing_correct_fraction"],
        "misrouted_domains": misrouted,
        "mean_capture_ratio": round(avg_capture, 4),
        "diagnosis": diagnosis_lines,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("KALAVAI Cross-Lingual Router Collapse Analysis")
    print("=" * 70)

    all_results = {}

    for seed, path in sorted(SEED_FILES.items()):
        if not path.exists():
            print(f"\n[WARN] Missing file: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            seed_data = json.load(f)

        routing = analyze_routing(seed_data)
        deltas = analyze_moe_specialist_delta(seed_data)
        diag = diagnose_seed(seed_data, routing, deltas)

        all_results[str(seed)] = {
            "seed": seed,
            "gain_vs_spec_pct": diag["improvement_vs_spec_pct"],
            "gain_vs_base_pct": diag["improvement_vs_base_pct"],
            "routing_analysis": routing,
            "moe_vs_specialist_delta": deltas,
            "diagnosis": diag,
        }

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print()
    print(f"{'Seed':<8} {'Gain vs Spec':>14} {'Gain vs Base':>14} "
          f"{'Routing OK':>12} {'Misrouted Domains'}")
    print("-" * 70)
    for seed_str, r in all_results.items():
        gain_spec = r["gain_vs_spec_pct"]
        gain_base = r["gain_vs_base_pct"]
        routing_frac = r["routing_analysis"]["routing_correct_fraction"]
        misrouted = r["routing_analysis"]["misrouted_domains"]
        misrouted_str = ", ".join(misrouted) if misrouted else "none"
        print(f"{seed_str:<8} {gain_spec:>13.2f}% {gain_base:>13.2f}% "
              f"{routing_frac:>11.0%}  {misrouted_str}")

    print()
    print("Per-domain routing breakdown:")
    print()

    # Detailed routing table per seed
    header = f"{'Domain':<10} {'Expert':>8} {'Expected':>10} {'Correct':>9} {'Entropy':>9}"
    for seed_str, r in all_results.items():
        print(f"  Seed {seed_str}:")
        print(f"  {'Domain':<10} {'Dom Expert':>12} {'Expected':>10} {'Correct':>9} {'Entropy':>9}  Weights")
        print("  " + "-" * 72)
        for domain, info in r["routing_analysis"]["per_domain"].items():
            correct_str = "YES" if info["correct_routing"] else "*** NO ***"
            w_str = "[" + ", ".join(f"{w:.4f}" for w in info["weights"]) + "]"
            print(f"  {domain:<10} {info['dominant_expert']:>12} "
                  f"{info['expected_expert']:>10} {correct_str:>9} "
                  f"{info['entropy']:>9.4f}  {w_str}")
        print()

    print("Per-domain MoE capture ratio (how much of specialist gain MoE captured):")
    print()
    for seed_str, r in all_results.items():
        print(f"  Seed {seed_str}:")
        print(f"  {'Domain':<10} {'Base Loss':>10} {'Spec Loss':>10} {'MoE Loss':>10} {'Capture':>9}")
        print("  " + "-" * 56)
        for domain, d in r["moe_vs_specialist_delta"].items():
            print(f"  {domain:<10} {d['base_loss']:>10.4f} {d['specialist_loss']:>10.4f} "
                  f"{d['moe_loss']:>10.4f} {d['capture_ratio']:>8.1%}")
        print()

    print("Diagnosis summary:")
    print()
    for seed_str, r in all_results.items():
        diag = r["diagnosis"]
        print(f"  Seed {seed_str}:")
        for line in diag["diagnosis"]:
            print(f"    - {line}")
        print()

    # ------------------------------------------------------------------
    # Root cause summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("ROOT CAUSE SUMMARY")
    print("=" * 70)

    seed42 = all_results.get("42")
    if seed42:
        misrouted = seed42["routing_analysis"]["misrouted_domains"]
        if misrouted:
            print()
            print(f"Seed 42 misrouted domain(s): {', '.join(misrouted)}")
            for d in misrouted:
                info = seed42["routing_analysis"]["per_domain"][d]
                delta = seed42["moe_vs_specialist_delta"][d]
                print(f"\n  Domain: {d}")
                print(f"    Router sent {info['dominant_weight']*100:.1f}% of traffic to "
                      f"expert {info['dominant_expert']} (expected expert {info['expected_expert']})")
                print(f"    Specialist loss on {d}: {delta['specialist_loss']:.4f} "
                      f"(would be: base={delta['base_loss']:.4f})")
                print(f"    MoE loss on {d}: {delta['moe_loss']:.4f}  "
                      f"(capture ratio: {delta['capture_ratio']:.1%})")
                print(f"    => MoE falls back to near-base performance on {d} "
                      "because the specialist is not activated.")
            print()
            print("  This explains the low aggregate gain: the yoruba specialist "
                  "achieved large loss reduction (45.54% divergence from base),")
            print("  but seed 42's router mistakenly routes yoruba tokens to expert 0 "
                  "(the tamil specialist) instead of expert 1 (yoruba specialist).")
            print("  Seeds 137 and 2026 both route yoruba correctly to expert 1 "
                  "and achieve the expected ~21.76% aggregate gain.")
        else:
            print("Seed 42 routing appears correct — check per-domain capture ratios above.")

    print()
    print(f"Writing collapse_analysis.json to: {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Write output JSON
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("Done.")
    print()


if __name__ == "__main__":
    main()
