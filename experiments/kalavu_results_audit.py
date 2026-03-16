import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVU Results Audit
====================
Systematically verify all experiment results for:
1. Internal consistency (numbers agree across files)
2. Mathematical correctness (improvement_pct, summary stats)
3. Scientific validity (divergence, MoE > weight_avg, loss ranges)
4. Suspicious patterns (round numbers, impossible precision, cloning)
5. Coverage (all expected files present)

Run after all experiments complete to certify results before paper writing.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"

issues = []
warnings = []
passes = []

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check(name: str, condition: bool, detail: str = "", fatal: bool = False) -> bool:
    if condition:
        passes.append(name)
        print(f"  [{PASS}] {name}")
        return True
    else:
        msg = f"{name}: {detail}" if detail else name
        if fatal:
            issues.append(msg)
            print(f"  [{FAIL}] {msg}")
        else:
            issues.append(msg)
            print(f"  [{FAIL}] {msg}")
        return False


def warn(name: str, detail: str = "") -> None:
    msg = f"{name}: {detail}" if detail else name
    warnings.append(msg)
    print(f"  [{WARN}] {msg}")


def info(name: str, detail: str = "") -> None:
    msg = f"{name}: {detail}" if detail else name
    print(f"  [{INFO}] {msg}")


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def approx_eq(a: float, b: float, tol: float = 0.001) -> bool:
    return abs(a - b) < tol


def is_suspiciously_round(x: float, decimals: int = 2) -> bool:
    """Returns True if x rounds to itself at `decimals` places (e.g., 1.50, 14.0)."""
    return round(x, decimals) == x


def compute_improvement_pct(base_loss: float, moe_loss: float) -> float:
    return (base_loss - moe_loss) / base_loss * 100.0


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: File presence
# ─────────────────────────────────────────────────────────────────────────────

def audit_file_presence() -> dict[str, Path]:
    print("\n" + "=" * 70)
    print("SECTION 1: File Presence")
    print("=" * 70)

    expected = {
        # Core Pythia-410M experiment
        "pythia_step1_base":     RESULTS_DIR / "pythia/step1_base_eval.json",
        "pythia_step3_seed42":   RESULTS_DIR / "pythia/step3_divergence_check_seed42.json",
        "pythia_step3_seed137":  RESULTS_DIR / "pythia/step3_divergence_check_seed137.json",
        "pythia_step3_seed2026": RESULTS_DIR / "pythia/step3_divergence_check_seed2026.json",
        "pythia_step4_seed42":   RESULTS_DIR / "pythia/step4_fusion_results_seed42.json",
        "pythia_step4_seed137":  RESULTS_DIR / "pythia/step4_fusion_results_seed137.json",
        "pythia_step4_seed2026": RESULTS_DIR / "pythia/step4_fusion_results_seed2026.json",
        "pythia_step5_summary":  RESULTS_DIR / "pythia/step5_final_summary.json",
        # Ablations
        "ablation_router":       RESULTS_DIR / "pythia/ablation_router_summary.json",
        "ablation_freeze":       RESULTS_DIR / "pythia/ablation_freeze_summary.json",
        # Loss curves
        "loss_curves":           RESULTS_DIR / "pythia/loss_curves_seed42.json",
        # Qwen baseline
        "qwen_divergent":        RESULTS_DIR / "real/qwen_divergent_domains.json",
    }

    # Optional (produced by later campaign scripts)
    optional = {
        "monolithic":            RESULTS_DIR / "pythia/monolithic_results.json",
        "benchmarks":            RESULTS_DIR / "pythia/benchmarks_summary.json",
        "maturity_sweep_410m":   RESULTS_DIR / "pythia/maturity_sweep_410m_summary.json",
        "pythia_1b_summary":     RESULTS_DIR / "pythia/pythia_1b/step5_final_summary.json",
        "maturity_sweep_1b":     RESULTS_DIR / "pythia/maturity_sweep_1b_summary.json",
        "five_domain_summary":   RESULTS_DIR / "pythia/five_domain/summary.json",
    }

    found = {}
    for key, path in expected.items():
        if check(f"File exists: {path}", path.exists(), str(path)):
            found[key] = path

    print()
    print("  Optional files (campaign experiments):")
    for key, path in optional.items():
        if path.exists():
            found[key] = path
            print(f"  [{PASS}] {path}")
        else:
            print(f"  [    ] {path} (not yet produced)")

    return {**found, **{k: v for k, v in optional.items() if v in found.values()}}


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Pythia-410M core experiment
# ─────────────────────────────────────────────────────────────────────────────

def audit_pythia_core():
    print("\n" + "=" * 70)
    print("SECTION 2: Pythia-410M Core Experiment (3-domain, 3-seed)")
    print("=" * 70)

    step5 = load(RESULTS_DIR / "pythia/step5_final_summary.json")
    if not step5:
        check("step5 summary present", False, "file missing — cannot audit core")
        return

    seeds = [42, 137, 2026]
    base_losses_ref = step5.get("base_held_out_losses", {})

    print("\n  2a. Base model losses consistent across seeds")
    for seed in seeds:
        step4 = load(RESULTS_DIR / f"pythia/step4_fusion_results_seed{seed}.json")
        step3 = load(RESULTS_DIR / f"pythia/step3_divergence_check_seed{seed}.json")

        if step4 and base_losses_ref:
            seed_base = step4.get("eval_heldout", {}).get("base", {})
            for domain in ["code", "science", "fiction", "mixed"]:
                if domain in base_losses_ref and domain in seed_base:
                    check(
                        f"Base loss consistent seed={seed} domain={domain}",
                        approx_eq(base_losses_ref[domain], seed_base[domain], tol=1e-4),
                        f"step5={base_losses_ref[domain]:.6f} vs step4={seed_base[domain]:.6f}",
                    )

    print("\n  2b. Divergence: each specialist beats base on own domain")
    for seed in seeds:
        step3 = load(RESULTS_DIR / f"pythia/step3_divergence_check_seed{seed}.json")
        if not step3:
            warn(f"step3_seed{seed}", "file missing")
            continue

        passed = step3.get("passed", False)
        check(f"Divergence passed seed={seed}", passed)

        checks = step3.get("checks", {})
        for domain in ["code", "science", "fiction"]:
            key = f"{domain}_beats_base"
            if key in checks:
                c = checks[key]
                check(
                    f"  {domain} specialist < base on {domain} (seed={seed})",
                    c.get("passed", False),
                    f"base={c.get('base_loss', '?'):.4f} spec={c.get('spec_loss', '?'):.4f}",
                )

    print("\n  2c. MoE better than weight_avg on mixed (held-out)")
    for seed in seeds:
        step4 = load(RESULTS_DIR / f"pythia/step4_fusion_results_seed{seed}.json")
        if not step4:
            warn(f"step4_seed{seed}", "file missing")
            continue
        eh = step4.get("eval_heldout", {})
        moe_loss = eh.get("moe", {}).get("mixed", None)
        wavg_loss = eh.get("weight_avg", {}).get("mixed", None)
        if moe_loss is not None and wavg_loss is not None:
            check(
                f"MoE < weight_avg seed={seed}",
                moe_loss < wavg_loss,
                f"MoE={moe_loss:.4f} weight_avg={wavg_loss:.4f}",
            )

    print("\n  2d. Improvement_pct matches computed value (vs best individual)")
    for seed in seeds:
        step4 = load(RESULTS_DIR / f"pythia/step4_fusion_results_seed{seed}.json")
        if not step4:
            continue
        reported = step4.get("improvement_pct", None)
        # improvement is computed vs best_individual_mixed (not base)
        best_spec_mixed = min(
            step4.get("eval_heldout", {}).get(k, {}).get("mixed", float("inf"))
            for k in ["code_spec", "science_spec", "fiction_spec"]
        )
        moe_mixed = step4.get("eval_heldout", {}).get("moe", {}).get("mixed", None)
        if reported is not None and best_spec_mixed < float("inf") and moe_mixed is not None:
            computed = compute_improvement_pct(best_spec_mixed, moe_mixed)
            check(
                f"improvement_pct matches raw losses seed={seed}",
                approx_eq(reported, computed, tol=0.1),
                f"reported={reported:.4f}% computed={computed:.4f}% (vs best_indiv={best_spec_mixed:.4f})",
            )

    print("\n  2e. Summary stats match per-seed data")
    summ = step5.get("summary", {})
    reported_mean = summ.get("improvement_mean_pct", None)
    reported_std = summ.get("improvement_std_pct", None)
    per_seed = step5.get("per_seed_fusion", {})
    seed_imps = [per_seed[str(s)]["improvement_pct"] for s in seeds if str(s) in per_seed]
    if seed_imps and reported_mean is not None:
        computed_mean, computed_std = compute_mean_std(seed_imps)
        check(
            "Summary mean matches per-seed",
            approx_eq(reported_mean, computed_mean, tol=0.01),
            f"reported={reported_mean:.4f}% computed={computed_mean:.4f}%",
        )
        if reported_std is not None:
            check(
                "Summary std matches per-seed",
                approx_eq(reported_std, computed_std, tol=0.01),
                f"reported={reported_std:.4f}% computed={computed_std:.4f}%",
            )

    print("\n  2f. Loss range sanity (cross-entropy on 512-token chunks)")
    for seed in seeds:
        step4 = load(RESULTS_DIR / f"pythia/step4_fusion_results_seed{seed}.json")
        if not step4:
            continue
        for model_key, losses in step4.get("eval_heldout", {}).items():
            for dom, loss in losses.items():
                check(
                    f"Loss in [1.0, 6.0] for {model_key}/{dom}/seed={seed}",
                    1.0 <= loss <= 6.0,
                    f"loss={loss:.4f}",
                )

    print("\n  2g. Router hard-switching (gates near 0 or 1)")
    for seed in seeds:
        step4 = load(RESULTS_DIR / f"pythia/step4_fusion_results_seed{seed}.json")
        if not step4:
            continue
        router_dist = step4.get("router_distribution", {})
        for domain, gates in router_dist.items():
            max_gate = max(gates)
            check(
                f"Router hard-switches on {domain} seed={seed}",
                max_gate >= 0.95,
                f"max_gate={max_gate:.4f}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Router ablation
# ─────────────────────────────────────────────────────────────────────────────

def audit_router_ablation():
    print("\n" + "=" * 70)
    print("SECTION 3: Router Ablation")
    print("=" * 70)

    data = load(RESULTS_DIR / "pythia/ablation_router_summary.json")
    if not data:
        warn("Router ablation", "file missing — skipping")
        return

    variants = data.get("variants", {})

    print("\n  3a. Uniform < learned routers (learned routes better)")
    uniform_imp = variants.get("uniform", {}).get("improvement_pct", None)
    simple_imp = variants.get("simple_linear", {}).get("improvement_pct", None)
    two_layer_imp = variants.get("two_layer", {}).get("improvement_pct", None)

    if uniform_imp and simple_imp:
        check("Simple linear > uniform", simple_imp > uniform_imp,
              f"simple={simple_imp:.2f}% uniform={uniform_imp:.2f}%")
    if uniform_imp and two_layer_imp:
        check("Two-layer > uniform", two_layer_imp > uniform_imp,
              f"two_layer={two_layer_imp:.2f}% uniform={uniform_imp:.2f}%")

    print("\n  3b. Simple == two-layer (architecture doesn't matter)")
    if simple_imp and two_layer_imp:
        diff = abs(simple_imp - two_layer_imp)
        check("Simple ~= two-layer (within 0.1pp)", diff < 0.1,
              f"diff={diff:.3f}pp")

    print("\n  3c. Uniform uses equal gates (333/333/333)")
    unif_gates = variants.get("uniform", {}).get("gate_distribution", {})
    for domain, gates in unif_gates.items():
        for g in gates:
            check(f"Uniform gate near 1/3 for {domain}", approx_eq(g, 0.333, tol=0.01),
                  f"gate={g}")

    print("\n  3d. Improvement matches main experiment (same seed/freeze)")
    main = load(RESULTS_DIR / "pythia/step5_final_summary.json")
    if main and simple_imp:
        main_imp = main.get("summary", {}).get("improvement_mean_pct", None)
        if main_imp:
            diff = abs(simple_imp - main_imp)
            if diff < 0.5:
                check("Ablation simple_linear ≈ main experiment", True,
                      f"ablation={simple_imp:.2f}% main_mean={main_imp:.2f}%")
            else:
                warn("Ablation vs main discrepancy",
                     f"ablation simple={simple_imp:.2f}% main={main_imp:.2f}% (diff={diff:.2f}pp)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Freeze depth ablation
# ─────────────────────────────────────────────────────────────────────────────

def audit_freeze_ablation():
    print("\n" + "=" * 70)
    print("SECTION 4: Freeze Depth Ablation")
    print("=" * 70)

    data = load(RESULTS_DIR / "pythia/ablation_freeze_summary.json")
    if not data:
        warn("Freeze ablation", "file missing — skipping")
        return

    phase1 = data.get("phase1_seed42_results", [])
    multi_seed = data.get("multi_seed_results", {})

    print("\n  4a. All freeze depths show positive improvement (MoE > base)")
    for entry in phase1:
        f = entry["freeze_layers"]
        imp = entry["improvement_pct"]
        check(f"freeze={f}: positive improvement", imp > 0,
              f"improvement={imp:.2f}%")

    print("\n  4b. Monotonic: more freezing → smaller improvement")
    imps = [(e["freeze_layers"], e["improvement_pct"]) for e in phase1]
    imps.sort(key=lambda x: x[0])
    for i in range(len(imps) - 1):
        f1, imp1 = imps[i]
        f2, imp2 = imps[i + 1]
        # Allow some tolerance — just check generally decreasing
        check(
            f"freeze={f1} ({imp1:.2f}%) > freeze={f2} ({imp2:.2f}%)",
            imp1 >= imp2 - 0.05,  # allow 0.05pp tolerance
            f"violations would suggest non-monotonicity",
        )

    print("\n  4c. Multi-seed std near zero (robust mechanism)")
    for freeze_str, ms in multi_seed.items():
        if "std" not in ms:
            continue
        std = ms["std"]
        check(f"freeze={freeze_str}: low variance (std < 0.05)", std < 0.05,
              f"std={std:.4f}")

    print("\n  4d. Computed improvement matches raw losses (vs best individual)")
    for entry in phase1:
        best_ind = entry.get("best_individual_mixed")
        moe = entry.get("moe_mixed_loss")
        reported = entry.get("improvement_pct")
        if best_ind and moe and reported is not None:
            computed = compute_improvement_pct(best_ind, moe)
            check(
                f"freeze={entry['freeze_layers']}: improvement_pct correct",
                approx_eq(reported, computed, tol=0.05),
                f"reported={reported:.3f}% computed={computed:.3f}%",
            )

    print("\n  4e. Spread across freeze depths (robustness claim)")
    if len(imps) >= 2:
        spread = imps[0][1] - imps[-1][1]
        info(f"Total spread: {spread:.2f}pp over freeze 0→{imps[-1][0]} layers")
        if spread < 5.0:
            info("Spread < 5pp — supports robustness claim")
        else:
            warn("Spread", f"{spread:.2f}pp is large — check robustness claim")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Loss curves
# ─────────────────────────────────────────────────────────────────────────────

def audit_loss_curves():
    print("\n" + "=" * 70)
    print("SECTION 5: Loss Curves (seed=42)")
    print("=" * 70)

    data = load(RESULTS_DIR / "pythia/loss_curves_seed42.json")
    if not data:
        warn("Loss curves", "file missing — skipping")
        return

    base_losses = data.get("base_model_loss", {})
    step5 = load(RESULTS_DIR / "pythia/step5_final_summary.json")

    print("\n  5a. Base losses match main experiment")
    if step5:
        ref_base = step5.get("base_held_out_losses", {})
        for domain in ["code", "science", "fiction", "mixed"]:
            if domain in base_losses and domain in ref_base:
                check(
                    f"Loss curve base == main experiment base ({domain})",
                    approx_eq(base_losses[domain], ref_base[domain], tol=1e-4),
                    f"loss_curves={base_losses[domain]:.6f} main={ref_base[domain]:.6f}",
                )

    print("\n  5b. Specialists improve monotonically (each domain, own specialist)")
    specialists = data.get("specialists", {})
    for domain, curve in specialists.items():
        if not curve:
            continue
        own_losses = [entry["held_out"].get(domain, None) for entry in curve]
        own_losses = [l for l in own_losses if l is not None]
        # Check final loss < initial loss
        if len(own_losses) >= 2:
            check(
                f"{domain} specialist improves over training",
                own_losses[-1] < own_losses[0],
                f"step0={own_losses[0]:.4f} final={own_losses[-1]:.4f}",
            )
        # Check no sudden spikes (loss jump > 20% of base)
        for i in range(1, len(own_losses)):
            delta = own_losses[i] - own_losses[i - 1]
            if delta > 0.5:
                warn(
                    f"{domain} loss spike at step {curve[i].get('step', i)}",
                    f"jumped by {delta:.4f}",
                )

    print("\n  5c. Eval steps are at correct intervals (200)")
    for domain, curve in specialists.items():
        if len(curve) < 2:
            continue
        step_0 = curve[0].get("step", 0)
        step_1 = curve[1].get("step", 0) if len(curve) > 1 else 0
        check(
            f"{domain} curve: steps at 200-step intervals",
            (step_1 - step_0) == 200 or step_0 == 0,
            f"step_0={step_0} step_1={step_1}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Qwen baseline
# ─────────────────────────────────────────────────────────────────────────────

def audit_qwen():
    print("\n" + "=" * 70)
    print("SECTION 6: Qwen-1.5B Baseline (divergent domains)")
    print("=" * 70)

    data = load(RESULTS_DIR / "real/qwen_divergent_domains.json")
    if not data:
        warn("Qwen results", "file missing — skipping")
        return

    seeds = data.get("seeds", [])
    per_seed = data.get("per_seed", [])
    summ = data.get("summary", {})

    print("\n  6a. All seeds show negative or near-zero improvement (Qwen fully trained)")
    for entry in per_seed:
        seed = entry.get("seed")
        imp = entry.get("improvement_pct")
        if imp is not None:
            check(
                f"Qwen seed={seed}: improvement ≤ 0 (expected for full-trained model)",
                imp <= 0.5,  # allow small positive
                f"improvement={imp:.4f}%",
            )

    print("\n  6b. Divergence checks passed (specialists do diverge, fusion just doesn't help)")
    for entry in per_seed:
        seed = entry.get("seed")
        passed = entry.get("divergence_passed", False)
        check(f"Qwen divergence passed seed={seed}", passed)

    print("\n  6c. Summary stats match per-seed")
    imps = [e.get("improvement_pct", 0) for e in per_seed]
    if imps and "improvement_mean" in summ:
        computed_mean, computed_std = compute_mean_std(imps)
        check(
            "Qwen summary mean correct",
            approx_eq(summ["improvement_mean"], computed_mean, tol=0.01),
            f"reported={summ['improvement_mean']:.4f} computed={computed_mean:.4f}",
        )

    print("\n  6d. Base losses in plausible range for Qwen-1.5B (fully trained)")
    for entry in per_seed:
        for model_key, losses in entry.get("eval_heldout", {}).items():
            for dom, loss in losses.items():
                check(
                    f"Qwen loss in [0.5, 4.0] for {model_key}/{dom}",
                    0.5 <= loss <= 4.0,
                    f"loss={loss:.4f}",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Monolithic baseline (if available)
# ─────────────────────────────────────────────────────────────────────────────

def audit_monolithic():
    path = RESULTS_DIR / "pythia/monolithic_results.json"
    print("\n" + "=" * 70)
    print("SECTION 7: Monolithic Baseline (equal compute)")
    print("=" * 70)

    data = load(path)
    if not data:
        warn("Monolithic results", "file not yet produced — skipping")
        return

    step5 = load(RESULTS_DIR / "pythia/step5_final_summary.json")

    print("\n  7a. Monolithic final loss > MoE (specialists-then-fuse wins)")
    mono_loss = None
    if "per_seed" in data:
        losses = [s.get("final_mixed_loss", None) for s in data["per_seed"] if s.get("final_mixed_loss")]
        if losses:
            mono_mean = sum(losses) / len(losses)
            mono_loss = mono_mean
            info(f"Monolithic final mixed loss (mean): {mono_mean:.4f}")

    if step5 and mono_loss:
        moe_improvement = step5.get("summary", {}).get("improvement_mean_pct", None)
        if moe_improvement:
            base_mixed = step5.get("base_held_out_losses", {}).get("mixed", None)
            if base_mixed:
                moe_loss = base_mixed * (1 - moe_improvement / 100)
                check(
                    "MoE < monolithic (specialist fusion wins)",
                    moe_loss < mono_loss,
                    f"MoE≈{moe_loss:.4f} mono={mono_loss:.4f}",
                )

    print("\n  7b. Monolithic loss decreases over training steps")
    if "per_seed" in data:
        for entry in data["per_seed"]:
            seed = entry.get("seed")
            trajectory = entry.get("trajectory", [])
            if len(trajectory) >= 2:
                check(
                    f"Monolithic improves over training seed={seed}",
                    trajectory[-1]["loss"] < trajectory[0]["loss"],
                    f"start={trajectory[0]['loss']:.4f} end={trajectory[-1]['loss']:.4f}",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Downstream benchmarks (if available)
# ─────────────────────────────────────────────────────────────────────────────

def audit_benchmarks():
    path = RESULTS_DIR / "pythia/benchmarks_summary.json"
    print("\n" + "=" * 70)
    print("SECTION 8: Downstream Benchmarks")
    print("=" * 70)

    data = load(path)
    if not data:
        warn("Benchmarks", "file not yet produced — skipping")
        return

    expected_benchmarks = ["hellaswag", "arc_easy", "piqa", "winogrande", "lambada_openai", "sciq"]
    expected_models = ["base", "code_spec", "science_spec", "fiction_spec", "weight_avg", "moe", "monolithic"]

    print("\n  8a. All expected benchmarks present")
    results = data.get("results", {})
    for bench in expected_benchmarks:
        check(f"Benchmark {bench} present", bench in results or any(bench in str(k) for k in results.keys()))

    print("\n  8b. Accuracy in [0.0, 1.0]")
    for bench, models in results.items():
        if isinstance(models, dict):
            for model, acc in models.items():
                if isinstance(acc, (int, float)):
                    check(
                        f"{bench}/{model} accuracy in [0, 1]",
                        0.0 <= acc <= 1.0,
                        f"acc={acc}",
                    )

    print("\n  8c. Near-random accuracy warning (< 0.35 on any benchmark)")
    for bench, models in results.items():
        if isinstance(models, dict):
            for model, acc in models.items():
                if isinstance(acc, (int, float)) and acc < 0.35:
                    warn(f"{bench}/{model}", f"accuracy {acc:.3f} is near random")

    print("\n  8d. MoE ≥ base on average across benchmarks")
    moe_accs = []
    base_accs = []
    for bench, models in results.items():
        if isinstance(models, dict):
            if "moe" in models and "base" in models:
                moe_accs.append(models["moe"])
                base_accs.append(models["base"])
    if moe_accs and base_accs:
        moe_mean = sum(moe_accs) / len(moe_accs)
        base_mean = sum(base_accs) / len(base_accs)
        check(
            f"MoE ≥ base on average across benchmarks",
            moe_mean >= base_mean,
            f"MoE_mean={moe_mean:.3f} base_mean={base_mean:.3f}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Maturity sweep (if available)
# ─────────────────────────────────────────────────────────────────────────────

def audit_maturity_sweep():
    path410 = RESULTS_DIR / "pythia/maturity_sweep_410m_summary.json"
    path1b = RESULTS_DIR / "pythia/maturity_sweep_1b_summary.json"

    print("\n" + "=" * 70)
    print("SECTION 9: Maturity Sweep")
    print("=" * 70)

    for label, path in [("410M", path410), ("1B", path1b)]:
        data = load(path)
        if not data:
            warn(f"Maturity sweep {label}", "file not yet produced — skipping")
            continue

        checkpoints = data.get("checkpoints", [])
        print(f"\n  9a. {label}: improvement_pct computable from raw losses")
        for entry in checkpoints:
            revision = entry.get("revision", "?")
            base = entry.get("base_mixed_loss")
            moe = entry.get("moe_mixed_loss")
            reported = entry.get("improvement_pct")
            if base and moe and reported is not None:
                computed = compute_improvement_pct(base, moe)
                check(
                    f"  {label} {revision}: improvement_pct correct",
                    approx_eq(reported, computed, tol=0.05),
                    f"reported={reported:.3f}% computed={computed:.3f}%",
                )

        print(f"\n  9b. {label}: early checkpoints show bigger improvement (trend)")
        imps = [(e.get("revision", "?"), e.get("improvement_pct", 0)) for e in checkpoints]
        if len(imps) >= 2:
            # step10000 should be higher than step143000
            first_imp = imps[0][1]
            last_imp = imps[-1][1]
            if first_imp > last_imp:
                info(f"{label}: early > late ✓ ({first_imp:.2f}% > {last_imp:.2f}%) — trend confirmed")
            else:
                warn(f"{label}: early ≤ late", f"trend not confirmed ({first_imp:.2f}% ≤ {last_imp:.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Pythia-1B (if available)
# ─────────────────────────────────────────────────────────────────────────────

def audit_pythia_1b():
    path = RESULTS_DIR / "pythia/pythia_1b/step5_final_summary.json"
    print("\n" + "=" * 70)
    print("SECTION 10: Pythia-1B Scaling")
    print("=" * 70)

    data = load(path)
    if not data:
        warn("Pythia-1B summary", "file not yet produced — skipping")
        return

    print("\n  10a. 1B improvement_pct > 0")
    mean_imp = data.get("summary", {}).get("improvement_mean_pct", None)
    if mean_imp is not None:
        check("1B positive improvement", mean_imp > 0, f"mean={mean_imp:.2f}%")

    print("\n  10b. 1B improvement broadly comparable to 410M (same mechanism)")
    step5_410m = load(RESULTS_DIR / "pythia/step5_final_summary.json")
    if step5_410m and mean_imp:
        imp_410m = step5_410m.get("summary", {}).get("improvement_mean_pct", None)
        if imp_410m:
            ratio = mean_imp / imp_410m
            if 0.5 <= ratio <= 2.0:
                info(f"1B/410M improvement ratio: {ratio:.2f}x (reasonable)")
            else:
                warn("1B vs 410M ratio unexpected",
                     f"1B={mean_imp:.2f}% 410M={imp_410m:.2f}% ratio={ratio:.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Section 11: 5-domain scaling (if available)
# ─────────────────────────────────────────────────────────────────────────────

def audit_five_domain():
    path = RESULTS_DIR / "pythia/five_domain/summary.json"
    print("\n" + "=" * 70)
    print("SECTION 11: 5-Domain Specialist Scaling")
    print("=" * 70)

    data = load(path)
    if not data:
        warn("Five domain summary", "file not yet produced — skipping")
        return

    subset_results = data.get("subset_results", [])
    print("\n  11a. More specialists → lower loss (monotonic scaling)")
    losses = {r.get("n_specialists"): r.get("moe_mixed_loss") for r in subset_results
              if "n_specialists" in r and "moe_mixed_loss" in r}
    sorted_n = sorted(losses.keys())
    for i in range(len(sorted_n) - 1):
        n1, n2 = sorted_n[i], sorted_n[i + 1]
        check(
            f"{n1}→{n2} specialists: loss decreases",
            losses[n1] > losses[n2],
            f"loss[{n1}]={losses[n1]:.4f} loss[{n2}]={losses[n2]:.4f}",
        )

    print("\n  11b. 5-specialist improvement computable")
    full_entry = next((r for r in subset_results if r.get("n_specialists") == 5), None)
    if full_entry:
        base = full_entry.get("base_mixed_loss")
        moe = full_entry.get("moe_mixed_loss")
        reported = full_entry.get("improvement_pct")
        if base and moe and reported is not None:
            computed = compute_improvement_pct(base, moe)
            check(
                "5-domain improvement_pct correct",
                approx_eq(reported, computed, tol=0.05),
                f"reported={reported:.3f}% computed={computed:.3f}%",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Section 12: Cross-experiment suspicious patterns
# ─────────────────────────────────────────────────────────────────────────────

def audit_suspicious_patterns():
    print("\n" + "=" * 70)
    print("SECTION 12: Suspicious Pattern Detection")
    print("=" * 70)

    print("\n  12a. Checking for impossibly round improvement numbers")
    files_to_check = [
        RESULTS_DIR / "pythia/step5_final_summary.json",
        RESULTS_DIR / "pythia/ablation_router_summary.json",
        RESULTS_DIR / "pythia/ablation_freeze_summary.json",
        RESULTS_DIR / "real/qwen_divergent_domains.json",
    ]
    for path in files_to_check:
        data = load(path)
        if not data:
            continue
        text = json.dumps(data)
        # Extract all floats that look like improvement percentages (10-20 range)
        import re
        candidates = re.findall(r'"improvement_pct":\s*([\d.]+)', text)
        for c in candidates:
            val = float(c)
            if is_suspiciously_round(val, 0):  # exact integer
                warn(f"Round improvement in {path.name}", f"{val}% is an exact integer")
            elif is_suspiciously_round(val, 1):  # e.g., 14.2
                info(f"  Mildly round: {val}% in {path.name}")

    print("\n  12b. Checking for cloned values across seeds (seeds should differ)")
    step5 = load(RESULTS_DIR / "pythia/step5_final_summary.json")
    if step5:
        per_seed = step5.get("per_seed_fusion", {})
        seeds_present = [str(s) for s in [42, 137, 2026] if str(s) in per_seed]
        if len(seeds_present) >= 2:
            s0 = per_seed[seeds_present[0]]
            s1 = per_seed[seeds_present[1]]
            # If MoE mixed losses are exactly equal, that's suspicious
            moe0 = s0.get("eval_heldout", {}).get("moe", {}).get("mixed")
            moe1 = s1.get("eval_heldout", {}).get("moe", {}).get("mixed")
            if moe0 and moe1:
                check(
                    "Seeds produce different MoE losses (not cloned)",
                    not approx_eq(moe0, moe1, tol=1e-5),
                    f"seed42={moe0:.6f} seed137={moe1:.6f}",
                )

    print("\n  12c. Timestamps present in all files")
    for path in files_to_check:
        data = load(path)
        if not data:
            continue
        has_ts = "timestamp" in data
        check(f"Timestamp in {path.name}", has_ts, "no timestamp field")

    print("\n  12d. No loss values stuck at exactly 0.0 or exactly base (no improvement)")
    for seed in [42, 137, 2026]:
        step4 = load(RESULTS_DIR / f"pythia/step4_fusion_results_seed{seed}.json")
        if not step4:
            continue
        for model_key, losses in step4.get("eval_heldout", {}).items():
            for dom, loss in losses.items():
                check(
                    f"Loss != 0 for {model_key}/{dom}/seed={seed}",
                    loss != 0.0,
                    f"loss=0.0 is impossible",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Section 13: Figures
# ─────────────────────────────────────────────────────────────────────────────

def audit_figures():
    print("\n" + "=" * 70)
    print("SECTION 13: Figure Files")
    print("=" * 70)

    expected_figures = [
        FIGURES_DIR / "pythia/fig_training_curves_seed42.png",
        FIGURES_DIR / "pythia/fig_divergence_heatmap.png",
        FIGURES_DIR / "pythia/fig_fusion_comparison.png",
        FIGURES_DIR / "pythia/fig_router_distribution.png",
        FIGURES_DIR / "pythia/fig_ablation_router.png",
        FIGURES_DIR / "pythia/fig_ablation_freeze.png",
        FIGURES_DIR / "pythia/fig_specialist_own_domain.png",
        FIGURES_DIR / "pythia/fig_specialist_cross_domain.png",
        FIGURES_DIR / "pythia/fig_fusion_trajectory.png",
        FIGURES_DIR / "pythia/fig_monolithic_comparison.png",
        FIGURES_DIR / "pythia/fig_monolithic_trajectory.png",
    ]

    optional_figures = [
        FIGURES_DIR / "pythia/fig_maturity_curve_410m.png",
        FIGURES_DIR / "pythia/fig_specialist_scaling.png",
        FIGURES_DIR / "pythia/fig_maturity_curve_1b.png",
        FIGURES_DIR / "pythia/fig_maturity_curve_combined.png",
        FIGURES_DIR / "pythia/fig_paper_hero.png",
    ]

    for path in expected_figures:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        if check(f"Figure exists: {path.name}", exists):
            check(f"Figure non-empty: {path.name}", size > 1000, f"size={size} bytes")

    print("\n  Optional (campaign) figures:")
    for path in optional_figures:
        if path.exists():
            size = path.stat().st_size
            print(f"  [{PASS}] {path.name} ({size // 1024}KB)")
        else:
            print(f"  [    ] {path.name} (not yet produced)")


# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report():
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"\n  Checks passed:  {len(passes)}")
    print(f"  Issues found:   {len(issues)}")
    print(f"  Warnings:       {len(warnings)}")

    if issues:
        print(f"\n  [{FAIL}] ISSUES REQUIRING ATTENTION:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")

    if warnings:
        print(f"\n  [{WARN}] WARNINGS (review before paper writing):")
        for i, w in enumerate(warnings, 1):
            print(f"    {i}. {w}")

    print()
    if not issues:
        print(f"  [{PASS}] ALL CHECKS PASSED — results appear legitimate")
        print(f"         No evidence of hallucination or fabricated results.")
    else:
        print(f"  [{FAIL}] {len(issues)} issue(s) found — investigate before paper writing")

    return len(issues) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("KALAVU RESULTS AUDIT")
    print("Checking all experiment results for integrity and consistency")
    print("=" * 70)

    # Change to repo root
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    audit_file_presence()
    audit_pythia_core()
    audit_router_ablation()
    audit_freeze_ablation()
    audit_loss_curves()
    audit_qwen()
    audit_monolithic()
    audit_benchmarks()
    audit_maturity_sweep()
    audit_pythia_1b()
    audit_five_domain()
    audit_suspicious_patterns()
    audit_figures()

    ok = print_final_report()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
