"""
Synthetic data generation for KALAVAI proof-of-concept experiments.

Ported from kalavai_experiment.py. Generates byte-level tokenized text
for multiple knowledge domains with distinctly different statistical patterns.
"""

import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from kalavai.config import ExperimentConfig


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_simple(text: str, vocab_size: int) -> list[int]:
    """Byte-level tokenization with vocab_size cap."""
    return [min(b, vocab_size - 1) for b in text.encode("utf-8")]


# ============================================================================
# Text generators per domain
# ============================================================================

def _gen_code(rng: random.Random, data_tokens: int) -> str:
    keywords = [
        "def", "return", "if", "else", "for", "in", "range", "import",
        "class", "self", "print", "while", "try", "except", "with",
        "as", "from", "None", "True", "False", "lambda", "yield",
        "break", "continue", "pass", "raise", "global", "not", "and", "or",
    ]
    ops = ["=", "==", "!=", "+=", "-=", ":", "(", ")", "[", "]", "{", "}", ",", ".", "+", "-", "*", "/"]

    def line():
        k = rng.choice(keywords)
        var = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz_", k=rng.randint(2, 8)))
        op = rng.choice(ops)
        num = str(rng.randint(0, 1000))
        return rng.choice([
            f"{k} {var}{op} {num}\n",
            f"    {var} {op} {var}({num})\n",
            f"{k} {var} in range({num}):\n",
            f"    return {var}\n",
            f"# {var} {k} {num}\n",
        ])

    return "".join(line() for _ in range(data_tokens // 20))


def _gen_stories(rng: random.Random, data_tokens: int) -> str:
    subjects = [
        "the cat", "a dog", "she", "he", "the old man", "the child",
        "the queen", "a bird", "the river", "the moon", "they", "we",
        "the knight", "a wizard", "the forest", "the ship",
    ]
    verbs = [
        "walked", "ran", "said", "looked", "found", "lost", "saw",
        "heard", "felt", "knew", "wanted", "needed", "loved", "feared",
        "remembered", "forgot", "believed", "hoped", "dreamed", "whispered",
    ]
    places = [
        "in the garden", "by the river", "through the forest", "at the castle",
        "under the stars", "near the mountain", "along the path", "in the village",
        "across the sea", "into the cave", "beyond the hills", "beside the fire",
    ]
    adjs = ["quietly", "slowly", "suddenly", "carefully", "bravely", "gently", "fearfully"]

    def line():
        s, v, p = rng.choice(subjects), rng.choice(verbs), rng.choice(places)
        return rng.choice([
            f"{s} {v} {rng.choice(adjs)} {p}. ",
            f"once upon a time, {s} {v} {p}. ",
            f'"{s} {v}," {rng.choice(subjects)} {rng.choice(verbs)}. ',
            f"{p}, {s} {v}. ",
            f"and then {s} {v} {p}, never to return. ",
        ])

    return "".join(line() for _ in range(data_tokens // 30))


def _gen_math(rng: random.Random, data_tokens: int) -> str:
    ops = ["+", "-", "*", "/", "=", "^", "sqrt", "log", "sin", "cos"]
    words = [
        "solve", "calculate", "find", "prove", "given", "where", "let",
        "equals", "therefore", "thus", "since", "because", "if", "then",
        "sum", "product", "integral", "derivative", "limit", "value",
    ]
    vars_ = list("xyznabc") + ["theta", "alpha", "beta", "lambda"]

    def line():
        v1, v2 = rng.choice(vars_), rng.choice(vars_)
        n1, n2 = rng.randint(1, 100), rng.randint(1, 100)
        op = rng.choice(ops)
        w = rng.choice(words)
        return rng.choice([
            f"{w} {v1} {op} {n1} = {n2}. ",
            f"{v1} = {n1} {op} {v2}. ",
            f"{w}: {v1}^2 + {v2}^2 = {n1}. ",
            f"if {v1} = {n1}, then {v2} = {n1 * n2 % 100}. ",
            f"{w} the {op} of {v1} and {v2}. ",
        ])

    return "".join(line() for _ in range(data_tokens // 25))


def _gen_legal(rng: random.Random, data_tokens: int) -> str:
    terms = [
        "plaintiff", "defendant", "court", "jurisdiction", "statute",
        "liability", "contract", "damages", "tort", "negligence",
        "remedy", "pursuant", "herein", "whereby", "notwithstanding",
        "stipulation", "adjudication", "indemnification", "waiver",
    ]
    verbs = [
        "shall", "must", "may", "agrees to", "warrants", "represents",
        "covenants", "undertakes", "acknowledges", "stipulates",
    ]
    entities = [
        "Party A", "Party B", "the Corporation", "the Court", "the State",
        "Respondent", "Petitioner", "the Agreement", "Section 3(b)",
    ]

    def line():
        e1, e2 = rng.choice(entities), rng.choice(entities)
        v = rng.choice(verbs)
        t = rng.choice(terms)
        return rng.choice([
            f"{e1} {v} comply with all applicable {t} requirements. ",
            f"pursuant to the {t}, {e1} {v} indemnify {e2}. ",
            f"notwithstanding the foregoing, {e1} {v} provide notice. ",
            f"the {t} {v} be construed in accordance with applicable law. ",
            f"{e1} hereby acknowledges the {t} obligations herein. ",
        ])

    return "".join(line() for _ in range(data_tokens // 28))


def _gen_creative(rng: random.Random, data_tokens: int) -> str:
    adjectives = [
        "crimson", "silver", "ancient", "forgotten", "whispering",
        "eternal", "shattered", "golden", "luminous", "hollow",
        "trembling", "infinite", "velvet", "shadowed", "blazing",
    ]
    nouns = [
        "sky", "dream", "voice", "shadow", "light", "silence", "flame",
        "heart", "wind", "stone", "memory", "echo", "night", "dawn",
    ]
    verbs = [
        "flows", "breaks", "sings", "fades", "burns", "rises", "falls",
        "lingers", "dances", "weeps", "breathes", "glimmers", "aches",
    ]

    def line():
        a1, a2 = rng.choice(adjectives), rng.choice(adjectives)
        n1, n2 = rng.choice(nouns), rng.choice(nouns)
        v = rng.choice(verbs)
        return rng.choice([
            f"the {a1} {n1} {v} beneath the {a2} {n2}. ",
            f"o {a1} {n1}, thou {v} forever. ",
            f"when the {n1} {v}, the {a2} {n2} remembers. ",
            f"a {a1} {n1} and a {a2} {n2}— ",
            f"the {n1} {v}, the {n2} remains. ",
        ])

    return "".join(line() for _ in range(data_tokens // 25))


# ============================================================================
# Public API
# ============================================================================

_GENERATORS = {
    "code": _gen_code,
    "stories": _gen_stories,
    "math": _gen_math,
    "legal": _gen_legal,
    "creative": _gen_creative,
    # Alias
    "bio": _gen_math,  # biology uses math-like notation in synthetic context
}


def generate_synthetic_data(config: ExperimentConfig, seed_offset: int = 100) -> dict[str, str]:
    """
    Generate synthetic text for each domain in config.data.domains.

    Returns:
        dict of {domain_name: raw_text}
    """
    rng = random.Random((config.alignment.seed if config.alignment else 42) + seed_offset)
    data_tokens = config.data.data_tokens if config.data else 2_000_000

    texts = {}
    for domain in config.data.domains:
        if domain not in _GENERATORS:
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {list(_GENERATORS.keys())}"
            )
        gen_fn = _GENERATORS[domain]
        texts[domain] = gen_fn(rng, data_tokens)
        print(f"  Generated {domain}: {len(texts[domain]):,} chars")

    return texts


class TextDataset(Dataset):
    """Sliding-window dataset over a token sequence."""

    def __init__(self, tokens: list[int], context_length: int):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.context_length - 1)

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx: idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_datasets(
    config: ExperimentConfig,
    train_frac: float = 0.9,
) -> tuple[dict[str, TextDataset], dict[str, TextDataset]]:
    """
    Generate synthetic data for all domains and split into train/eval.

    Returns:
        (train_sets, eval_sets) — dicts keyed by domain name
    """
    vocab_size = config.architecture.vocab_size if config.architecture else 512
    context_length = config.architecture.context_length if config.architecture else 256

    print("Generating synthetic training data...")
    raw_texts = generate_synthetic_data(config)

    train_sets: dict[str, TextDataset] = {}
    eval_sets: dict[str, TextDataset] = {}

    for domain, text in raw_texts.items():
        tokens = tokenize_simple(text, vocab_size)
        split = int(len(tokens) * train_frac)
        train_sets[domain] = TextDataset(tokens[:split], context_length)
        eval_sets[domain] = TextDataset(tokens[split:], context_length)
        print(f"  {domain}: {len(tokens):,} tokens → {split:,} train / {len(tokens)-split:,} eval")

    return train_sets, eval_sets


def make_mixed_dataset(
    datasets: dict[str, TextDataset],
    context_length: int,
) -> TextDataset:
    """
    Interleave tokens from all domains into a single mixed dataset.
    Used for MoE router training and mixed-domain evaluation.
    """
    domain_tokens: list[list[int]] = []
    for ds in datasets.values():
        # Reconstruct flat token sequence from dataset
        flat: list[int] = []
        for i in range(len(ds)):
            x, _ = ds[i]
            flat.extend(x.tolist())
        domain_tokens.append(flat)

    min_len = min(len(t) for t in domain_tokens)
    mixed: list[int] = []
    step = context_length // 2
    for i in range(0, min_len, step):
        for tok_seq in domain_tokens:
            mixed.extend(tok_seq[i: i + step])

    return TextDataset(mixed, context_length)
