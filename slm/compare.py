"""Compara base e adaptado com as métricas do PictoInstruct (porte de training/evaluation.py)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

from slm.data import Card, concept_label, load_records, parse_card_set
from slm.models import MODELS, RECIPE
from slm.train import RECORDS

SEMANTIC = (
    "concept_similarity",
    "semantic_nonredundancy",
    "vendi_score_at_5",
    "reference_semantic_alignment",
    "emoji_reference_overlap_diagnostic",
)
LOWER_IS_BETTER = {"reference_nll", "latency_seconds", "generated_tokens"}


def paired_bootstrap(left, right, iterations, seed=3407):
    left, right = np.asarray(left, float), np.asarray(right, float)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(left))
    replicates = np.empty(iterations)
    for i in range(iterations):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        replicates[i] = np.mean(left[sampled] - right[sampled])
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return float(np.mean(left - right)), float(lower), float(upper)


def cluster_bootstrap(left, right, blocks, iterations, seed=3407):
    differences = np.asarray(left, float) - np.asarray(right, float)
    blocks = np.asarray(blocks)
    names = sorted(set(blocks.tolist()))
    sums = np.asarray([differences[blocks == name].sum() for name in names])
    counts = np.asarray([np.count_nonzero(blocks == name) for name in names], float)
    rng = np.random.default_rng(seed)
    replicates = np.empty(iterations)
    for i in range(iterations):
        sampled = rng.integers(0, len(names), size=len(names))
        replicates[i] = sums[sampled].sum() / counts[sampled].sum()
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return float(np.mean(differences)), float(lower), float(upper), len(names)


def exact_mcnemar(left, right):
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    total = left_only + right_only
    p_value = float(binomtest(left_only, total, 0.5).pvalue) if total else 1.0
    return {"left_only": left_only, "right_only": right_only, "p_value": p_value}


def vendi_score(embeddings):
    eigenvalues = np.clip(np.linalg.eigvalsh(np.clip(embeddings @ embeddings.T, -1, 1)), 0, None)
    if eigenvalues.sum() <= 0:
        return 1.0
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def combined(card: Card) -> str:
    return f"{card.short_text}. {card.spoken_text}"


def semantic_values(label, generated, reference, vectors):
    gen = np.stack([vectors[combined(card)] for card in generated])
    ref = np.stack([vectors[combined(card)] for card in reference])
    pairs = [float(gen[i] @ gen[j]) for i in range(5) for j in range(i + 1, 5)]
    similarities = gen @ ref.T
    rows, cols = linear_sum_assignment(-similarities)
    reference_emojis = {card.emoji for card in reference}
    return {
        "concept_similarity": float(np.mean(gen @ vectors[label])),
        "semantic_nonredundancy": 1.0 - max(pairs),
        "vendi_score_at_5": vendi_score(gen),
        "reference_semantic_alignment": float(np.mean(similarities[rows, cols])),
        "emoji_reference_overlap_diagnostic": float(np.mean([c.emoji in reference_emojis for c in generated])),
    }


def load_eval(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["concept_id"]: row for row in rows}


def cards_of(row) -> tuple[Card, ...]:
    return tuple(Card(**card) for card in row["cards"])


def interval(adapted, base, blocks, iterations):
    estimate, lower, upper = paired_bootstrap(adapted, base, iterations)
    _, block_lower, block_upper, n_blocks = cluster_bootstrap(adapted, base, blocks, iterations)
    return {
        "n_paired": len(base),
        "base_mean": float(np.mean(base)),
        "adapted_mean": float(np.mean(adapted)),
        "adapted_minus_base": estimate,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "synset_block_ci95": [block_lower, block_upper],
        "n_blocks": n_blocks,
    }


def compare(key, tests, blocks, encoders, vectors_cache, iterations, root=Path("outputs")):
    folder = root / key
    base, adapted = load_eval(folder / "eval_base.jsonl"), load_eval(folder / "eval_adapted.jsonl")
    concept_ids = [record.concept_id for record in tests]
    if set(base) != set(concept_ids) or set(adapted) != set(concept_ids):
        raise ValueError(f"{key}: base and adapted must cover the full test population")
    references = {record.concept_id: parse_card_set(record.target_text) for record in tests}
    labels = {record.concept_id: concept_label(record.input_text) for record in tests}

    texts = []
    for concept_id in concept_ids:
        texts.append(labels[concept_id])
        texts.extend(combined(card) for card in references[concept_id])
        for row in (base[concept_id], adapted[concept_id]):
            if row["parse_valid"]:
                texts.extend(combined(card) for card in cards_of(row))
    for name, encoder in encoders.items():
        missing = [text for text in dict.fromkeys(texts) if text not in vectors_cache[name]]
        for start in range(0, len(missing), RECIPE["semantic_batch_size"]):
            chunk = missing[start : start + RECIPE["semantic_batch_size"]]
            matrix = encoder.encode(chunk, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            vectors_cache[name].update(zip(chunk, np.asarray(matrix, np.float32)))

    paired = {m: {"base": [], "adapted": [], "ids": []} for m in ("structural_validity", "reference_nll", "latency_seconds", "generated_tokens", *SEMANTIC)}
    per_encoder = {name: {m: {"base": [], "adapted": [], "ids": []} for m in SEMANTIC} for name in encoders}
    valid_only = {m: {"base": [], "adapted": []} for m in SEMANTIC}
    per_concept = []
    for concept_id in concept_ids:
        b, a = base[concept_id], adapted[concept_id]
        row = {"concept_id": concept_id, "base": {}, "adapted": {}}
        for metric, getter in (
            ("structural_validity", lambda r: float(r["parse_valid"])),
            ("reference_nll", lambda r: r["reference_nll"]),
            ("latency_seconds", lambda r: r["latency_seconds"]),
            ("generated_tokens", lambda r: float(r["generated_tokens"])),
        ):
            paired[metric]["base"].append(getter(b))
            paired[metric]["adapted"].append(getter(a))
            paired[metric]["ids"].append(concept_id)
            row["base"][metric], row["adapted"][metric] = getter(b), getter(a)
        for variant, record in (("base", b), ("adapted", a)):
            if not record["parse_valid"]:
                continue
            values = {
                name: semantic_values(labels[concept_id], cards_of(record), references[concept_id], vectors_cache[name])
                for name in encoders
            }
            for metric in SEMANTIC:
                row[variant][metric] = float(np.mean([values[name][metric] for name in encoders]))
                valid_only[metric][variant].append(row[variant][metric])
            row[variant]["encoders"] = values
        if b["parse_valid"] and a["parse_valid"]:
            for variant in ("base", "adapted"):
                for metric in SEMANTIC:
                    paired[metric][variant].append(row[variant][metric])
                    for name in encoders:
                        per_encoder[name][metric][variant].append(row[variant]["encoders"][name][metric])
            for metric in SEMANTIC:
                paired[metric]["ids"].append(concept_id)
                for name in encoders:
                    per_encoder[name][metric]["ids"].append(concept_id)
        per_concept.append(row)

    def summarize(values):
        return interval(values["adapted"], values["base"], [blocks[i] for i in values["ids"]], iterations)

    comparisons = {m: {**summarize(v), "higher_is_better": m not in LOWER_IS_BETTER} for m, v in paired.items() if v["base"]}
    encoder_comparisons = {
        name: {m: {**summarize(v), "higher_is_better": True} for m, v in metrics.items() if v["base"]}
        for name, metrics in per_encoder.items()
    }
    nll_tokens = {variant: sum(r["reference_tokens"] for r in rows.values()) for variant, rows in (("base", base), ("adapted", adapted))}
    corpus_nll = {
        variant: sum(r["reference_nll"] * r["reference_tokens"] for r in rows.values()) / nll_tokens[variant]
        for variant, rows in (("base", base), ("adapted", adapted))
    }
    report = {
        "model": key,
        "model_id": MODELS[key].model_id,
        "test_concepts": len(concept_ids),
        "parseable": {"base": sum(r["parse_valid"] for r in base.values()), "adapted": sum(r["parse_valid"] for r in adapted.values())},
        "semantic_complete_cases": len(paired["concept_similarity"]["base"]),
        "token_weighted_nll": corpus_nll,
        "token_weighted_perplexity": {variant: math.exp(value) for variant, value in corpus_nll.items()},
        "structural_validity_mcnemar": exact_mcnemar(
            [base[i]["parse_valid"] for i in concept_ids], [adapted[i]["parse_valid"] for i in concept_ids]
        ),
        "comparisons": comparisons,
        "encoder_comparisons": encoder_comparisons,
        "valid_output_means": {
            m: {v: {"mean": float(np.mean(x)) if x else None, "n": len(x)} for v, x in values.items()}
            for m, values in valid_only.items()
        },
        "encoders": RECIPE["encoders"],
        "note": "Métricas de embedding são diagnósticos offline, não julgamento profissional de CAA.",
    }
    with (folder / "per_concept.jsonl").open("w") as handle:
        for row in per_concept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (folder / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--iterations", type=int, default=RECIPE["bootstrap_iterations"])
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    tests = load_records(RECORDS)["test"][: args.limit]
    blocks = json.loads(Path("data/c2/test_synset_blocks.json").read_text())
    encoders = {name: SentenceTransformer(model_id) for name, model_id in RECIPE["encoders"].items()}
    cache = {name: {} for name in encoders}
    for key in args.models:
        report = compare(key, tests, blocks, encoders, cache, args.iterations, args.root)
        print(key, json.dumps({"parseable": report["parseable"], "ppl": report["token_weighted_perplexity"]}))


if __name__ == "__main__":
    main()
