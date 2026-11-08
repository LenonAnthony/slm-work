"""Consolida o dataset C2-Record a partir das gerações c2_full do PictoInstruct.

Mantém a partição do dataset_offline_silver_v1 (mesmos conceitos de treino, validação e
teste) e aplica as mesmas exclusões estruturais da seleção v1, escolhendo a primeira
tentativa válida de cada conceito.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import orjson

from slm.data import (
    is_single_emoji_grapheme,
    normalized_key,
    parse_card_set,
    sha256_file,
    stable_id,
)

PICTOINSTRUCT = Path("/Documents/livox-ai/livox-data-science/PictoInstruct/data")
CANDIDATES = PICTOINSTRUCT / "processed/offline/c2_c3_v2/generation/c2_full/candidate_sets.jsonl"
SPLITS = PICTOINSTRUCT / "processed/offline/dataset_offline_silver_v1/split_assignments.json"
CONCEPTS = PICTOINSTRUCT / "interim/offline/grouped/generation_concepts.jsonl"


def jsonl(path: Path) -> list[dict]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b"\n" for row in rows))


def exclusions(candidate: dict) -> list[str]:
    reasons = []
    metrics = candidate.get("metrics")
    if metrics is None:
        reasons.append("missing_proxy_metrics")
    elif metrics.get("diagnostics", {}).get("safety_failures", 0) != 0:
        reasons.append("deterministic_safety_failure")
    cards = candidate["cards"]
    keys = {(normalized_key(c["txt_card"]), normalized_key(c["txt_spoken"]), c["emoji"]) for c in cards}
    if len(cards) != 5 or len(keys) != 5:
        reasons.append("duplicate_complete_card_tuple")
    if any(not is_single_emoji_grapheme(c["emoji"]) for c in cards):
        reasons.append("invalid_emoji")
    return reasons


def render_target(candidate: dict) -> str:
    return "output: " + "\n".join(
        f"{c['txt_card'].strip()}, {c['txt_spoken'].strip()}, {c['emoji']}" for c in candidate["cards"]
    )


def synset_blocks(concepts: list[dict]) -> dict[str, str]:
    parent = {c["concept_id"]: c["concept_id"] for c in concepts}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    first: dict[str, str] = {}
    for concept in sorted(concepts, key=lambda c: c["concept_id"]):
        for synset in sorted(set(concept["synsets"])):
            root, other = sorted((find(first.setdefault(synset, concept["concept_id"])), find(concept["concept_id"])))
            parent[other] = root
    return {concept_id: f"synset-block:{find(concept_id)}" for concept_id in parent}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/c2"))
    args = parser.parse_args()

    assignments = {a["concept_id"]: a for a in orjson.loads(SPLITS.read_bytes())["assignments"]}
    concepts = {c["concept_id"]: c for c in jsonl(CONCEPTS) if c["concept_id"] in assignments}
    candidates: dict[str, list[dict]] = defaultdict(list)
    for candidate in jsonl(CANDIDATES):
        if candidate["concept_id"] in assignments:
            candidates[candidate["concept_id"]].append(candidate)

    records, quarantine = [], []
    for concept_id in sorted(assignments):
        assignment = assignments[concept_id]
        attempts = sorted(candidates.get(concept_id, []), key=lambda c: c["attempt"])
        rejected = {}
        selected = None
        for candidate in attempts:
            reasons = exclusions(candidate)
            if not reasons:
                try:
                    parse_card_set(render_target(candidate))
                except ValueError as error:
                    reasons = [f"parse_round_trip: {error}"]
            if not reasons:
                selected = candidate
                break
            rejected[candidate["attempt"]] = reasons
        if selected is None:
            quarantine.append(
                {
                    "concept_id": concept_id,
                    "split": assignment["split"],
                    "reason": "no_attempt" if not attempts else "no_valid_attempt",
                    "rejected_attempts": rejected,
                }
            )
            continue
        records.append(
            {
                "record_id": stable_id("c2-record", [selected["candidate_set_id"], assignment["component_id"]]),
                "concept_id": concept_id,
                "split": assignment["split"],
                "condition": "C2-Record",
                "input_text": f"input: {concepts[concept_id]['canonical_label'].strip()}",
                "target_text": render_target(selected),
                "source_candidate_set_id": selected["candidate_set_id"],
                "attempt": selected["attempt"],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    training = [{k: r[k] for k in ("record_id", "concept_id", "split", "condition", "input_text", "target_text")} for r in records]
    write_jsonl(args.output / "training_examples.jsonl", training)
    write_jsonl(args.output / "selection.jsonl", records)
    write_jsonl(args.output / "quarantine.jsonl", quarantine)
    test_ids = [r["concept_id"] for r in records if r["split"] == "test"]
    blocks = synset_blocks([concepts[i] for i in test_ids])
    (args.output / "test_synset_blocks.json").write_bytes(orjson.dumps(blocks, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2))

    manifest = {
        "version": "slm_work_c2_record_v1",
        "condition": "C2-Record",
        "generator": {"models": sorted({c["model"] for group in candidates.values() for c in group}), "prompt_versions": sorted({c["prompt_version"] for group in candidates.values() for c in group})},
        "selection_rule": "first attempt without missing metrics, safety failure, duplicate card tuple or parse failure",
        "split_source": "dataset_offline_silver_v1/split_assignments.json",
        "professional_validation": False,
        "release_eligible": False,
        "counts": {
            "records_by_split": dict(sorted(Counter(r["split"] for r in records).items())),
            "selected_attempt": dict(sorted(Counter(str(r["attempt"]) for r in records).items())),
            "quarantine_by_split": dict(sorted(Counter(q["split"] for q in quarantine).items())),
            "quarantine_by_reason": dict(sorted(Counter(q["reason"] for q in quarantine).items())),
            "test_synset_blocks": len(set(blocks.values())),
        },
        "input_hashes": {"candidates": sha256_file(CANDIDATES), "splits": sha256_file(SPLITS), "concepts": sha256_file(CONCEPTS)},
        "output_hashes": {
            name: sha256_file(args.output / name)
            for name in ("training_examples.jsonl", "selection.jsonl", "quarantine.jsonl", "test_synset_blocks.json")
        },
    }
    (args.output / "manifest.json").write_bytes(orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2))
    print(orjson.dumps(manifest["counts"], option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
