"""Junta os comparison.json dos modelos numa tabela única (JSON, CSV e Markdown)."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from slm.compare import SEMANTIC, cards_of, cluster_bootstrap, load_eval, paired_bootstrap
from slm.data import load_records, parse_card_set
from slm.models import MODELS, RECIPE
from slm.train import RECORDS

ROWS = (
    ("parse_rate", "Validade estrutural"),
    ("perplexity", "Perplexidade (ponderada por token)"),
    ("reference_nll", "NLL média da referência"),
    ("concept_similarity", "Similaridade com o conceito"),
    ("semantic_nonredundancy", "Não redundância semântica"),
    ("vendi_score_at_5", "Vendi score @5"),
    ("reference_semantic_alignment", "Alinhamento com a referência"),
    ("emoji_reference_overlap_diagnostic", "Sobreposição de emoji (diagnóstico)"),
    ("latency_seconds", "Latência média (s)"),
    ("generated_tokens", "Tokens gerados"),
)


def model_row(report: dict) -> dict:
    comparisons = report["comparisons"]
    row = {"model": report["model"], "name": MODELS[report["model"]].name}
    for variant in ("base", "adapted"):
        row[f"parse_rate_{variant}"] = report["parseable"][variant] / report["test_concepts"]
        row[f"parseable_{variant}"] = report["parseable"][variant]
        row[f"perplexity_{variant}"] = report["token_weighted_perplexity"][variant]
        for metric, values in comparisons.items():
            if metric != "structural_validity":
                row[f"{metric}_{variant}"] = values[f"{variant}_mean"]
    for metric, values in comparisons.items():
        row[f"{metric}_delta"] = values["adapted_minus_base"]
        row[f"{metric}_ci95"] = [values["ci95_lower"], values["ci95_upper"]]
        row[f"{metric}_n"] = values["n_paired"]
    for metric, values in report["valid_output_means"].items():
        for variant in ("base", "adapted"):
            row[f"{metric}_valid_{variant}"] = values[variant]["mean"]
            row[f"{metric}_valid_n_{variant}"] = values[variant]["n"]
    row["mcnemar_p"] = report["structural_validity_mcnemar"]["p_value"]
    row["semantic_complete_cases"] = report["semantic_complete_cases"]
    row["test_concepts"] = report["test_concepts"]
    return row


def between_models(root: Path, keys: list[str], iterations: int) -> list[dict]:
    blocks = json.loads(Path("data/c2/test_synset_blocks.json").read_text())
    adapted = {}
    for key in keys:
        rows = [json.loads(line) for line in (root / key / "per_concept.jsonl").read_text().splitlines() if line.strip()]
        adapted[key] = {row["concept_id"]: row["adapted"] for row in rows if "concept_similarity" in row["adapted"]}
    result = []
    for left, right in combinations(keys, 2):
        ids = sorted(adapted[left].keys() & adapted[right].keys())
        entry = {"left": left, "right": right, "n_paired": len(ids)}
        for metric in SEMANTIC:
            a = [adapted[left][i][metric] for i in ids]
            b = [adapted[right][i][metric] for i in ids]
            estimate, lower, upper = paired_bootstrap(a, b, iterations)
            _, block_lower, block_upper, _ = cluster_bootstrap(a, b, [blocks[i] for i in ids], iterations)
            entry[metric] = {"left_minus_right": estimate, "ci95": [lower, upper], "synset_block_ci95": [block_lower, block_upper]}
        result.append(entry)
    return result


def repeated_short_text(root: Path, keys: list[str]) -> dict:
    def rate(card_sets):
        card_sets = list(card_sets)
        repeated = sum(len({card.short_text.strip().lower() for card in cards}) < len(cards) for cards in card_sets)
        return {"rate": repeated / len(card_sets), "n": len(card_sets)}

    result = {"reference": rate(parse_card_set(record.target_text) for record in load_records(RECORDS)["test"])}
    for key in keys:
        rows = load_eval(root / key / "eval_adapted.jsonl").values()
        result[key] = rate(cards_of(row) for row in rows if row["parse_valid"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    rows = [
        model_row(json.loads(path.read_text()))
        for key in MODELS
        if (path := args.root / key / "comparison.json").exists()
    ]
    destination = args.root / "report"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    flat = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows]
    with (destination / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["| Métrica | " + " | ".join(row["name"] for row in rows) + " |"]
    lines.append("|---|" + "---|" * len(rows))
    for key, label in ROWS:
        cells = []
        for row in rows:
            source = f"{key}_valid" if f"{key}_valid_adapted" in row else key
            values = [row.get(f"{source}_{variant}") for variant in ("base", "adapted")]
            cells.append(" → ".join("—" if v is None else f"{v:.3f}" for v in values))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Métricas semânticas: média sobre todas as saídas válidas de cada variante; os deltas pareados estão em summary.json.")

    keys = [row["model"] for row in rows]
    pairs = between_models(args.root, keys, RECIPE["bootstrap_iterations"])
    repeated = repeated_short_text(args.root, keys)
    (destination / "between_models.json").write_text(
        json.dumps({"pairs": pairs, "repeated_short_text": repeated}, indent=2, ensure_ascii=False)
    )
    lines.append("")
    lines.append("Modelos adaptados, par a par (esquerda − direita), conceitos válidos nos dois; IC 95% por bootstrap pareado.")
    lines.append("")
    lines.append("| Par | n | Alinhamento | Similaridade | Não redundância | Vendi @5 |")
    lines.append("|---|---|---|---|---|---|")
    for pair in pairs:
        cells = [
            f"{pair[m]['left_minus_right']:+.3f} [{pair[m]['ci95'][0]:+.3f}, {pair[m]['ci95'][1]:+.3f}]"
            for m in ("reference_semantic_alignment", "concept_similarity", "semantic_nonredundancy", "vendi_score_at_5")
        ]
        lines.append(f"| {pair['left']} − {pair['right']} | {pair['n_paired']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Saídas com texto curto repetido: "
        + ", ".join(f"{key} {value['rate']:.3f} (n={value['n']})" for key, value in repeated.items())
    )
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
