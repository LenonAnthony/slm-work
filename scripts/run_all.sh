#!/usr/bin/env bash
# Treina e avalia os quatro modelos em sequência. É retomável: pode ser interrompido e rodado de novo.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
MODELS=${MODELS:-"gemma4-e2b s5-nano minicpm-v4.7-1b nemotron-bolt-e4b"}
mkdir -p outputs/logs
for model in $MODELS; do
  if [ ! -d "outputs/$model/train/final_adapter" ]; then
    uv run python -m slm.train --model "$model" >> "outputs/logs/${model}_train.log" 2>&1
  fi
  for variant in base adapted; do
    if [ ! -f "outputs/$model/eval_${variant}.summary.json" ] || \
       [ "$(wc -l < "outputs/$model/eval_${variant}.jsonl")" -lt 2199 ]; then
      uv run python -m slm.evaluate --model "$model" --variant "$variant" >> "outputs/logs/${model}_eval_${variant}.log" 2>&1
    fi
  done
done
uv run python -m slm.compare --models $MODELS >> outputs/logs/compare.log 2>&1
uv run python -m slm.report >> outputs/logs/report.log 2>&1
