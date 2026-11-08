"""Gera e pontua o conjunto de teste C2 com o modelo base ou adaptado (retomável)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from slm.data import chat_messages, load_records, parse_card_set, tokenize_record
from slm.models import MODELS, RECIPE, load_quantized, load_tokenizer
from slm.train import RECORDS


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["concept_id"] for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--variant", choices=("base", "adapted"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = MODELS[args.model]
    output = args.output or Path("outputs") / spec.key / f"eval_{args.variant}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    tests = load_records(RECORDS)["test"][: args.limit]
    tokenizer = load_tokenizer(spec)
    model = load_quantized(spec)
    if args.variant == "adapted":
        from peft import PeftModel

        adapter = args.adapter or Path("outputs") / spec.key / "train" / "final_adapter"
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    completed = done_ids(output)
    with output.open("a") as handle:
        for index, record in enumerate(tests, start=1):
            if record.concept_id in completed:
                continue
            encoded = tokenize_record(record, tokenizer, max_length=RECIPE["max_length"])
            labels = torch.tensor([encoded["labels"]], device=model.device)
            with torch.inference_mode():
                loss = model(
                    input_ids=torch.tensor([encoded["input_ids"]], device=model.device),
                    attention_mask=torch.tensor([encoded["attention_mask"]], device=model.device),
                    labels=labels,
                ).loss
            prompt, _ = chat_messages(record)
            inputs = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(model.device)
            prompt_length = int(inputs["input_ids"].shape[-1])
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=RECIPE["max_new_tokens"],
                    do_sample=RECIPE["do_sample"],
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            new_tokens = generated[0][prompt_length:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            try:
                cards = [card.__dict__ for card in parse_card_set(prediction)]
                error = None
            except ValueError as exc:
                cards, error = [], str(exc)
            row = {
                "variant": args.variant,
                "concept_id": record.concept_id,
                "input_text": record.input_text,
                "reference_id": record.record_id,
                "prediction_text": prediction,
                "parse_valid": error is None,
                "parse_error": error,
                "cards": cards,
                "reference_nll": float(loss.float().item()),
                "reference_tokens": sum(label != -100 for label in encoded["labels"]),
                "generated_tokens": int(new_tokens.shape[-1]),
                "latency_seconds": latency,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 50 == 0:
                print(json.dumps({"model": spec.key, "variant": args.variant, "done": index, "total": len(tests)}), flush=True)
    summary = {
        "model": spec.key,
        "variant": args.variant,
        "test_concepts": len(tests),
        "completed": len(done_ids(output)),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
