"""Fine-tuning QLoRA de um modelo do registro no dataset C2."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from slm.data import load_records, sha256_file, tokenize_record
from slm.models import MODELS, RECIPE, attach_lora, load_quantized, load_tokenizer

RECORDS = Path("data/c2/training_examples.jsonl")


def versions() -> dict[str, str]:
    names = ("accelerate", "bitsandbytes", "datasets", "peft", "torch", "transformers", "trl")
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = MODELS[args.model]
    output_dir = args.output or Path("outputs") / spec.key / "train"
    records = load_records(RECORDS)
    train_records = records["optimization"][: args.train_limit]
    validation_records = records["development"][: args.train_limit]
    tokenizer = load_tokenizer(spec)
    max_length = RECIPE["max_length"]
    train = Dataset.from_list([tokenize_record(r, tokenizer, max_length=max_length) for r in train_records])
    validation = Dataset.from_list(
        [tokenize_record(r, tokenizer, max_length=max_length) for r in validation_records]
    )
    lengths = [len(row) for row in train["input_ids"]]

    model, targets, stats = attach_lora(load_quantized(spec), spec)
    steps_per_epoch = math.ceil(
        len(train) / (RECIPE["per_device_batch_size"] * RECIPE["gradient_accumulation_steps"])
    )
    planned_steps = args.max_steps or math.ceil(steps_per_epoch * RECIPE["epochs"])
    eval_steps = min(RECIPE["eval_steps"], planned_steps)
    manifest = {
        "model_key": spec.key,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "records_sha256": sha256_file(RECORDS),
        "train_records": len(train),
        "validation_records": len(validation),
        "max_train_tokens": max(lengths),
        "recipe": RECIPE,
        "lora_target_suffixes": sorted({t.rsplit(".", 1)[-1] for t in targets}),
        "lora_target_modules": len(targets),
        **stats,
        "planned_optimizer_steps": planned_steps,
        "packages": versions(),
        "gpu": torch.cuda.get_device_name(0),
        "test_used_during_training": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=RECIPE["epochs"],
            max_steps=args.max_steps or -1,
            learning_rate=RECIPE["learning_rate"],
            weight_decay=RECIPE["weight_decay"],
            warmup_steps=math.ceil(planned_steps * RECIPE["warmup_ratio"]),
            per_device_train_batch_size=RECIPE["per_device_batch_size"],
            per_device_eval_batch_size=RECIPE["per_device_batch_size"],
            gradient_accumulation_steps=RECIPE["gradient_accumulation_steps"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=eval_steps,
            logging_steps=RECIPE["logging_steps"],
            save_total_limit=RECIPE["save_total_limit"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim=RECIPE["optimizer"],
            lr_scheduler_type=RECIPE["scheduler"],
            bf16=True,
            tf32=True,
            seed=RECIPE["seed"],
            data_seed=RECIPE["seed"],
            max_length=max_length,
            packing=False,
            completion_only_loss=True,
            report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
        train_dataset=train,
        eval_dataset=validation,
        processing_class=tokenizer,
    )
    checkpoint = get_last_checkpoint(str(output_dir))
    result = trainer.train(resume_from_checkpoint=checkpoint)
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    manifest.update(
        {
            "status": "completed",
            "resumed_from": checkpoint,
            "train_metrics": result.metrics,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric,
            "log_history": trainer.state.log_history,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
