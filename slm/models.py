"""Registro dos modelos e receita QLoRA, igual à do Gemma 4 E2B no PictoInstruct."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    model_id: str
    revision: str
    text_prefix: str
    target_suffixes: tuple[str, ...]


GEMMA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "per_layer_input_gate",
    "per_layer_projection",
    "per_layer_model_projection",
)
QWEN35_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

MODELS = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "s5-nano",
            "S5-nano (Maritaca)",
            "maritaca-ai/S5-nano",
            "4aef89241cc24023d767e6edb0ae3c1ca8e8f6fb",
            "model.language_model.",
            QWEN35_TARGETS,
        ),
        ModelSpec(
            "gemma4-e2b",
            "Gemma 4 E2B",
            "google/gemma-4-E2B-it",
            "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "model.language_model.",
            GEMMA_TARGETS,
        ),
        ModelSpec(
            "minicpm-v4.7-1b",
            "MiniCPM-V 4.7 1B",
            "openbmb/MiniCPM-V-4.7-1B",
            "ad9bf379feb71204c888b22923d63322452407f9",
            "model.language_model.",
            QWEN35_TARGETS,
        ),
        ModelSpec(
            "nemotron-bolt-e4b",
            "Nemotron 3.5 Bolt E4B",
            "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16",
            "49843eed8db1f1c616c47af75cfd36e07aec0667",
            "model.language_model.",
            GEMMA_TARGETS,
        ),
    )
}

RECIPE = {
    "load_in_4bit": True,
    "quant_type": "nf4",
    "double_quantization": True,
    "compute_dtype": "bfloat16",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "scheduler": "cosine",
    "optimizer": "paged_adamw_8bit",
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "epochs": 1.0,
    "eval_steps": 250,
    "save_steps": 250,
    "logging_steps": 10,
    "save_total_limit": 2,
    "max_length": 384,
    "seed": 3407,
    "max_new_tokens": 512,
    "do_sample": False,
    "bootstrap_iterations": 10_000,
    "semantic_batch_size": 256,
    "encoders": {
        "serafim": "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder",
        "labse": "sentence-transformers/LaBSE",
    },
}


def load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


def load_quantized(spec: ModelSpec) -> Any:
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=RECIPE["quant_type"],
        bnb_4bit_use_double_quant=RECIPE["double_quantization"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    return AutoModelForImageTextToText.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        quantization_config=quantization,
    )


def lora_targets(model: Any, spec: ModelSpec) -> tuple[str, ...]:
    targets = tuple(
        sorted(
            name
            for name, _ in model.named_modules()
            if name.startswith(spec.text_prefix)
            and name.rsplit(".", 1)[-1] in spec.target_suffixes
        )
    )
    missing = set(spec.target_suffixes) - {name.rsplit(".", 1)[-1] for name in targets}
    if not targets or missing:
        raise RuntimeError(f"{spec.key}: LoRA target suffixes not found: {sorted(missing)}")
    return targets


def prepare_for_kbit_training(model: Any) -> Any:
    """Igual a peft.prepare_model_for_kbit_training, mas sem promover tabelas de embedding a fp32.

    A promoção bf16 -> fp32 não altera os valores congelados das tabelas, e as tabelas por
    camada do Gemma 4 (2,8 a 3,5 bilhões de parâmetros) não caberiam em fp32 numa GPU de 12 GB.
    """
    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    for parameter in model.parameters():
        parameter.requires_grad = False
        if (
            parameter.dtype in (torch.float16, torch.bfloat16)
            and parameter.__class__.__name__ != "Params4bit"
            and id(parameter) not in embedding_ids
        ):
            parameter.data = parameter.data.to(torch.float32)
    torch.cuda.empty_cache()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def attach_lora(model: Any, spec: ModelSpec) -> tuple[Any, tuple[str, ...], dict[str, float]]:
    from peft import LoraConfig, TaskType, get_peft_model

    model = prepare_for_kbit_training(model)
    targets = lora_targets(model, spec)
    model = get_peft_model(
        model,
        LoraConfig(
            r=RECIPE["lora_rank"],
            lora_alpha=RECIPE["lora_alpha"],
            lora_dropout=RECIPE["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(targets),
        ),
    )
    trainable = total = 0
    escaped = []
    text_marker = "." + spec.text_prefix.removeprefix("model.")
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
            if "lora_" not in name or text_marker not in name:
                escaped.append(name)
    if escaped:
        raise RuntimeError(f"trainable parameters escaped text-only LoRA: {escaped[:5]}")
    model.config.use_cache = False
    stats = {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }
    return model, targets, stats
