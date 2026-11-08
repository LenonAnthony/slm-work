"""Formato da tarefa, porte direto de pictoinstruct.training.data (renderer v1)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import emoji as emoji_lib
import ftfy
import orjson
import regex

SYSTEM_INSTRUCTION = """\
Você cria cartões textuais de Comunicação Aumentativa e Alternativa em português brasileiro.
Para o conceito informado, produza exatamente cinco linhas. Cada linha deve conter, nesta ordem,
um texto curto, uma forma falada mais completa e um único emoji, separados por vírgulas. Use
linguagem simples, direta e adequada a situações cotidianas. As cinco linhas devem representar
situações diferentes. Não inclua numeração, cabeçalhos, comentários ou explicações.
"""
SPLITS = ("optimization", "development", "test")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Record:
    record_id: str
    concept_id: str
    split: str
    condition: str
    input_text: str
    target_text: str


@dataclass(frozen=True)
class Card:
    short_text: str
    spoken_text: str
    emoji: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(orjson.dumps(value, option=orjson.OPT_SORT_KEYS)).hexdigest()
    return f"{namespace}:{digest}"


def normalized_key(value: str) -> str:
    text = unicodedata.normalize("NFC", ftfy.fix_text(value))
    return WHITESPACE.sub(" ", text).strip().casefold()


def is_single_emoji_grapheme(value: str) -> bool:
    return len(regex.findall(r"\X", value)) == 1 and emoji_lib.is_emoji(value)


def concept_label(input_text: str) -> str:
    prefix, separator, value = input_text.partition(":")
    if not separator or prefix.strip().casefold() != "input" or not value.strip():
        raise ValueError(f"unsupported input: {input_text!r}")
    return value.strip()


def _card_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].casefold().startswith("output:"):
        lines[0] = lines[0].split(":", 1)[1].strip()
        if not lines[0]:
            lines.pop(0)
    return lines


def parse_card_set(text: str) -> tuple[Card, ...]:
    lines = _card_lines(text)
    if len(lines) != 5:
        raise ValueError("the output must contain exactly five non-empty card lines")
    cards = []
    for line in lines:
        parts = [part.strip() for part in line.rsplit(",", 2)]
        if len(parts) != 3 or not parts[0] or not parts[1]:
            raise ValueError("each card line must contain three comma-separated fields")
        if not is_single_emoji_grapheme(parts[2]):
            raise ValueError("each card line must end with exactly one emoji grapheme")
        cards.append(Card(*parts))
    return tuple(cards)


def normalized_target(target_text: str) -> str:
    normalized = "\n".join(_card_lines(target_text))
    parse_card_set(normalized)
    return normalized


def load_records(path: Path) -> dict[str, list[Record]]:
    by_split: dict[str, list[Record]] = defaultdict(list)
    for line in path.read_bytes().splitlines():
        if line.strip():
            record = Record(**orjson.loads(line))
            by_split[record.split].append(record)
    for split in SPLITS:
        by_split[split].sort(key=lambda item: item.concept_id)
    return dict(by_split)


def chat_messages(record: Record) -> tuple[list[dict[str, str]], str]:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Conceito: {concept_label(record.input_text)}"},
    ]
    return messages, normalized_target(record.target_text)


def _ids(encoded: Any) -> list[int]:
    value = encoded["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def tokenize_record(record: Record, tokenizer: Any, *, max_length: int) -> dict[str, list[int]]:
    prompt, completion = chat_messages(record)
    kwargs = {"tokenize": True, "return_dict": True, "enable_thinking": False}
    prompt_ids = _ids(tokenizer.apply_chat_template(prompt, add_generation_prompt=True, **kwargs))
    full_ids = _ids(
        tokenizer.apply_chat_template(
            [*prompt, {"role": "assistant", "content": completion}],
            add_generation_prompt=False,
            **kwargs,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template does not preserve the prompt as a token prefix")
    if len(full_ids) > max_length:
        raise ValueError(f"{record.record_id} needs {len(full_ids)} tokens > {max_length}")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
    }
