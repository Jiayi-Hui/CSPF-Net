from __future__ import annotations

import re
from typing import Iterable, List


TOKEN_PATTERN = re.compile(r"\b\w+\b|[^\w\s]", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def simple_word_tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text or "")


def simple_sentence_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences = [chunk.strip() for chunk in SENTENCE_PATTERN.split(text) if chunk.strip()]
    return sentences or [text]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def flatten_feature_dicts(feature_dicts: Iterable[dict[str, float]]) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    for feature_dict in feature_dicts:
        for key, value in feature_dict.items():
            names.append(key)
            values.append(float(value))
    return names, values
