from __future__ import annotations

import random
import string
from dataclasses import dataclass

from .data import SentenceExample
from .utils import simple_sentence_split, simple_word_tokenize


SYNONYM_MAP = {
    "important": "crucial",
    "good": "strong",
    "bad": "poor",
    "big": "large",
    "small": "minor",
    "show": "demonstrate",
    "use": "employ",
    "help": "support",
    "make": "produce",
    "change": "alter",
    "problem": "issue",
    "result": "outcome",
    "student": "learner",
    "essay": "article",
    "model": "system",
}


@dataclass
class AttackResult:
    name: str
    text: str
    metadata: dict[str, float | int | str]


def _join_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return text.strip()


def synonym_substitution_attack(text: str, max_changes: int = 3) -> AttackResult:
    tokens = simple_word_tokenize(text)
    changed = 0
    output: list[str] = []
    for token in tokens:
        key = token.lower()
        replacement = SYNONYM_MAP.get(key)
        if replacement and changed < max_changes:
            output.append(replacement.capitalize() if token[:1].isupper() else replacement)
            changed += 1
        else:
            output.append(token)
    return AttackResult("synonym_substitution", _join_tokens(output), {"changes": changed})


def punctuation_strip_attack(text: str) -> AttackResult:
    stripped = "".join(ch for ch in text if ch not in string.punctuation)
    return AttackResult("punctuation_strip", stripped, {"removed_chars": len(text) - len(stripped)})


def sentence_shuffle_attack(text: str, seed: int = 42) -> AttackResult:
    sentences = simple_sentence_split(text)
    if len(sentences) <= 1:
        return AttackResult("sentence_shuffle", text, {"shuffled": 0})
    rng = random.Random(seed)
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    return AttackResult("sentence_shuffle", " ".join(shuffled), {"shuffled": len(sentences)})


def character_noise_attack(text: str, swap_rate: float = 0.05, seed: int = 42) -> AttackResult:
    rng = random.Random(seed)
    chars = list(text)
    swaps = 0
    for idx in range(len(chars) - 1):
        if chars[idx].isspace() or chars[idx + 1].isspace():
            continue
        if rng.random() < swap_rate:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            swaps += 1
    return AttackResult("character_noise", "".join(chars), {"swaps": swaps})


def context_truncation_attack(example: SentenceExample) -> SentenceExample:
    return SentenceExample(
        text=example.text,
        label=example.label,
        document_id=example.document_id,
        sentence_id=example.sentence_id,
        left_context="",
        right_context="",
        full_document=example.full_document,
        source=example.source,
        metadata=dict(example.metadata),
    )


def apply_attack(text: str, attack_name: str, seed: int = 42) -> AttackResult:
    attack_name = attack_name.lower()
    if attack_name == "synonym_substitution":
        return synonym_substitution_attack(text)
    if attack_name == "punctuation_strip":
        return punctuation_strip_attack(text)
    if attack_name == "sentence_shuffle":
        return sentence_shuffle_attack(text, seed=seed)
    if attack_name == "character_noise":
        return character_noise_attack(text, seed=seed)
    raise ValueError(f"Unknown attack: {attack_name}")
