from __future__ import annotations

from dataclasses import dataclass


DATASET_ALIASES = {
    "raid": "liamdugan/raid",
    "asap2": "kristyutsa/ASAP2",
    "asap_2": "kristyutsa/ASAP2",
}


@dataclass
class DatasetBundle:
    texts: list[str]
    labels: list[int]
    rows: list[dict]


def _infer_text_column(row: dict) -> str:
    for candidate in ("text", "generation", "document", "content", "essay", "sentence", "full_text"):
        if candidate in row and isinstance(row[candidate], str):
            return candidate
    raise KeyError("Could not infer text column. Pass `text_column` explicitly.")


def _infer_label_column(row: dict) -> str:
    for candidate in ("label", "labels", "model", "source", "target", "generated", "is_ai", "__label__"):
        if candidate in row:
            return candidate
    raise KeyError("Could not infer label column. Pass `label_column` explicitly.")


def _normalize_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {
            "1", "ai", "machine", "generated", "__label__ai", "__label__machine", "fake",
            "chatgpt", "gpt4", "gpt-4", "gpt3", "gpt-3", "gpt2", "llama-chat", "mistral",
            "mistral-chat", "mpt", "mpt-chat", "cohere", "cohere-chat",
        }:
            return 1
        if lowered in {"0", "human", "__label__human", "real"}:
            return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def normalize_text_dataset(rows: list[dict], text_column: str | None = None, label_column: str | None = None) -> DatasetBundle:
    if not rows:
        return DatasetBundle(texts=[], labels=[], rows=[])

    text_column = text_column or _infer_text_column(rows[0])
    label_column = label_column or _infer_label_column(rows[0])

    texts: list[str] = []
    labels: list[int] = []
    normalized_rows: list[dict] = []
    for row in rows:
        text = row.get(text_column)
        label = row.get(label_column)
        if not isinstance(text, str):
            continue
        try:
            normalized_label = _normalize_label(label)
        except ValueError:
            continue
        texts.append(text)
        labels.append(normalized_label)
        normalized_rows.append(row)

    return DatasetBundle(texts=texts, labels=labels, rows=normalized_rows)


def load_text_dataset(
    name: str,
    split: str = "train",
    dataset_id: str | None = None,
    text_column: str | None = None,
    label_column: str | None = None,
    sample_size: int | None = None,
    cache_dir: str | None = None,
) -> DatasetBundle:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("data.py requires `datasets`.") from exc

    resolved_id = dataset_id or DATASET_ALIASES.get(name.lower(), name)
    dataset = load_dataset(
        resolved_id,
        split=split,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    rows = [dict(row) for row in dataset]
    return normalize_text_dataset(rows, text_column=text_column, label_column=label_column)
