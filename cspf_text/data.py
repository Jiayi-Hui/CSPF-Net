from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import simple_sentence_split


DATASET_ALIASES = {
    "raid": "liamdugan/raid",
    "asap2": "kristyutsa/ASAP2",
    "asap_2": "kristyutsa/ASAP2",
    "ag_news": "ag_news",
    "agnews": "ag_news",
    "arxiv": "ccdv/arxiv-summarization",
    "arxiv_summarization": "ccdv/arxiv-summarization",
    "haco-det": "JiayiHui/HACoDet",
    "haco_det": "JiayiHui/HACoDet",
}

AI_LABEL = 1
HUMAN_LABEL = 0


@dataclass
class SpanAnnotation:
    start: int
    end: int
    label: int
    source: str = "span"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SentenceExample:
    text: str
    label: int
    document_id: str
    sentence_id: int
    left_context: str = ""
    right_context: str = ""
    full_document: str = ""
    source: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentExample:
    document_id: str
    text: str
    label: int
    sentences: list[str]
    sentence_labels: list[int]
    span_annotations: list[SpanAnnotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    texts: list[str]
    labels: list[int]
    rows: list[dict]
    documents: list[DocumentExample] = field(default_factory=list)
    sentence_examples: list[SentenceExample] = field(default_factory=list)


def _infer_text_column(row: dict) -> str:
    for candidate in (
        "text",
        "generation",
        "document",
        "content",
        "essay",
        "sentence",
        "full_text",
        "article",
    ):
        if candidate in row and isinstance(row[candidate], str):
            return candidate
    raise KeyError("Could not infer text column. Pass `text_column` explicitly.")


def _infer_label_column(row: dict) -> str:
    for candidate in (
        "label",
        "labels",
        "model",
        "source",
        "target",
        "generated",
        "is_ai",
        "__label__",
        "document_label",
    ):
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
            "1",
            "ai",
            "machine",
            "generated",
            "__label__ai",
            "__label__machine",
            "fake",
            "chatgpt",
            "gpt4",
            "gpt-4",
            "gpt3",
            "gpt-3",
            "gpt2",
            "llama",
            "llama-chat",
            "mistral",
            "mistral-chat",
            "mpt",
            "mpt-chat",
            "cohere",
            "cohere-chat",
            "synthetic",
            "machine_written",
            "ai_generated",
        }:
            return AI_LABEL
        if lowered in {"0", "human", "__label__human", "real", "authentic", "human_written"}:
            return HUMAN_LABEL
    raise ValueError(f"Unsupported label value: {value!r}")


def _normalize_label_list(values: Any) -> list[int] | None:
    if not isinstance(values, list):
        return None
    try:
        return [_normalize_label(value) for value in values]
    except ValueError:
        return None


def _coerce_annotation_label(annotation: dict[str, Any]) -> int | None:
    for key in ("label", "type", "source", "origin", "author"):
        if key not in annotation:
            continue
        try:
            return _normalize_label(annotation[key])
        except ValueError:
            continue
    return None


def _parse_span_annotations(row: dict) -> list[SpanAnnotation]:
    candidates = (
        "spans",
        "span_annotations",
        "annotations",
        "edits",
        "segments",
        "coauthor_spans",
    )
    annotations: list[SpanAnnotation] = []
    for candidate in candidates:
        raw_value = row.get(candidate)
        if not isinstance(raw_value, list):
            continue
        for item in raw_value:
            if not isinstance(item, dict):
                continue
            start = item.get("start", item.get("begin", item.get("char_start")))
            end = item.get("end", item.get("stop", item.get("char_end")))
            label = _coerce_annotation_label(item)
            if not isinstance(start, int) or not isinstance(end, int) or label is None:
                continue
            annotations.append(
                SpanAnnotation(
                    start=max(0, start),
                    end=max(start, end),
                    label=label,
                    source=candidate,
                    metadata=dict(item),
                )
            )
        if annotations:
            break
    return annotations


def _extract_explicit_sentences(row: dict) -> list[str] | None:
    for candidate in ("sentences", "sentence_texts", "sentence_list"):
        value = row.get(candidate)
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return [item.strip() for item in value if item and item.strip()]
    return None


def _extract_sentence_labels(row: dict, sentence_count: int) -> list[int] | None:
    for candidate in (
        "sentence_labels",
        "sentence_level_labels",
        "labels_per_sentence",
        "sent_labels",
        "coauthor_sentence_labels",
    ):
        labels = _normalize_label_list(row.get(candidate))
        if labels and len(labels) == sentence_count:
            return labels
    return None


def _map_spans_to_sentence_labels(
    text: str,
    sentences: list[str],
    document_label: int,
    annotations: list[SpanAnnotation],
) -> list[int]:
    if not sentences:
        return []
    labels = [document_label for _ in sentences]
    if not annotations:
        return labels

    cursor = 0
    sentence_offsets: list[tuple[int, int]] = []
    for sentence in sentences:
        start = text.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        sentence_offsets.append((start, end))
        cursor = end

    for idx, (sent_start, sent_end) in enumerate(sentence_offsets):
        ai_votes = 0
        human_votes = 0
        for annotation in annotations:
            overlap = max(0, min(sent_end, annotation.end) - max(sent_start, annotation.start))
            if overlap <= 0:
                continue
            if annotation.label == AI_LABEL:
                ai_votes += overlap
            else:
                human_votes += overlap
        if ai_votes or human_votes:
            labels[idx] = AI_LABEL if ai_votes >= human_votes else HUMAN_LABEL
    return labels


def _build_sentence_examples(
    document_id: str,
    text: str,
    sentences: list[str],
    sentence_labels: list[int],
    context_window: int,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> list[SentenceExample]:
    examples: list[SentenceExample] = []
    for sentence_id, sentence in enumerate(sentences):
        left_slice = sentences[max(0, sentence_id - context_window) : sentence_id]
        right_slice = sentences[sentence_id + 1 : sentence_id + 1 + context_window]
        examples.append(
            SentenceExample(
                text=sentence,
                label=sentence_labels[sentence_id],
                document_id=document_id,
                sentence_id=sentence_id,
                left_context=" ".join(left_slice).strip(),
                right_context=" ".join(right_slice).strip(),
                full_document=text,
                source=source,
                metadata=dict(metadata or {}),
            )
        )
    return examples


def _normalize_row_to_document(
    row: dict,
    index: int,
    text_column: str,
    label_column: str,
    dataset_name: str | None = None,
) -> DocumentExample | None:
    text = row.get(text_column)
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        document_label = _normalize_label(row.get(label_column))
    except ValueError:
        document_label = HUMAN_LABEL

    document_id = str(
        row.get("document_id")
        or row.get("id")
        or row.get("doc_id")
        or f"{dataset_name or 'doc'}-{index}"
    )
    sentences = _extract_explicit_sentences(row) or simple_sentence_split(text)
    sentence_labels = _extract_sentence_labels(row, len(sentences))
    span_annotations = _parse_span_annotations(row)
    if sentence_labels is None:
        sentence_labels = _map_spans_to_sentence_labels(text, sentences, document_label, span_annotations)

    metadata = {
        "dataset_name": dataset_name,
        "source_dataset": row.get("source_dataset", dataset_name),
        "raw_keys": sorted(row.keys()),
    }
    return DocumentExample(
        document_id=document_id,
        text=text,
        label=document_label,
        sentences=sentences,
        sentence_labels=sentence_labels,
        span_annotations=span_annotations,
        metadata=metadata,
    )


def normalize_text_dataset(
    rows: list[dict],
    text_column: str | None = None,
    label_column: str | None = None,
    context_window: int = 1,
    dataset_name: str | None = None,
) -> DatasetBundle:
    if not rows:
        return DatasetBundle(texts=[], labels=[], rows=[], documents=[], sentence_examples=[])

    text_column = text_column or _infer_text_column(rows[0])
    label_column = label_column or _infer_label_column(rows[0])

    texts: list[str] = []
    labels: list[int] = []
    normalized_rows: list[dict] = []
    documents: list[DocumentExample] = []
    sentence_examples: list[SentenceExample] = []

    for index, row in enumerate(rows):
        document = _normalize_row_to_document(
            row=row,
            index=index,
            text_column=text_column,
            label_column=label_column,
            dataset_name=dataset_name,
        )
        if document is None or not document.sentences:
            continue

        texts.append(document.text)
        labels.append(document.label)
        normalized_rows.append(row)
        documents.append(document)
        source = "coauthorship" if len(set(document.sentence_labels)) > 1 else "document"
        sentence_examples.extend(
            _build_sentence_examples(
                document_id=document.document_id,
                text=document.text,
                sentences=document.sentences,
                sentence_labels=document.sentence_labels,
                context_window=context_window,
                source=source,
                metadata=document.metadata,
            )
        )

    return DatasetBundle(
        texts=texts,
        labels=labels,
        rows=normalized_rows,
        documents=documents,
        sentence_examples=sentence_examples,
    )


def build_sentence_dataset(
    texts: list[str],
    labels: list[int],
    context_window: int = 1,
    document_prefix: str = "doc",
) -> list[SentenceExample]:
    examples: list[SentenceExample] = []
    for index, (text, label) in enumerate(zip(texts, labels)):
        sentences = simple_sentence_split(text)
        sentence_labels = [label for _ in sentences]
        examples.extend(
            _build_sentence_examples(
                document_id=f"{document_prefix}-{index}",
                text=text,
                sentences=sentences,
                sentence_labels=sentence_labels,
                context_window=context_window,
                source="document",
            )
        )
    return examples


def load_text_dataset(
    name: str,
    split: str = "train",
    dataset_id: str | None = None,
    text_column: str | None = None,
    label_column: str | None = None,
    sample_size: int | None = None,
    cache_dir: str | None = None,
    context_window: int = 1,
    local_files_only: bool = False,
) -> DatasetBundle:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("data.py requires `datasets`.") from exc

    resolved_id = dataset_id or DATASET_ALIASES.get(name.lower(), name)
    path = Path(resolved_id)
    if path.exists():
        suffix = path.suffix.lower()
        if path.is_dir():
            data_files = {}
            for candidate_split in {split, "train", "validation", "test"}:
                for candidate_suffix in (".json", ".jsonl", ".csv", ".parquet"):
                    candidate = path / f"{candidate_split}{candidate_suffix}"
                    if candidate.exists():
                        data_files[candidate_split] = str(candidate)
                        break
            if not data_files:
                raise FileNotFoundError(f"No supported dataset files found in local directory: {path}")
            format_name = Path(next(iter(data_files.values()))).suffix.lower().lstrip(".")
            if format_name == "jsonl":
                format_name = "json"
            dataset = load_dataset(format_name, data_files=data_files, split=split, cache_dir=cache_dir)
        elif suffix in {".csv", ".json", ".jsonl", ".parquet"}:
            format_name = suffix.lstrip(".")
            if format_name == "jsonl":
                format_name = "json"
            dataset = load_dataset(format_name, data_files=str(path), split="train", cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported local dataset format: {path}")
    else:
        dataset = load_dataset(
            resolved_id,
            split=split,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    rows = [dict(row) for row in dataset]
    return normalize_text_dataset(
        rows,
        text_column=text_column,
        label_column=label_column,
        context_window=context_window,
        dataset_name=name,
    )


def _load_stream_rows(
    dataset_id: str,
    split: str,
    sample_size: int | None,
    cache_dir: str | None,
    seed: int,
    predicate=None,
    text_key: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset
    from datasets import config as datasets_config

    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache_path)
        os.environ["HF_HOME"] = str(cache_path)
        os.environ["HF_DATASETS_CACHE"] = str(cache_path / "datasets")
        os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "hub")
        datasets_config.HF_DATASETS_CACHE = str(cache_path / "datasets")
        datasets_config.DOWNLOADED_DATASETS_PATH = Path(cache_path / "datasets" / "downloads")

    dataset = load_dataset(
        dataset_id,
        split=split,
        trust_remote_code=True,
        cache_dir=cache_dir,
        streaming=True,
    )
    if sample_size is not None:
        dataset = dataset.shuffle(seed=seed, buffer_size=max(sample_size * 2, 10_000))

    rows: list[dict[str, Any]] = []
    for row in dataset:
        row = dict(row)
        if predicate is not None and not predicate(row):
            continue
        if text_key is not None:
            text = row.get(text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            row["text"] = text
        if extra_metadata:
            row.update(extra_metadata)
        rows.append(row)
        if sample_size is not None and len(rows) >= sample_size:
            break
    return rows


def load_mixed_source_dataset(
    cache_dir: str | None = None,
    context_window: int = 1,
    seed: int = 42,
    raid_split: str = "train",
    raid_sample_size: int = 36_000,
    asap2_split: str = "train",
    asap2_sample_size: int | None = None,
    ag_news_split: str = "train",
    ag_news_sample_size: int = 9_000,
    arxiv_split: str = "train",
    arxiv_sample_size: int = 9_000,
) -> DatasetBundle:
    rng = random.Random(seed)

    raid_rows = _load_stream_rows(
        dataset_id=DATASET_ALIASES["raid"],
        split=raid_split,
        sample_size=raid_sample_size,
        cache_dir=cache_dir,
        seed=seed,
        predicate=lambda row: row.get("model") not in {None, "human"} and row.get("attack", "none") == "none",
        text_key="generation",
        extra_metadata={"label": AI_LABEL, "source_dataset": "raid", "document_label": AI_LABEL},
    )
    for index, row in enumerate(raid_rows):
        row["document_id"] = f"raid-{index}"

    asap2_rows = _load_stream_rows(
        dataset_id=DATASET_ALIASES["asap2"],
        split=asap2_split,
        sample_size=asap2_sample_size,
        cache_dir=cache_dir,
        seed=seed + 1,
        text_key="sentence",
        extra_metadata={"label": HUMAN_LABEL, "source_dataset": "asap2", "document_label": HUMAN_LABEL},
    )
    for index, row in enumerate(asap2_rows):
        row["document_id"] = f"asap2-{index}"

    ag_news_rows = _load_stream_rows(
        dataset_id=DATASET_ALIASES["ag_news"],
        split=ag_news_split,
        sample_size=ag_news_sample_size,
        cache_dir=cache_dir,
        seed=seed + 2,
        text_key="text",
        extra_metadata={"label": HUMAN_LABEL, "source_dataset": "ag_news", "document_label": HUMAN_LABEL},
    )
    for index, row in enumerate(ag_news_rows):
        row["document_id"] = f"agnews-{index}"

    arxiv_rows = _load_stream_rows(
        dataset_id=DATASET_ALIASES["arxiv"],
        split=arxiv_split,
        sample_size=arxiv_sample_size,
        cache_dir=cache_dir,
        seed=seed + 3,
        text_key="abstract",
        extra_metadata={"label": HUMAN_LABEL, "source_dataset": "arxiv", "document_label": HUMAN_LABEL},
    )
    for index, row in enumerate(arxiv_rows):
        row["document_id"] = f"arxiv-{index}"

    rows = raid_rows + asap2_rows + ag_news_rows + arxiv_rows
    rng.shuffle(rows)
    return normalize_text_dataset(
        rows,
        text_column="text",
        label_column="label",
        context_window=context_window,
        dataset_name="mixed_binary",
    )
