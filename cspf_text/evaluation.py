from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .attacks import apply_attack, context_truncation_attack
from .data import DatasetBundle, DocumentExample, SentenceExample


@dataclass
class ExperimentResult:
    name: str
    metrics: dict[str, float]
    details: dict


def compute_classification_metrics(labels, predictions, probabilities) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }
    if len(set(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return metrics


def sentence_level_metrics(pipeline, sentence_examples: list[SentenceExample], threshold: float = 0.5) -> dict[str, float]:
    if not sentence_examples:
        return {}
    labels = [example.label for example in sentence_examples]
    probabilities = pipeline.predict_sentence_examples_proba(sentence_examples)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return compute_classification_metrics(labels, predictions, probabilities)


def document_level_metrics(pipeline, bundle: DatasetBundle, threshold: float = 0.5) -> dict[str, float]:
    if not bundle.texts:
        return {}
    probabilities = pipeline.predict_proba(bundle.texts)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return compute_classification_metrics(bundle.labels, predictions, probabilities)


def coauthorship_metrics(pipeline, documents: list[DocumentExample], threshold: float = 0.5) -> dict[str, float]:
    if not documents:
        return {}

    sentence_accuracy = []
    ratio_mae = []
    boundary_errors = []
    for document in documents:
        if len(set(document.sentence_labels)) <= 1:
            continue
        prediction = pipeline.predict_document(document.text, threshold=threshold)
        predicted_labels = [int(prob >= threshold) for prob in prediction.sentence_probabilities]
        aligned_count = min(len(predicted_labels), len(document.sentence_labels))
        if aligned_count == 0:
            continue
        gold = np.asarray(document.sentence_labels[:aligned_count])
        pred = np.asarray(predicted_labels[:aligned_count])
        sentence_accuracy.append(float((gold == pred).mean()))

        gold_ratio = float(np.mean(gold))
        ratio_mae.append(abs(gold_ratio - prediction.ai_contribution_ratio))

        gold_boundaries = {idx for idx in range(1, aligned_count) if gold[idx] != gold[idx - 1]}
        pred_boundaries = {idx for idx in range(1, aligned_count) if pred[idx] != pred[idx - 1]}
        union = gold_boundaries | pred_boundaries
        if union:
            boundary_errors.append(1.0 - (len(gold_boundaries & pred_boundaries) / len(union)))
        else:
            boundary_errors.append(0.0)

    if not sentence_accuracy:
        return {}
    return {
        "coauthor_sentence_accuracy": float(np.mean(sentence_accuracy)),
        "coauthor_ratio_mae": float(np.mean(ratio_mae)),
        "coauthor_boundary_error": float(np.mean(boundary_errors)),
    }


def robustness_benchmark(
    pipeline,
    sentence_examples: list[SentenceExample],
    attacks: list[str],
    threshold: float = 0.5,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    if not sentence_examples:
        return {}

    results: dict[str, dict[str, float]] = {}
    base_labels = [example.label for example in sentence_examples]
    base_probabilities = pipeline.predict_sentence_examples_proba(sentence_examples)[:, 1]
    base_predictions = (base_probabilities >= threshold).astype(int)
    results["clean"] = compute_classification_metrics(base_labels, base_predictions, base_probabilities)

    for attack_name in attacks:
        attacked_examples = []
        for example in sentence_examples:
            if attack_name == "context_truncation":
                attacked_examples.append(context_truncation_attack(example))
                continue
            attack = apply_attack(example.text, attack_name, seed=seed)
            attacked_examples.append(
                SentenceExample(
                    text=attack.text,
                    label=example.label,
                    document_id=example.document_id,
                    sentence_id=example.sentence_id,
                    left_context=example.left_context,
                    right_context=example.right_context,
                    full_document=example.full_document,
                    source=example.source,
                    metadata={**example.metadata, "attack": attack.name, **attack.metadata},
                )
            )
        probabilities = pipeline.predict_sentence_examples_proba(attacked_examples)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        metrics = compute_classification_metrics(base_labels, predictions, probabilities)
        metrics["mean_probability_shift"] = float(np.mean(np.abs(probabilities - base_probabilities)))
        results[attack_name] = metrics
    return results


def permutation_feature_importance(pipeline, sentence_examples: list[SentenceExample], top_k: int = 15) -> list[dict[str, float | str]]:
    if not sentence_examples:
        return []

    X = np.asarray(pipeline.transform_sentence_examples(sentence_examples), dtype=float)
    labels = np.asarray([example.label for example in sentence_examples], dtype=int)
    baseline_probabilities = pipeline.predict_sentence_examples_proba(sentence_examples)[:, 1]
    baseline_loss = float(np.mean((baseline_probabilities - labels) ** 2))

    rng = np.random.default_rng(42)
    importances = []
    for feature_index, feature_name in enumerate(pipeline.feature_names_):
        shuffled = X.copy()
        shuffled[:, feature_index] = rng.permutation(shuffled[:, feature_index])
        shuffled_probabilities = pipeline.model.predict_proba(shuffled)
        shuffled_probabilities = pipeline._apply_calibration(shuffled_probabilities)[:, 1]
        shuffled_loss = float(np.mean((shuffled_probabilities - labels) ** 2))
        importances.append(
            {
                "feature": feature_name,
                "importance": shuffled_loss - baseline_loss,
            }
        )
    importances.sort(key=lambda item: float(item["importance"]), reverse=True)
    return importances[:top_k]


def sample_interpretability_report(pipeline, documents: list[DocumentExample], top_k: int = 3) -> list[dict]:
    reports: list[dict] = []
    for document in documents[:top_k]:
        prediction = pipeline.predict_document(document.text)
        suspicious_sentences = sorted(
            prediction.sentence_predictions,
            key=lambda item: item.probability,
            reverse=True,
        )[:3]
        reports.append(
            {
                "document_id": document.document_id,
                "document_probability": prediction.document_probability,
                "ai_contribution_ratio": prediction.ai_contribution_ratio,
                "top_document_features": [asdict(item) for item in prediction.top_document_features[:5]],
                "suspicious_sentences": [
                    {
                        "sentence_id": item.sentence_id,
                        "sentence": item.sentence,
                        "probability": item.probability,
                        "top_features": [asdict(feature) for feature in item.top_features[:5]],
                        "top_tokens": [asdict(token) for token in item.top_tokens[:5]],
                        "context_effects": item.context_effects,
                    }
                    for item in suspicious_sentences
                ],
            }
        )
    return reports


def save_experiment_outputs(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    try:
        import pandas as pd
    except ImportError:
        return

    flattened_rows = []
    for section, section_value in results.items():
        if isinstance(section_value, dict):
            for name, metrics in section_value.items():
                if isinstance(metrics, dict):
                    row = {"section": section, "name": name}
                    row.update(metrics)
                    flattened_rows.append(row)
    if flattened_rows:
        frame = pd.DataFrame(flattened_rows)
        frame.to_csv(output_dir / "results.csv", index=False)
        try:
            os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
            import matplotlib.pyplot as plt
        except ImportError:
            return

        numeric_columns = [column for column in frame.columns if column not in {"section", "name"}]
        for metric_name in numeric_columns[:6]:
            plot_frame = frame[["section", "name", metric_name]].dropna()
            if plot_frame.empty:
                continue
            plt.figure(figsize=(10, 4))
            labels = plot_frame["section"] + ":" + plot_frame["name"]
            plt.bar(labels, plot_frame[metric_name])
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.ylabel(metric_name)
            plt.savefig(output_dir / f"{metric_name}.png")
            plt.close()
