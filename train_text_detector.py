from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from cspf_text import CSPFTextPipeline, load_text_dataset
from cspf_text.modeling import TorchMLPClassifier
from cspf_text.features import ProbabilisticFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CSPF-Net text detector.")
    parser.add_argument("--dataset", default="raid", help="Dataset alias or Hugging Face dataset id.")
    parser.add_argument("--dataset-id", default=None, help="Override Hugging Face dataset id.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--eval-sample-size", type=int, default=None)
    parser.add_argument("--model-type", choices=("stacking", "mlp"), default="stacking")
    parser.add_argument("--prob-model-name", default="gpt2")
    parser.add_argument("--device", default=None, help="Computation device, e.g. cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--prob-batch-size", type=int, default=8)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/text_detector")
    parser.add_argument("--document-threshold", type=float, default=0.5)
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace, train_texts: list[str]) -> CSPFTextPipeline:
    pipeline = CSPFTextPipeline(
        probabilistic_extractor=ProbabilisticFeatureExtractor(
            model_name=args.prob_model_name,
            batch_size=args.prob_batch_size,
            device=args.device,
            cache_dir=args.hf_cache_dir,
            local_files_only=args.local_files_only,
        )
    )
    if args.model_type == "mlp":
        feature_dim = len(pipeline.transform(train_texts[:1] or ["placeholder"])[0])
        pipeline.model = TorchMLPClassifier(input_dim=feature_dim, device=args.device)
    return pipeline


def compute_metrics(labels, predictions, probabilities) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }
    if len(set(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_bundle = load_text_dataset(
        name=args.dataset,
        dataset_id=args.dataset_id,
        split=args.train_split,
        text_column=args.text_column,
        label_column=args.label_column,
        sample_size=args.sample_size,
        cache_dir=args.hf_cache_dir,
    )

    pipeline = build_pipeline(args, train_bundle.texts)
    pipeline.fit(train_bundle.texts, train_bundle.labels)

    metrics: dict[str, float] = {}
    if args.eval_split:
        eval_bundle = load_text_dataset(
            name=args.dataset,
            dataset_id=args.dataset_id,
            split=args.eval_split,
            text_column=args.text_column,
            label_column=args.label_column,
            sample_size=args.eval_sample_size,
            cache_dir=args.hf_cache_dir,
        )
        probabilities = pipeline.predict_proba(eval_bundle.texts)[:, 1]
        predictions = (probabilities >= args.document_threshold).astype(int)
        metrics = compute_metrics(eval_bundle.labels, predictions, probabilities)

    with open(output_dir / "pipeline.pkl", "wb") as handle:
        pickle.dump(pipeline, handle)

    metadata = {
        "dataset": args.dataset,
        "dataset_id": args.dataset_id,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "sample_size": args.sample_size,
        "eval_sample_size": args.eval_sample_size,
        "model_type": args.model_type,
        "prob_model_name": args.prob_model_name,
        "device": args.device,
        "prob_batch_size": args.prob_batch_size,
        "hf_cache_dir": args.hf_cache_dir,
        "local_files_only": args.local_files_only,
        "feature_count": len(pipeline.feature_names_),
        "feature_names": pipeline.feature_names_,
        "resolved_prob_device": pipeline.probabilistic_extractor.device,
        "resolved_model_device": getattr(pipeline.model, "device", None),
        "metrics": metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
