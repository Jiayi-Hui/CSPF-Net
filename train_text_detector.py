from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

from cspf_text import CSPFTextPipeline, load_text_dataset
from cspf_text.evaluation import (
    coauthorship_metrics,
    document_level_metrics,
    permutation_feature_importance,
    robustness_benchmark,
    sample_interpretability_report,
    save_experiment_outputs,
    sentence_level_metrics,
)
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
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--disable-context", action="store_true")
    parser.add_argument("--train-granularity", choices=("sentence", "document"), default="sentence")
    parser.add_argument("--calibration-split", default=None)
    parser.add_argument("--calibration-sample-size", type=int, default=None)
    parser.add_argument(
        "--robustness-attacks",
        default="synonym_substitution,punctuation_strip,sentence_shuffle,character_noise,context_truncation",
    )
    parser.add_argument("--interpretability-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace, train_texts: list[str]) -> CSPFTextPipeline:
    pipeline = CSPFTextPipeline(
        probabilistic_extractor=ProbabilisticFeatureExtractor(
            model_name=args.prob_model_name,
            batch_size=args.prob_batch_size,
            device=args.device,
            cache_dir=args.hf_cache_dir,
            local_files_only=args.local_files_only,
        ),
        context_window=args.context_window,
        use_context=not args.disable_context,
    )
    if args.model_type == "mlp":
        feature_dim = len(pipeline.transform(train_texts[:1] or ["placeholder"])[0])
        pipeline.model = TorchMLPClassifier(input_dim=feature_dim, device=args.device)
    return pipeline


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
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
        context_window=args.context_window,
        local_files_only=args.local_files_only,
    )

    pipeline = build_pipeline(args, train_bundle.texts)
    if args.train_granularity == "sentence":
        pipeline.fit_sentence_examples(train_bundle.sentence_examples)
    else:
        pipeline.fit(train_bundle.texts, train_bundle.labels)

    if args.calibration_split:
        calibration_bundle = load_text_dataset(
            name=args.dataset,
            dataset_id=args.dataset_id,
            split=args.calibration_split,
            text_column=args.text_column,
            label_column=args.label_column,
            sample_size=args.calibration_sample_size,
            cache_dir=args.hf_cache_dir,
            context_window=args.context_window,
            local_files_only=args.local_files_only,
        )
        pipeline.calibrate(calibration_bundle.sentence_examples)

    metrics: dict[str, dict[str, float]] = {}
    robustness_results: dict[str, dict[str, float]] = {}
    interpretability_report = []
    feature_importance = []
    coauthor_results: dict[str, float] = {}
    if args.eval_split:
        eval_bundle = load_text_dataset(
            name=args.dataset,
            dataset_id=args.dataset_id,
            split=args.eval_split,
            text_column=args.text_column,
            label_column=args.label_column,
            sample_size=args.eval_sample_size,
            cache_dir=args.hf_cache_dir,
            context_window=args.context_window,
            local_files_only=args.local_files_only,
        )
        metrics = {
            "document": document_level_metrics(pipeline, eval_bundle, threshold=args.document_threshold),
            "sentence": sentence_level_metrics(pipeline, eval_bundle.sentence_examples, threshold=args.document_threshold),
        }
        coauthor_results = coauthorship_metrics(pipeline, eval_bundle.documents, threshold=args.document_threshold)
        attacks = [attack.strip() for attack in args.robustness_attacks.split(",") if attack.strip()]
        robustness_results = robustness_benchmark(
            pipeline,
            eval_bundle.sentence_examples,
            attacks=attacks,
            threshold=args.document_threshold,
            seed=args.seed,
        )
        interpretability_report = sample_interpretability_report(
            pipeline,
            eval_bundle.documents,
            top_k=args.interpretability_samples,
        )
        feature_importance = permutation_feature_importance(
            pipeline,
            eval_bundle.sentence_examples[: min(512, len(eval_bundle.sentence_examples))],
        )

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
        "context_window": args.context_window,
        "use_context": not args.disable_context,
        "train_granularity": args.train_granularity,
        "seed": args.seed,
        "hf_cache_dir": args.hf_cache_dir,
        "local_files_only": args.local_files_only,
        "feature_count": len(pipeline.feature_names_),
        "feature_names": pipeline.feature_names_,
        "resolved_prob_device": pipeline.probabilistic_extractor.device,
        "resolved_model_device": getattr(pipeline.model, "device", None),
        "sentence_example_count": len(train_bundle.sentence_examples),
        "metrics": metrics,
        "coauthorship_metrics": coauthor_results,
        "robustness": robustness_results,
        "feature_importance": feature_importance,
        "interpretability_report": interpretability_report,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    save_experiment_outputs(output_dir, metadata)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
