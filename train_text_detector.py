from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cspf_text import CSPFTextPipeline, configure_hf_cache, load_text_dataset
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
from cspf_text.run_tracking import RunTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CSPF-Net text detector.")
    parser.add_argument("--dataset", default="hc3_reborn", help="Dataset alias or Hugging Face dataset id.")
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
    args.hf_cache_dir = configure_hf_cache(args.hf_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = RunTracker(output_dir)
    tracker.save_json("run_config.json", vars(args), artifact_key="run_config")
    tracker.log("Initialized training run.", stage="init", extra={"output_dir": str(output_dir)})

    try:
        tracker.set_stage("load_train_data")
        tracker.log("Loading training dataset.", stage="load_train_data")
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
        train_summary = {
            "text_count": len(train_bundle.texts),
            "label_count": len(train_bundle.labels),
            "document_count": len(train_bundle.documents),
            "sentence_example_count": len(train_bundle.sentence_examples),
        }
        tracker.save_json("train_bundle_summary.json", train_summary, artifact_key="train_bundle_summary")
        tracker.complete_stage("load_train_data", extra={"train_bundle_summary": train_summary})
        tracker.log("Loaded training dataset.", stage="load_train_data", extra=train_summary)

        tracker.set_stage("build_pipeline")
        tracker.log("Building pipeline.", stage="build_pipeline")
        pipeline = build_pipeline(args, train_bundle.texts)
        tracker.complete_stage("build_pipeline")

        tracker.set_stage("fit")
        tracker.log("Fitting model.", stage="fit", extra={"train_granularity": args.train_granularity})
        if args.train_granularity == "sentence":
            pipeline.fit_sentence_examples(train_bundle.sentence_examples)
        else:
            pipeline.fit(train_bundle.texts, train_bundle.labels)
        tracker.save_pickle("pipeline_after_fit.pkl", pipeline, artifact_key="pipeline_after_fit")
        tracker.complete_stage("fit")
        tracker.log("Finished fitting model.", stage="fit")

        calibration_count = 0
        if args.calibration_split:
            tracker.set_stage("calibration")
            tracker.log("Loading calibration data.", stage="calibration")
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
            calibration_count = len(calibration_bundle.sentence_examples)
            tracker.save_json(
                "calibration_bundle_summary.json",
                {
                    "document_count": len(calibration_bundle.documents),
                    "sentence_example_count": calibration_count,
                },
                artifact_key="calibration_bundle_summary",
            )
            pipeline.calibrate(calibration_bundle.sentence_examples)
            tracker.save_pickle("pipeline_after_calibration.pkl", pipeline, artifact_key="pipeline_after_calibration")
            tracker.complete_stage("calibration", extra={"calibration_sentence_count": calibration_count})
            tracker.log("Finished calibration.", stage="calibration", extra={"sentence_example_count": calibration_count})

        metrics: dict[str, dict[str, float]] = {}
        robustness_results: dict[str, dict[str, float]] = {}
        interpretability_report = []
        feature_importance = []
        coauthor_results: dict[str, float] = {}
        eval_summary: dict[str, int] = {}
        if args.eval_split:
            tracker.set_stage("evaluation")
            tracker.log("Loading evaluation data.", stage="evaluation")
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
            eval_summary = {
                "document_count": len(eval_bundle.documents),
                "sentence_example_count": len(eval_bundle.sentence_examples),
            }
            tracker.save_json("eval_bundle_summary.json", eval_summary, artifact_key="eval_bundle_summary")
            tracker.log("Running evaluation metrics.", stage="evaluation", extra=eval_summary)
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
            feature_importance = permutation_feature_importance(
                pipeline,
                eval_bundle.sentence_examples[: min(512, len(eval_bundle.sentence_examples))],
            )
            interpretability_report = sample_interpretability_report(
                pipeline,
                eval_bundle.documents,
                top_k=args.interpretability_samples,
            )
            tracker.save_json(
                "intermediate_metrics.json",
                {
                    "metrics": metrics,
                    "coauthorship_metrics": coauthor_results,
                    "robustness": robustness_results,
                    "feature_importance": feature_importance,
                },
                artifact_key="intermediate_metrics",
            )
            tracker.complete_stage("evaluation", extra={"eval_bundle_summary": eval_summary})
            tracker.log("Finished evaluation.", stage="evaluation")

        tracker.set_stage("finalize")
        tracker.log("Saving final pipeline and metadata.", stage="finalize")
        tracker.save_pickle("pipeline.pkl", pipeline, artifact_key="pipeline_final")

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
            "train_bundle_summary": train_summary,
            "calibration_sentence_count": calibration_count,
            "eval_bundle_summary": eval_summary,
            "metrics": metrics,
            "coauthorship_metrics": coauthor_results,
            "robustness": robustness_results,
            "feature_importance": feature_importance,
            "interpretability_report": interpretability_report,
        }
        tracker.save_json("metadata.json", metadata, artifact_key="metadata")
        save_experiment_outputs(output_dir, metadata)
        tracker.complete_stage("finalize")
        tracker.finalize("completed")
        tracker.log("Run completed successfully.", stage="finalize")
        print(json.dumps(metadata, indent=2))
    except Exception as exc:
        tracker.log(f"Run failed: {exc}", stage=tracker.state.get("current_stage", "unknown"))
        tracker.finalize(
            "failed",
            extra={
                "failed_stage": tracker.state.get("current_stage"),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise


if __name__ == "__main__":
    main()
