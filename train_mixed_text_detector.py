from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cspf_text import CSPFTextPipeline, configure_hf_cache, load_mixed_source_dataset
from cspf_text.evaluation import (
    permutation_feature_importance,
    robustness_benchmark,
    sample_interpretability_report,
    save_experiment_outputs,
    sentence_level_metrics,
)
from cspf_text.features import ProbabilisticFeatureExtractor
from cspf_text.modeling import TorchMLPClassifier
from cspf_text.run_tracking import RunTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mixed-source binary text detector.")
    parser.add_argument("--hc3-reborn-sample-size", type=int, default=36000)
    parser.add_argument("--hc3-reborn-dataset-id", default=None)
    parser.add_argument("--asap2-sample-size", type=int, default=None)
    parser.add_argument("--ag-news-sample-size", type=int, default=9000)
    parser.add_argument("--arxiv-sample-size", type=int, default=9000)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--prob-model-name", default="gpt2")
    parser.add_argument("--prob-batch-size", type=int, default=8)
    parser.add_argument("--model-type", choices=("stacking", "mlp"), default="stacking")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--output-dir", default="/workspace/CSPF-Net/artifacts/mixed_text_detector")
    parser.add_argument("--eval-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--document-threshold", type=float, default=0.5)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--robustness-attacks",
        default="synonym_substitution,punctuation_strip,sentence_shuffle,character_noise,context_truncation",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_pipeline(args: argparse.Namespace, warmup_examples) -> CSPFTextPipeline:
    pipeline = CSPFTextPipeline(
        probabilistic_extractor=ProbabilisticFeatureExtractor(
            model_name=args.prob_model_name,
            batch_size=args.prob_batch_size,
            device=args.device,
            cache_dir=args.hf_cache_dir,
        ),
        context_window=args.context_window,
    )
    if args.model_type == "mlp":
        feature_dim = len(pipeline.transform_sentence_examples(warmup_examples[:1] or warmup_examples)[0])
        pipeline.model = TorchMLPClassifier(input_dim=feature_dim, device=args.device)
    return pipeline


def split_sentence_examples(sentence_examples, eval_size: float, seed: int):
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(sentence_examples))
    labels = np.asarray([example.label for example in sentence_examples])
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=eval_size,
        random_state=seed,
        stratify=labels,
    )
    train_examples = [sentence_examples[index] for index in train_idx]
    eval_examples = [sentence_examples[index] for index in eval_idx]
    return train_examples, eval_examples


def source_breakdown(sentence_examples) -> dict[str, int]:
    counts = Counter(example.metadata.get("dataset_name") or example.metadata.get("source_dataset") or example.source for example in sentence_examples)
    return dict(sorted(counts.items()))


def per_source_metrics(pipeline, sentence_examples, threshold: float) -> dict[str, dict[str, float]]:
    grouped = defaultdict(list)
    for example in sentence_examples:
        key = example.metadata.get("dataset_name") or example.metadata.get("source_dataset") or example.source
        grouped[key].append(example)
    return {
        key: sentence_level_metrics(pipeline, group, threshold=threshold)
        for key, group in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.hf_cache_dir = configure_hf_cache(args.hf_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = RunTracker(output_dir)
    tracker.save_json("run_config.json", vars(args), artifact_key="run_config")
    tracker.log("Initialized mixed-source training run.", stage="init", extra={"output_dir": str(output_dir)})

    try:
        tracker.set_stage("load_data")
        tracker.log("Loading mixed-source dataset.", stage="load_data")
        bundle = load_mixed_source_dataset(
            cache_dir=args.hf_cache_dir,
            context_window=args.context_window,
            seed=args.seed,
            ai_sample_size=args.hc3_reborn_sample_size,
            ai_dataset_id=args.hc3_reborn_dataset_id,
            asap2_sample_size=args.asap2_sample_size,
            ag_news_sample_size=args.ag_news_sample_size,
            arxiv_sample_size=args.arxiv_sample_size,
        )
        bundle_summary = {
            "document_count": len(bundle.documents),
            "sentence_example_count": len(bundle.sentence_examples),
            "source_breakdown": source_breakdown(bundle.sentence_examples),
        }
        tracker.save_json("dataset_summary.json", bundle_summary, artifact_key="dataset_summary")
        tracker.complete_stage("load_data", extra={"dataset_summary": bundle_summary})
        tracker.log("Loaded mixed-source dataset.", stage="load_data", extra=bundle_summary)

        tracker.set_stage("split_data")
        train_examples, eval_examples = split_sentence_examples(
            bundle.sentence_examples,
            eval_size=args.eval_size,
            seed=args.seed,
        )
        split_summary = {
            "train_sentence_count": len(train_examples),
            "eval_sentence_count": len(eval_examples),
            "train_source_breakdown": source_breakdown(train_examples),
            "eval_source_breakdown": source_breakdown(eval_examples),
        }
        tracker.save_json("split_summary.json", split_summary, artifact_key="split_summary")
        tracker.complete_stage("split_data", extra=split_summary)
        tracker.log("Created train/eval split.", stage="split_data", extra=split_summary)

        tracker.set_stage("build_pipeline")
        tracker.log("Building pipeline.", stage="build_pipeline")
        pipeline = build_pipeline(args, train_examples)
        tracker.complete_stage("build_pipeline")

        tracker.set_stage("fit")
        tracker.log("Fitting model.", stage="fit")
        pipeline.fit_sentence_examples(train_examples)
        tracker.save_pickle("pipeline_after_fit.pkl", pipeline, artifact_key="pipeline_after_fit")
        tracker.complete_stage("fit")
        tracker.log("Finished fitting model.", stage="fit")

        calibration_count = 0
        if args.calibrate:
            tracker.set_stage("calibration")
            calibration_cut = max(1, len(eval_examples) // 3)
            calibration_count = calibration_cut
            tracker.log("Running calibration.", stage="calibration", extra={"calibration_sentence_count": calibration_cut})
            pipeline.calibrate(eval_examples[:calibration_cut])
            tracker.save_pickle("pipeline_after_calibration.pkl", pipeline, artifact_key="pipeline_after_calibration")
            tracker.complete_stage("calibration", extra={"calibration_sentence_count": calibration_cut})
            tracker.log("Finished calibration.", stage="calibration")

        tracker.set_stage("evaluation")
        attacks = [attack.strip() for attack in args.robustness_attacks.split(",") if attack.strip()]
        tracker.log("Running evaluation and robustness checks.", stage="evaluation")
        metrics = sentence_level_metrics(pipeline, eval_examples, threshold=args.document_threshold)
        per_source = per_source_metrics(pipeline, eval_examples, threshold=args.document_threshold)
        robustness = robustness_benchmark(
            pipeline,
            eval_examples[: min(len(eval_examples), 4000)],
            attacks=attacks,
            threshold=args.document_threshold,
            seed=args.seed,
        )
        feature_importance = permutation_feature_importance(
            pipeline,
            eval_examples[: min(len(eval_examples), 1024)],
        )
        interpretability = sample_interpretability_report(
            pipeline,
            bundle.documents[:3],
            top_k=3,
        )
        tracker.save_json(
            "intermediate_metrics.json",
            {
                "metrics": metrics,
                "per_source_metrics": per_source,
                "robustness": robustness,
                "feature_importance": feature_importance,
            },
            artifact_key="intermediate_metrics",
        )
        tracker.complete_stage("evaluation")
        tracker.log("Finished evaluation.", stage="evaluation")

        tracker.set_stage("finalize")
        tracker.log("Saving final pipeline and metadata.", stage="finalize")
        tracker.save_pickle("pipeline.pkl", pipeline, artifact_key="pipeline_final")

        metadata = {
            "task": "mixed_source_binary_detection",
            "sources": {
                "hc3_reborn_label": 1,
                "asap2_label": 0,
                "ag_news_label": 0,
                "arxiv_label": 0,
            },
            "sample_sizes": {
                "hc3_reborn": args.hc3_reborn_sample_size,
                "asap2": args.asap2_sample_size or "all_available",
                "ag_news": args.ag_news_sample_size,
                "arxiv": args.arxiv_sample_size,
            },
            "context_window": args.context_window,
            "prob_model_name": args.prob_model_name,
            "prob_batch_size": args.prob_batch_size,
            "model_type": args.model_type,
            "device": args.device,
            "seed": args.seed,
            "hf_cache_dir": args.hf_cache_dir,
            "sentence_example_count": len(bundle.sentence_examples),
            "train_sentence_count": len(train_examples),
            "eval_sentence_count": len(eval_examples),
            "train_source_breakdown": source_breakdown(train_examples),
            "eval_source_breakdown": source_breakdown(eval_examples),
            "dataset_summary": bundle_summary,
            "calibration_sentence_count": calibration_count,
            "metrics": metrics,
            "per_source_metrics": per_source,
            "robustness": robustness,
            "feature_importance": feature_importance,
            "interpretability_report": interpretability,
        }
        tracker.save_json("metadata.json", metadata, artifact_key="metadata")
        save_experiment_outputs(output_dir, metadata)
        tracker.complete_stage("finalize")
        tracker.finalize("completed")
        tracker.log("Run completed successfully.", stage="finalize")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
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
