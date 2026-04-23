from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cspf_text import CSPFTextPipeline, load_mixed_source_dataset
from cspf_text.evaluation import (
    permutation_feature_importance,
    robustness_benchmark,
    sample_interpretability_report,
    save_experiment_outputs,
    sentence_level_metrics,
)
from cspf_text.features import ProbabilisticFeatureExtractor
from cspf_text.modeling import TorchMLPClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mixed-source binary text detector.")
    parser.add_argument("--raid-sample-size", type=int, default=36000)
    parser.add_argument("--asap2-sample-size", type=int, default=None)
    parser.add_argument("--ag-news-sample-size", type=int, default=9000)
    parser.add_argument("--arxiv-sample-size", type=int, default=9000)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--prob-model-name", default="gpt2")
    parser.add_argument("--prob-batch-size", type=int, default=8)
    parser.add_argument("--model-type", choices=("stacking", "mlp"), default="stacking")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-cache-dir", default="/workspace/CSPF-Net/hf_cache")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_mixed_source_dataset(
        cache_dir=args.hf_cache_dir,
        context_window=args.context_window,
        seed=args.seed,
        raid_sample_size=args.raid_sample_size,
        asap2_sample_size=args.asap2_sample_size,
        ag_news_sample_size=args.ag_news_sample_size,
        arxiv_sample_size=args.arxiv_sample_size,
    )

    train_examples, eval_examples = split_sentence_examples(
        bundle.sentence_examples,
        eval_size=args.eval_size,
        seed=args.seed,
    )

    pipeline = build_pipeline(args, train_examples)
    pipeline.fit_sentence_examples(train_examples)
    if args.calibrate:
        calibration_cut = max(1, len(eval_examples) // 3)
        pipeline.calibrate(eval_examples[:calibration_cut])

    attacks = [attack.strip() for attack in args.robustness_attacks.split(",") if attack.strip()]
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

    with open(output_dir / "pipeline.pkl", "wb") as handle:
        pickle.dump(pipeline, handle)

    metadata = {
        "task": "mixed_source_binary_detection",
        "sources": {
            "raid_label": 1,
            "asap2_label": 0,
            "ag_news_label": 0,
            "arxiv_label": 0,
        },
        "sample_sizes": {
            "raid": args.raid_sample_size,
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
        "sentence_example_count": len(bundle.sentence_examples),
        "train_sentence_count": len(train_examples),
        "eval_sentence_count": len(eval_examples),
        "train_source_breakdown": source_breakdown(train_examples),
        "eval_source_breakdown": source_breakdown(eval_examples),
        "metrics": metrics,
        "per_source_metrics": per_source,
        "robustness": robustness,
        "feature_importance": feature_importance,
        "interpretability_report": interpretability,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    save_experiment_outputs(output_dir, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
