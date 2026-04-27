from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from cspf_text import CSPFTextPipeline, configure_hf_cache, load_text_dataset
from cspf_text.evaluation import (
    coauthorship_metrics,
    document_level_metrics,
    robustness_benchmark,
    save_experiment_outputs,
    sentence_level_metrics,
)
from cspf_text.features import ProbabilisticFeatureExtractor
from cspf_text.modeling import TorchMLPClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-only CSPF-Net experiments.")
    parser.add_argument("--train-dataset", default="hc3_reborn")
    parser.add_argument("--train-dataset-id", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-datasets", default="hc3_reborn:test,asap2:test,haco-det:train")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--eval-sample-size", type=int, default=512)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--prob-model-name", default="gpt2")
    parser.add_argument("--prob-batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-type", choices=("stacking", "mlp"), default="stacking")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seeds", default="13,21,42")
    parser.add_argument("--attacks", default="synonym_substitution,punctuation_strip,sentence_shuffle,character_noise,context_truncation")
    parser.add_argument("--output-dir", default="artifacts/text_experiments")
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


def build_pipeline(args: argparse.Namespace, train_bundle) -> CSPFTextPipeline:
    pipeline = CSPFTextPipeline(
        probabilistic_extractor=ProbabilisticFeatureExtractor(
            model_name=args.prob_model_name,
            batch_size=args.prob_batch_size,
            device=args.device,
            cache_dir=args.hf_cache_dir,
            local_files_only=args.local_files_only,
        ),
        context_window=args.context_window,
    )
    if args.model_type == "mlp":
        warmup_examples = train_bundle.sentence_examples[:1] or train_bundle.sentence_examples
        feature_dim = len(pipeline.transform_sentence_examples(warmup_examples)[0])
        pipeline.model = TorchMLPClassifier(input_dim=feature_dim, device=args.device)
    return pipeline


def parse_eval_specs(spec: str) -> list[tuple[str, str]]:
    pairs = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            dataset_name, split = chunk.split(":", 1)
        else:
            dataset_name, split = chunk, "test"
        pairs.append((dataset_name, split))
    return pairs


def aggregate_runs(run_results: list[dict[str, float]]) -> dict[str, float]:
    if not run_results:
        return {}
    keys = sorted({key for result in run_results for key in result})
    summary = {}
    for key in keys:
        values = [result[key] for result in run_results if key in result]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def main() -> None:
    args = parse_args()
    args.hf_cache_dir = configure_hf_cache(args.hf_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    attacks = [item.strip() for item in args.attacks.split(",") if item.strip()]

    train_bundle = load_text_dataset(
        name=args.train_dataset,
        dataset_id=args.train_dataset_id,
        split=args.train_split,
        text_column=args.text_column,
        label_column=args.label_column,
        sample_size=args.sample_size,
        cache_dir=args.hf_cache_dir,
        context_window=args.context_window,
        local_files_only=args.local_files_only,
    )

    experiment_results: dict[str, dict] = {"cross_domain": {}, "ablations": {}, "robustness": {}}
    eval_specs = parse_eval_specs(args.eval_datasets)

    for seed in seeds:
        set_seed(seed)
        pipeline = build_pipeline(args, train_bundle)
        pipeline.fit_sentence_examples(train_bundle.sentence_examples)

        for dataset_name, split in eval_specs:
            eval_bundle = load_text_dataset(
                name=dataset_name,
                split=split,
                text_column=args.text_column,
                label_column=args.label_column,
                sample_size=args.eval_sample_size,
                cache_dir=args.hf_cache_dir,
                context_window=args.context_window,
                local_files_only=args.local_files_only,
            )
            key = f"{dataset_name}:{split}"
            experiment_results["cross_domain"].setdefault(key, []).append(
                {
                    **document_level_metrics(pipeline, eval_bundle, threshold=args.threshold),
                    **sentence_level_metrics(pipeline, eval_bundle.sentence_examples, threshold=args.threshold),
                    **coauthorship_metrics(pipeline, eval_bundle.documents, threshold=args.threshold),
                }
            )

        for ablation_name, use_context in (("full_model", True), ("no_context", False)):
            ablation_pipeline = build_pipeline(args, train_bundle)
            ablation_pipeline.use_context = use_context
            ablation_pipeline.fit_sentence_examples(train_bundle.sentence_examples)
            eval_bundle = load_text_dataset(
                name=args.train_dataset,
                dataset_id=args.train_dataset_id,
                split=args.train_split,
                text_column=args.text_column,
                label_column=args.label_column,
                sample_size=args.eval_sample_size,
                cache_dir=args.hf_cache_dir,
                context_window=args.context_window,
                local_files_only=args.local_files_only,
            )
            experiment_results["ablations"].setdefault(ablation_name, []).append(
                sentence_level_metrics(ablation_pipeline, eval_bundle.sentence_examples, threshold=args.threshold)
            )
            experiment_results["robustness"].setdefault(ablation_name, []).append(
                robustness_benchmark(
                    ablation_pipeline,
                    eval_bundle.sentence_examples,
                    attacks=attacks,
                    threshold=args.threshold,
                    seed=seed,
                )
            )

    aggregated = {
        "cross_domain": {key: aggregate_runs(value) for key, value in experiment_results["cross_domain"].items()},
        "ablations": {key: aggregate_runs(value) for key, value in experiment_results["ablations"].items()},
        "robustness": {
            model_name: {
                attack_name: aggregate_runs([run[attack_name] for run in runs if attack_name in run])
                for attack_name in sorted({attack for run in runs for attack in run})
            }
            for model_name, runs in experiment_results["robustness"].items()
        },
        "seeds": seeds,
        "train_dataset": args.train_dataset,
    }
    save_experiment_outputs(output_dir, aggregated)
    (output_dir / "summary.json").write_text(json.dumps(aggregated, indent=2, ensure_ascii=False))
    print(json.dumps(aggregated, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
