from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cspf_text import configure_hf_cache, load_text_dataset
from cspf_text.run_tracking import RunTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BERT baseline with bert-base-uncased.")
    parser.add_argument("--dataset", default="hc3_reborn", help="Dataset alias or Hugging Face dataset id.")
    parser.add_argument("--dataset-id", default=None, help="Override Hugging Face dataset id.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--eval-sample-size", type=int, default=None)
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/bert_baseline")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def resolve_device(requested: str | None):
    import torch

    if requested is not None:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but `torch.cuda.is_available()` is False.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available in this PyTorch build.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_metrics(labels, probabilities: list[float]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    predictions = [int(prob >= 0.5) for prob in probabilities]
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    if len(set(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.hf_cache_dir = configure_hf_cache(args.hf_cache_dir)

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class TextDataset(Dataset):
        def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int) -> None:
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, index: int):
            encoding = self.tokenizer(
                self.texts[index],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoding.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    artifact_root = Path(args.output_dir)
    checkpoint_dir = artifact_root / "checkpoints" / "bert-base-uncased"
    export_dir = artifact_root / "exports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    tracker = RunTracker(checkpoint_dir)
    tracker.save_json("run_config.json", vars(args), artifact_key="run_config")

    try:
        tracker.set_stage("load_train_data")
        train_bundle = load_text_dataset(
            name=args.dataset,
            dataset_id=args.dataset_id,
            split=args.train_split,
            text_column=args.text_column,
            label_column=args.label_column,
            sample_size=args.sample_size,
            cache_dir=args.hf_cache_dir,
            local_files_only=args.local_files_only,
        )
        train_summary = {
            "text_count": len(train_bundle.texts),
            "label_count": len(train_bundle.labels),
        }
        tracker.save_json("train_bundle_summary.json", train_summary, artifact_key="train_bundle_summary")
        tracker.complete_stage("load_train_data", extra={"train_bundle_summary": train_summary})

        tracker.set_stage("tokenizer_and_model")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.hf_cache_dir,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            cache_dir=args.hf_cache_dir,
            local_files_only=args.local_files_only,
        )
        device = resolve_device(args.device)
        model.to(device)
        tracker.complete_stage("tokenizer_and_model", extra={"resolved_model_device": device})

        tracker.set_stage("fit")
        from transformers import get_linear_schedule_with_warmup

        train_dataset = TextDataset(train_bundle.texts, train_bundle.labels, tokenizer, args.max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )

        model.train()
        for _ in range(args.epochs):
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()
        tracker.complete_stage("fit")

        metrics: dict[str, float] = {}
        eval_summary: dict[str, int] = {}
        if args.eval_split:
            tracker.set_stage("evaluation")
            eval_bundle = load_text_dataset(
                name=args.dataset,
                dataset_id=args.dataset_id,
                split=args.eval_split,
                text_column=args.text_column,
                label_column=args.label_column,
                sample_size=args.eval_sample_size,
                cache_dir=args.hf_cache_dir,
                local_files_only=args.local_files_only,
            )
            eval_summary = {
                "text_count": len(eval_bundle.texts),
                "label_count": len(eval_bundle.labels),
            }
            eval_dataset = TextDataset(eval_bundle.texts, eval_bundle.labels, tokenizer, args.max_length)
            eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

            model.eval()
            probabilities: list[float] = []
            with torch.no_grad():
                for batch in eval_loader:
                    labels = batch.pop("labels")
                    batch = {key: value.to(device) for key, value in batch.items()}
                    logits = model(**batch).logits
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                    probabilities.extend(probs)
            metrics = compute_metrics(eval_bundle.labels, probabilities)
            tracker.save_json("eval_bundle_summary.json", eval_summary, artifact_key="eval_bundle_summary")
            tracker.save_json("metrics.json", metrics, artifact_key="metrics")
            tracker.complete_stage("evaluation", extra={"eval_bundle_summary": eval_summary, "metrics": metrics})

        tracker.set_stage("finalize")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        metadata = {
            "dataset": args.dataset,
            "dataset_id": args.dataset_id,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "sample_size": args.sample_size,
            "eval_sample_size": args.eval_sample_size,
            "model_name": args.model_name,
            "resolved_model_device": device,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "hf_cache_dir": args.hf_cache_dir,
            "local_files_only": args.local_files_only,
            "train_bundle_summary": train_summary,
            "eval_bundle_summary": eval_summary,
            "metrics": metrics,
        }
        tracker.save_json("metadata.json", metadata, artifact_key="metadata")
        (export_dir / "bert_base_uncased_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tracker.complete_stage("finalize")
        tracker.finalize("completed")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
    except Exception as exc:
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
