from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ProbabilisticFeatureExtractor:
    """Extract GPT-style log-likelihood features from text."""

    model_name: str = "gpt2"
    max_length: int = 512
    batch_size: int = 8
    device: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def _lazy_load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "prob_features.py requires `torch` and `transformers`. "
                "Install them before calling ProbabilisticFeatureExtractor."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self._model.eval()

        device = self._resolve_device(torch)
        self.device = device
        self._model.to(device)

    def _resolve_device(self, torch) -> str:
        if self.device is not None:
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but `torch.cuda.is_available()` is False.")
            if self.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested, but it is not available in this PyTorch build.")
            return self.device

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _empty_features() -> dict[str, float]:
        return {
            "prob_avg_neg_log_likelihood": 0.0,
            "prob_perplexity": 1.0,
            "prob_token_nll_std": 0.0,
            "prob_token_nll_max": 0.0,
            "prob_token_nll_min": 0.0,
            "prob_first_last_nll_gap": 0.0,
        }

    def _features_from_token_nll(self, token_nll) -> dict[str, float]:
        if token_nll.numel() == 0:
            return self._empty_features()

        avg_nll = float(token_nll.mean().item())
        std_nll = float(token_nll.std(unbiased=False).item()) if token_nll.numel() > 1 else 0.0
        max_nll = float(token_nll.max().item())
        min_nll = float(token_nll.min().item())
        first_last_gap = float(token_nll[-1].item() - token_nll[0].item()) if token_nll.numel() > 1 else 0.0
        return {
            "prob_avg_neg_log_likelihood": avg_nll,
            "prob_perplexity": float(math.exp(min(avg_nll, 20.0))),
            "prob_token_nll_std": std_nll,
            "prob_token_nll_max": max_nll,
            "prob_token_nll_min": min_nll,
            "prob_first_last_nll_gap": first_last_gap,
        }

    def transform_batch(self, texts: list[str]) -> list[dict[str, float]]:
        if not texts:
            return []

        normalized_texts = [text if isinstance(text, str) else "" for text in texts]
        if not any(text.strip() for text in normalized_texts):
            return [self._empty_features() for _ in normalized_texts]

        self._lazy_load()
        import torch

        features: list[dict[str, float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = normalized_texts[start : start + self.batch_size]
            non_empty_pairs = [(idx, text) for idx, text in enumerate(batch_texts) if text.strip()]
            if not non_empty_pairs:
                features.extend(self._empty_features() for _ in batch_texts)
                continue

            non_empty_indices = [idx for idx, _ in non_empty_pairs]
            non_empty_texts = [text for _, text in non_empty_pairs]
            encoded = self._tokenizer(
                non_empty_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)

            logits = outputs.logits[:, :-1, :]
            shifted_labels = encoded["input_ids"][:, 1:]
            shifted_mask = encoded["attention_mask"][:, 1:].bool()
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
            token_nll = -token_log_probs

            batch_features = [self._empty_features() for _ in batch_texts]
            for feature_index, (row_nll, row_mask) in enumerate(zip(token_nll, shifted_mask)):
                row_token_nll = row_nll[row_mask]
                batch_features[non_empty_indices[feature_index]] = self._features_from_token_nll(row_token_nll)
            features.extend(batch_features)

        return features

    def transform(self, text: str) -> dict[str, float]:
        return self.transform_batch([text])[0]
