from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from ..utils import safe_divide, simple_word_tokenize


@dataclass
class CohesionFeatureExtractor:
    """
    TOCSIN-inspired token cohesion features.

    The implementation follows the paper's high-level logic:
    repeatedly delete random tokens and measure semantic drift.
    """

    deletion_ratio: float = 0.15
    num_rounds: int = 8
    random_seed: int = 42

    def _semantic_similarity(self, original: str, corrupted: str) -> float:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            matrix = TfidfVectorizer().fit_transform([original, corrupted])
            return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
        except Exception:
            source = set(simple_word_tokenize(original.lower()))
            target = set(simple_word_tokenize(corrupted.lower()))
            union_size = len(source | target)
            if union_size == 0:
                return 1.0
            return len(source & target) / union_size

    def _delete_random_tokens(self, text: str, rng: random.Random) -> tuple[str, float]:
        tokens = simple_word_tokenize(text)
        if len(tokens) <= 2:
            return text, 0.0

        delete_count = max(1, int(len(tokens) * self.deletion_ratio))
        indices = set(rng.sample(range(len(tokens)), k=min(delete_count, len(tokens) - 1)))
        kept_tokens = [token for idx, token in enumerate(tokens) if idx not in indices]
        corrupted = " ".join(kept_tokens)
        return corrupted, safe_divide(len(indices), len(tokens))

    def transform(self, text: str) -> dict[str, float]:
        rng = random.Random(self.random_seed)
        drifts: list[float] = []
        deletion_rates: list[float] = []

        for _ in range(self.num_rounds):
            corrupted, deletion_rate = self._delete_random_tokens(text, rng)
            deletion_rates.append(deletion_rate)
            similarity = self._semantic_similarity(text, corrupted)
            drifts.append(1.0 - similarity)

        token_count = len(simple_word_tokenize(text))
        return {
            "cohesion_avg_semantic_drift": statistics.mean(drifts) if drifts else 0.0,
            "cohesion_max_semantic_drift": max(drifts) if drifts else 0.0,
            "cohesion_min_semantic_drift": min(drifts) if drifts else 0.0,
            "cohesion_std_semantic_drift": statistics.pstdev(drifts) if len(drifts) > 1 else 0.0,
            "cohesion_avg_deletion_ratio": statistics.mean(deletion_rates) if deletion_rates else 0.0,
            "cohesion_length_normalized_drift": safe_divide(statistics.mean(drifts) if drifts else 0.0, token_count),
        }
