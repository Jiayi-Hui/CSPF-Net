from __future__ import annotations

from dataclasses import dataclass, field

from .features import CohesionFeatureExtractor, ProbabilisticFeatureExtractor, StyleFeatureExtractor
from .modeling import SklearnStackingDetector
from .utils import flatten_feature_dicts, simple_sentence_split


@dataclass
class DocumentPrediction:
    document_probability: float
    ai_contribution_ratio: float
    sentence_probabilities: list[float]
    sentences: list[str]


@dataclass
class CSPFTextPipeline:
    """
    Sentence/document detector inspired by the proposal's text branch.

    1. Extract style, log-likelihood, and cohesion features.
    2. Train a tabular classifier on the concatenated vector.
    3. Aggregate sentence probabilities to document-level AI ratios.
    """

    probabilistic_extractor: ProbabilisticFeatureExtractor = field(default_factory=ProbabilisticFeatureExtractor)
    style_extractor: StyleFeatureExtractor = field(default_factory=StyleFeatureExtractor)
    cohesion_extractor: CohesionFeatureExtractor = field(default_factory=CohesionFeatureExtractor)
    model: object = field(default_factory=SklearnStackingDetector)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def _extract_single(self, text: str, probabilistic_features: dict[str, float] | None = None) -> list[float]:
        feature_names, feature_values = flatten_feature_dicts(
            [
                self.style_extractor.transform(text),
                probabilistic_features or self.probabilistic_extractor.transform(text),
                self.cohesion_extractor.transform(text),
            ]
        )
        if not self.feature_names_:
            self.feature_names_ = feature_names
        return feature_values

    def transform(self, texts: list[str]) -> list[list[float]]:
        probabilistic_features = self.probabilistic_extractor.transform_batch(texts)
        return [
            self._extract_single(text, probabilistic_feature)
            for text, probabilistic_feature in zip(texts, probabilistic_features)
        ]

    def fit(self, texts: list[str], labels: list[int]) -> "CSPFTextPipeline":
        X = self.transform(texts)
        self.model.fit(X, labels)
        return self

    def predict(self, texts: list[str]):
        return self.model.predict(self.transform(texts))

    def predict_proba(self, texts: list[str]):
        return self.model.predict_proba(self.transform(texts))

    def predict_document(self, document: str, threshold: float = 0.5) -> DocumentPrediction:
        sentences = simple_sentence_split(document)
        if not sentences:
            return DocumentPrediction(0.0, 0.0, [], [])

        probabilities = self.predict_proba(sentences)[:, 1].tolist()
        ai_ratio = sum(prob >= threshold for prob in probabilities) / len(probabilities)
        document_probability = sum(probabilities) / len(probabilities)
        return DocumentPrediction(
            document_probability=document_probability,
            ai_contribution_ratio=ai_ratio,
            sentence_probabilities=probabilities,
            sentences=sentences,
        )
