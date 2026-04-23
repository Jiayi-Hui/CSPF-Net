from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import SentenceExample, build_sentence_dataset
from .features import CohesionFeatureExtractor, ProbabilisticFeatureExtractor, StyleFeatureExtractor
from .modeling import ProbabilityCalibrator, SklearnStackingDetector
from .utils import flatten_feature_dicts, simple_sentence_split, simple_word_tokenize


@dataclass
class FeatureContribution:
    name: str
    value: float
    contribution: float


@dataclass
class TokenContribution:
    token: str
    importance: float


@dataclass
class SentencePrediction:
    sentence_id: int
    sentence: str
    probability: float
    label: int | None = None
    left_context: str = ""
    right_context: str = ""
    suspicious_rank: int = 0
    top_features: list[FeatureContribution] = field(default_factory=list)
    top_tokens: list[TokenContribution] = field(default_factory=list)
    context_effects: dict[str, float] = field(default_factory=dict)


@dataclass
class DocumentPrediction:
    document_probability: float
    ai_contribution_ratio: float
    sentence_probabilities: list[float]
    sentences: list[str]
    sentence_predictions: list[SentencePrediction] = field(default_factory=list)
    top_document_features: list[FeatureContribution] = field(default_factory=list)


@dataclass
class CSPFTextPipeline:
    """
    Context-aware sentence/document detector for the proposal's text branch.

    1. Extract target-sentence, left-context, right-context, and fused features.
    2. Train a sentence-level classifier on the concatenated vector.
    3. Aggregate sentence probabilities into document-level AI ratios.
    """

    probabilistic_extractor: ProbabilisticFeatureExtractor = field(default_factory=ProbabilisticFeatureExtractor)
    style_extractor: StyleFeatureExtractor = field(default_factory=StyleFeatureExtractor)
    cohesion_extractor: CohesionFeatureExtractor = field(default_factory=CohesionFeatureExtractor)
    model: object = field(default_factory=SklearnStackingDetector)
    context_window: int = 1
    use_context: bool = True
    calibrator: ProbabilityCalibrator | None = None
    feature_names_: list[str] = field(default_factory=list, init=False)
    feature_baselines_: list[float] = field(default_factory=list, init=False)

    def _base_feature_dicts(
        self,
        text: str,
        probabilistic_features: dict[str, float] | None = None,
    ) -> list[dict[str, float]]:
        return [
            self.style_extractor.transform(text),
            probabilistic_features or self.probabilistic_extractor.transform(text),
            self.cohesion_extractor.transform(text),
        ]

    def _zero_feature_dicts(self) -> list[dict[str, float]]:
        return self._base_feature_dicts("", probabilistic_features=self.probabilistic_extractor._empty_features())

    def _prefix_feature_dicts(self, prefix: str, feature_dicts: list[dict[str, float]]) -> dict[str, float]:
        flattened: dict[str, float] = {}
        for feature_dict in feature_dicts:
            for key, value in feature_dict.items():
                flattened[f"{prefix}_{key}"] = float(value)
        return flattened

    def _delta_feature_dict(
        self,
        target_features: list[dict[str, float]],
        context_features: list[dict[str, float]],
    ) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for target_dict, context_dict in zip(target_features, context_features):
            for key, value in target_dict.items():
                deltas[f"delta_{key}"] = float(value) - float(context_dict.get(key, 0.0))
        return deltas

    def _extract_from_example(
        self,
        example: SentenceExample,
        target_prob: dict[str, float],
        left_prob: dict[str, float],
        right_prob: dict[str, float],
        merged_prob: dict[str, float],
    ) -> list[float]:
        target_features = self._base_feature_dicts(example.text, probabilistic_features=target_prob)
        if self.use_context:
            left_features = self._base_feature_dicts(example.left_context, probabilistic_features=left_prob)
            right_features = self._base_feature_dicts(example.right_context, probabilistic_features=right_prob)
            merged_context_text = " ".join(part for part in (example.left_context, example.right_context) if part).strip()
            merged_features = self._base_feature_dicts(merged_context_text, probabilistic_features=merged_prob)
        else:
            left_features = self._zero_feature_dicts()
            right_features = self._zero_feature_dicts()
            merged_features = self._zero_feature_dicts()

        feature_dicts = [
            self._prefix_feature_dicts("target", target_features),
            self._prefix_feature_dicts("left", left_features),
            self._prefix_feature_dicts("right", right_features),
            self._prefix_feature_dicts("context", merged_features),
            self._delta_feature_dict(target_features, merged_features),
        ]
        feature_names, feature_values = flatten_feature_dicts(feature_dicts)
        if not self.feature_names_:
            self.feature_names_ = feature_names
        return feature_values

    def transform_sentence_examples(self, examples: list[SentenceExample]) -> list[list[float]]:
        if not examples:
            return []

        target_probs = self.probabilistic_extractor.transform_batch([example.text for example in examples])
        left_probs = self.probabilistic_extractor.transform_batch([example.left_context for example in examples])
        right_probs = self.probabilistic_extractor.transform_batch([example.right_context for example in examples])
        merged_probs = self.probabilistic_extractor.transform_batch(
            [" ".join(part for part in (example.left_context, example.right_context) if part).strip() for example in examples]
        )
        return [
            self._extract_from_example(example, target_prob, left_prob, right_prob, merged_prob)
            for example, target_prob, left_prob, right_prob, merged_prob in zip(
                examples,
                target_probs,
                left_probs,
                right_probs,
                merged_probs,
            )
        ]

    def transform(self, texts: list[str]) -> list[list[float]]:
        examples = build_sentence_dataset(texts=texts, labels=[0 for _ in texts], context_window=self.context_window)
        grouped: dict[str, list[SentenceExample]] = {}
        document_order: list[str] = []
        for example in examples:
            if example.document_id not in grouped:
                grouped[example.document_id] = []
                document_order.append(example.document_id)
            grouped[example.document_id].append(example)
        if not grouped:
            return []
        sentence_vectors = self.transform_sentence_examples(examples)
        vector_size = len(sentence_vectors[0])
        document_vectors: list[list[float]] = []
        offset = 0
        for document_id in document_order:
            doc_examples = grouped[document_id]
            doc_vectors = sentence_vectors[offset : offset + len(doc_examples)]
            offset += len(doc_examples)
            document_vectors.append(np.mean(np.asarray(doc_vectors), axis=0).tolist() if doc_vectors else [0.0] * vector_size)
        return document_vectors

    def fit_sentence_examples(self, examples: list[SentenceExample]) -> "CSPFTextPipeline":
        X = self.transform_sentence_examples(examples)
        y = [example.label for example in examples]
        self.model.fit(X, y)
        self.feature_baselines_ = np.mean(np.asarray(X), axis=0).tolist() if X else []
        return self

    def fit(self, texts: list[str], labels: list[int]) -> "CSPFTextPipeline":
        examples = build_sentence_dataset(texts=texts, labels=labels, context_window=self.context_window)
        return self.fit_sentence_examples(examples)

    def calibrate(self, examples: list[SentenceExample]) -> "CSPFTextPipeline":
        if not examples:
            return self
        X = self.transform_sentence_examples(examples)
        y = [example.label for example in examples]
        raw_probabilities = self.model.predict_proba(X)[:, 1].tolist()
        self.calibrator = ProbabilityCalibrator()
        self.calibrator.fit(raw_probabilities, y)
        return self

    def _apply_calibration(self, probabilities):
        if self.calibrator is None:
            return probabilities
        calibrated = self.calibrator.predict_proba(probabilities[:, 1].tolist())
        probabilities = probabilities.copy()
        probabilities[:, 1] = calibrated
        probabilities[:, 0] = 1.0 - calibrated
        return probabilities

    def predict(self, texts: list[str]):
        probabilities = self.predict_proba(texts)[:, 1]
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, texts: list[str]):
        X = self.transform(texts)
        probabilities = self.model.predict_proba(X)
        return self._apply_calibration(probabilities)

    def predict_sentence_examples_proba(self, examples: list[SentenceExample]):
        X = self.transform_sentence_examples(examples)
        probabilities = self.model.predict_proba(X)
        return self._apply_calibration(probabilities)

    def _predict_single_vector_probability(self, vector: list[float]) -> float:
        probabilities = self.model.predict_proba([vector])
        probabilities = self._apply_calibration(probabilities)
        return float(probabilities[0, 1])

    def explain_sentence_example(
        self,
        example: SentenceExample,
        top_k_features: int = 8,
        top_k_tokens: int = 8,
    ) -> SentencePrediction:
        vector = self.transform_sentence_examples([example])[0]
        probability = self._predict_single_vector_probability(vector)

        feature_contributions: list[FeatureContribution] = []
        baselines = self.feature_baselines_ or [0.0 for _ in vector]
        for index, value in enumerate(vector):
            counterfactual = list(vector)
            counterfactual[index] = baselines[index]
            contribution = abs(probability - self._predict_single_vector_probability(counterfactual))
            feature_contributions.append(
                FeatureContribution(
                    name=self.feature_names_[index],
                    value=float(value),
                    contribution=float(contribution),
                )
            )
        feature_contributions.sort(key=lambda item: item.contribution, reverse=True)

        token_contributions: list[TokenContribution] = []
        if top_k_tokens > 0:
            for token in simple_word_tokenize(example.text):
                perturbed_text = example.text.replace(token, "", 1).strip()
                perturbed_example = SentenceExample(
                    text=perturbed_text or example.text,
                    label=example.label,
                    document_id=example.document_id,
                    sentence_id=example.sentence_id,
                    left_context=example.left_context,
                    right_context=example.right_context,
                    full_document=example.full_document,
                    source=example.source,
                    metadata=dict(example.metadata),
                )
                perturbed_prob = self.predict_sentence_examples_proba([perturbed_example])[0, 1]
                token_contributions.append(TokenContribution(token=token, importance=float(abs(probability - perturbed_prob))))
            token_contributions.sort(key=lambda item: item.importance, reverse=True)

        target_only = SentenceExample(
            text=example.text,
            label=example.label,
            document_id=example.document_id,
            sentence_id=example.sentence_id,
            full_document=example.full_document,
            source=example.source,
            metadata=dict(example.metadata),
        )
        no_left = SentenceExample(
            text=example.text,
            label=example.label,
            document_id=example.document_id,
            sentence_id=example.sentence_id,
            right_context=example.right_context,
            full_document=example.full_document,
            source=example.source,
            metadata=dict(example.metadata),
        )
        no_right = SentenceExample(
            text=example.text,
            label=example.label,
            document_id=example.document_id,
            sentence_id=example.sentence_id,
            left_context=example.left_context,
            full_document=example.full_document,
            source=example.source,
            metadata=dict(example.metadata),
        )
        context_effects = {
            "full_context": probability,
            "target_only": float(self.predict_sentence_examples_proba([target_only])[0, 1]),
            "without_left_context": float(self.predict_sentence_examples_proba([no_left])[0, 1]),
            "without_right_context": float(self.predict_sentence_examples_proba([no_right])[0, 1]),
        }
        return SentencePrediction(
            sentence_id=example.sentence_id,
            sentence=example.text,
            probability=probability,
            label=example.label,
            left_context=example.left_context,
            right_context=example.right_context,
            top_features=feature_contributions[:top_k_features],
            top_tokens=token_contributions[:top_k_tokens],
            context_effects=context_effects,
        )

    def _document_feature_summary(self, examples: list[SentenceExample]) -> list[FeatureContribution]:
        if not examples:
            return []
        explanations = [self.explain_sentence_example(example, top_k_features=5, top_k_tokens=0) for example in examples]
        aggregate: dict[str, list[float]] = {}
        values: dict[str, list[float]] = {}
        for explanation in explanations:
            for contribution in explanation.top_features:
                aggregate.setdefault(contribution.name, []).append(contribution.contribution)
                values.setdefault(contribution.name, []).append(contribution.value)
        summary = [
            FeatureContribution(
                name=name,
                value=float(np.mean(values[name])),
                contribution=float(np.mean(scores)),
            )
            for name, scores in aggregate.items()
        ]
        summary.sort(key=lambda item: item.contribution, reverse=True)
        return summary[:10]

    def predict_document(self, document: str, threshold: float = 0.5) -> DocumentPrediction:
        sentences = simple_sentence_split(document)
        if not sentences:
            return DocumentPrediction(0.0, 0.0, [], [])

        examples = build_sentence_dataset(
            texts=[document],
            labels=[0],
            context_window=self.context_window,
            document_prefix="prediction",
        )
        probabilities = self.predict_sentence_examples_proba(examples)[:, 1].tolist()
        ai_ratio = sum(prob >= threshold for prob in probabilities) / len(probabilities)
        document_probability = sum(probabilities) / len(probabilities)

        ranking = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)
        rank_map = {sentence_idx: rank + 1 for rank, sentence_idx in enumerate(ranking)}
        sentence_predictions = []
        for example, probability in zip(examples, probabilities):
            explanation = self.explain_sentence_example(example)
            explanation.probability = float(probability)
            explanation.suspicious_rank = rank_map[example.sentence_id]
            sentence_predictions.append(explanation)

        return DocumentPrediction(
            document_probability=document_probability,
            ai_contribution_ratio=ai_ratio,
            sentence_probabilities=probabilities,
            sentences=sentences,
            sentence_predictions=sentence_predictions,
            top_document_features=self._document_feature_summary(examples),
        )
