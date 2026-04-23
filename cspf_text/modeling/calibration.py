from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProbabilityCalibrator:
    """Lightweight Platt-style calibrator over sentence probabilities."""

    _model: object | None = field(default=None, init=False, repr=False)

    def fit(self, probabilities: list[float], labels: list[int]) -> "ProbabilityCalibrator":
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ImportError("calibration.py requires `numpy` and `scikit-learn`.") from exc

        X = np.asarray(probabilities, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        self._model = LogisticRegression(max_iter=1000)
        self._model.fit(X, y)
        return self

    def predict_proba(self, probabilities: list[float]):
        if self._model is None:
            return probabilities
        import numpy as np

        X = np.asarray(probabilities, dtype=float).reshape(-1, 1)
        return self._model.predict_proba(X)[:, 1]
