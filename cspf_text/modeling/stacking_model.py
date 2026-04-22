from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SklearnStackingDetector:
    """Feature-level ensemble baseline for CSPF-Net text experiments."""

    random_state: int = 42
    _model: object | None = field(default=None, init=False, repr=False)

    def _build_model(self) -> object:
        try:
            from sklearn.ensemble import RandomForestClassifier, StackingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import LinearSVC
        except ImportError as exc:
            raise ImportError(
                "stacking_model.py requires `scikit-learn`."
            ) from exc

        estimators = [
            (
                "lr",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=2000, random_state=self.random_state)),
                    ]
                ),
            ),
            (
                "svc",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LinearSVC(random_state=self.random_state, dual="auto")),
                    ]
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),
        ]

        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, random_state=self.random_state),
            stack_method="auto",
            passthrough=True,
            n_jobs=-1,
        )

    def fit(self, X, y) -> "SklearnStackingDetector":
        self._model = self._build_model()
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)
