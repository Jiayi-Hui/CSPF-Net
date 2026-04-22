from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TorchMLPClassifier:
    """Simple tabular MLP for concatenated handcrafted features."""

    input_dim: int
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    device: str | None = None
    _model: object | None = field(default=None, init=False, repr=False)
    _scaler: object | None = field(default=None, init=False, repr=False)

    def _lazy_setup(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "mlp_model.py requires `torch` and `scikit-learn`."
            ) from exc

        self.device = self._resolve_device(torch)

        layers = []
        current_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))

        self._model = nn.Sequential(*layers).to(self.device)
        self._scaler = StandardScaler()

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

    def fit(self, X, y) -> "TorchMLPClassifier":
        self._lazy_setup()

        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_scaled = self._scaler.fit_transform(X)
        X_tensor = torch.tensor(np.asarray(X_scaled), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        self._model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self._model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        import numpy as np
        import torch

        X_scaled = self._scaler.transform(X)
        X_tensor = torch.tensor(np.asarray(X_scaled), dtype=torch.float32).to(self.device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.hstack([1.0 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
