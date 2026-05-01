from __future__ import annotations

from dataclasses import dataclass, field


try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class _BranchTransformerModule(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        branch_dims: dict[str, int],
        branch_order: tuple[str, str, str],
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        if nn is None or torch is None:
            raise ImportError("feature_transformer_model.py requires `torch`.")
        super().__init__()
        self.branch_order = branch_order
        self.branch_projectors = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.Linear(branch_dims[branch], d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for branch in branch_order
            }
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, len(branch_order), d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, branch_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for branch in self.branch_order:
            tokens.append(self.branch_projectors[branch](branch_tensors[branch]).unsqueeze(1))
        token_tensor = torch.cat(tokens, dim=1) + self.position_embedding
        encoded = self.encoder(token_tensor)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


@dataclass
class FeatureTokenTransformerClassifier:
    """Fuse handcrafted feature branches with a lightweight transformer encoder."""

    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 192
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 64
    random_state: int = 42
    device: str | None = None
    _feature_names: list[str] = field(default_factory=list, init=False, repr=False)
    _branch_indices: dict[str, list[int]] = field(default_factory=dict, init=False, repr=False)
    _branch_order: tuple[str, str, str] = field(default=("style", "prob", "cohesion"), init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _scaler: object | None = field(default=None, init=False, repr=False)

    def set_feature_names(self, feature_names: list[str]) -> None:
        self._feature_names = list(feature_names)
        self._branch_indices = {branch: [] for branch in self._branch_order}
        for index, name in enumerate(self._feature_names):
            branch = self._infer_branch(name)
            if branch is not None:
                self._branch_indices[branch].append(index)

        if not any(self._branch_indices.values()):
            raise ValueError("Could not map any features into style/prob/cohesion branches.")

    def _infer_branch(self, name: str) -> str | None:
        lowered = name.lower()
        if "_style_" in lowered or lowered.startswith("style_") or lowered.startswith("delta_style_"):
            return "style"
        if "_prob_" in lowered or lowered.startswith("prob_") or lowered.startswith("delta_prob_"):
            return "prob"
        if "_cohesion_" in lowered or lowered.startswith("cohesion_") or lowered.startswith("delta_cohesion_"):
            return "cohesion"
        return None

    def _lazy_setup(self) -> None:
        try:
            import torch
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "feature_transformer_model.py requires `torch` and `scikit-learn`."
            ) from exc

        self.device = self._resolve_device(torch)
        self._scaler = StandardScaler()

        branch_dims = {
            branch: max(1, len(self._branch_indices.get(branch, [])))
            for branch in self._branch_order
        }

        self._model = _BranchTransformerModule(
            branch_dims=branch_dims,
            branch_order=self._branch_order,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

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

    def _split_branches(self, X):
        import numpy as np

        X_array = np.asarray(X, dtype=float)
        branches: dict[str, np.ndarray] = {}
        for branch in self._branch_order:
            indices = self._branch_indices.get(branch, [])
            if indices:
                branches[branch] = X_array[:, indices]
            else:
                branches[branch] = np.zeros((len(X_array), 1), dtype=float)
        return branches

    def fit(self, X, y) -> "FeatureTokenTransformerClassifier":
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if not self._feature_names:
            raise ValueError("set_feature_names(...) must be called before fit(...).")

        if self._model is None:
            self._lazy_setup()

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        X_scaled = self._scaler.fit_transform(X)
        branch_arrays = self._split_branches(X_scaled)

        tensors = {
            branch: torch.tensor(values, dtype=torch.float32)
            for branch, values in branch_arrays.items()
        }
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(tensors["style"], tensors["prob"], tensors["cohesion"], y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        self._model.train()
        for _ in range(self.epochs):
            for style_batch, prob_batch, cohesion_batch, label_batch in loader:
                style_batch = style_batch.to(self.device)
                prob_batch = prob_batch.to(self.device)
                cohesion_batch = cohesion_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                optimizer.zero_grad()
                logits = self._model(
                    {
                        "style": style_batch,
                        "prob": prob_batch,
                        "cohesion": cohesion_batch,
                    }
                )
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        import numpy as np
        import torch

        X_scaled = self._scaler.transform(X)
        branch_arrays = self._split_branches(X_scaled)
        branch_tensors = {
            branch: torch.tensor(values, dtype=torch.float32).to(self.device)
            for branch, values in branch_arrays.items()
        }

        self._model.eval()
        with torch.no_grad():
            logits = self._model(branch_tensors)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.hstack([1.0 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
