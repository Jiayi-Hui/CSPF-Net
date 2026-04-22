"""Text-side prototype for CSPF-Net."""

from .data import load_text_dataset, normalize_text_dataset
from .pipeline import CSPFTextPipeline, DocumentPrediction

__all__ = [
    "CSPFTextPipeline",
    "DocumentPrediction",
    "load_text_dataset",
    "normalize_text_dataset",
]
