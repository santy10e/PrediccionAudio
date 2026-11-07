"""Clasificación de audio industrial normal/anómalo basada en MFCC + Random Forest."""

from .audio_io import AudioClip, AudioLoadError, load_audio
from .features import extract_features
from .model import (
    IncompatibleModelError,
    ModelBundle,
    ModelNotFoundError,
    Prediction,
    load_model,
    train_model,
)

__all__ = [
    "AudioClip",
    "AudioLoadError",
    "IncompatibleModelError",
    "ModelBundle",
    "ModelNotFoundError",
    "Prediction",
    "extract_features",
    "load_audio",
    "load_model",
    "train_model",
]

__version__ = "2.0.0"
