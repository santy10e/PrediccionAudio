"""Configuración central del proyecto: rutas y parámetros de procesamiento."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "source_test"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "audio_classifier.joblib"

LABEL_NORMAL = "normal"
LABEL_ANOMALY = "anomaly"
LABELS = (LABEL_NORMAL, LABEL_ANOMALY)

SUPPORTED_EXTENSIONS = ("wav", "flac", "ogg")


@dataclass(frozen=True)
class FeatureConfig:
    """Parámetros de extracción de características.

    Todo audio se convierte a mono y se remuestrea a ``sample_rate`` antes de
    calcular los MFCC, de modo que entrenamiento e inferencia sean consistentes
    sin importar el formato de origen.
    """

    sample_rate: int = 16_000
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    min_duration_s: float = 0.1

    @property
    def n_features(self) -> int:
        # media temporal de cada coeficiente MFCC
        return self.n_mfcc


DEFAULT_FEATURE_CONFIG = FeatureConfig()
