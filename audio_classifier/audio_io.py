"""Lectura y normalización de audio."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf

from .config import DEFAULT_FEATURE_CONFIG, FeatureConfig


class AudioLoadError(ValueError):
    """El archivo no pudo interpretarse como audio válido."""


@dataclass(frozen=True)
class AudioClip:
    """Señal mono normalizada lista para análisis."""

    signal: np.ndarray
    sample_rate: int
    original_sample_rate: int
    original_channels: int

    @property
    def duration(self) -> float:
        return len(self.signal) / self.sample_rate

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.signal)))) if self.signal.size else 0.0

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.signal))) if self.signal.size else 0.0


Source = str | Path | bytes | BinaryIO


def load_audio(source: Source, config: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> AudioClip:
    """Carga audio desde ruta, bytes o archivo en memoria.

    - Convierte a ``float32`` mono (promediando canales).
    - Remuestrea a ``config.sample_rate``.
    - Valida que la señal no esté vacía, sea finita y tenga duración mínima.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif hasattr(source, "seek"):
        source.seek(0)

    try:
        data, sr = sf.read(source, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise AudioLoadError(
            "El archivo no es un audio válido o está dañado. Formatos admitidos: WAV, FLAC y OGG."
        ) from exc

    if data.size == 0:
        raise AudioLoadError("El archivo de audio está vacío.")

    channels = data.shape[1]
    signal = data.mean(axis=1) if channels > 1 else data[:, 0]

    if not np.all(np.isfinite(signal)):
        raise AudioLoadError("El audio contiene valores no válidos (NaN/Inf).")

    if sr != config.sample_rate:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=config.sample_rate)

    clip = AudioClip(
        signal=np.ascontiguousarray(signal, dtype=np.float32),
        sample_rate=config.sample_rate,
        original_sample_rate=int(sr),
        original_channels=int(channels),
    )
    if clip.duration < config.min_duration_s:
        raise AudioLoadError(
            f"El audio es demasiado corto ({clip.duration:.2f} s). Mínimo: {config.min_duration_s} s."
        )
    return clip
