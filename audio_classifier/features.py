"""Extracción de características acústicas."""

from __future__ import annotations

import librosa
import numpy as np

from .audio_io import AudioClip
from .config import DEFAULT_FEATURE_CONFIG, FeatureConfig


def compute_mfcc(clip: AudioClip, config: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> np.ndarray:
    """Matriz MFCC de forma ``(n_mfcc, frames)``."""
    return librosa.feature.mfcc(
        y=clip.signal,
        sr=clip.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )


def extract_features(clip: AudioClip, config: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> np.ndarray:
    """Vector de longitud fija ``(n_mfcc,)``: media temporal de cada coeficiente MFCC.

    Se evaluó añadir la desviación estándar por coeficiente; no mejoró la
    validación cruzada, por lo que se mantiene la representación más simple.
    """
    return compute_mfcc(clip, config).mean(axis=1).astype(np.float32)


def compute_mel_spectrogram_db(
    clip: AudioClip, config: FeatureConfig = DEFAULT_FEATURE_CONFIG, n_mels: int = 128
) -> np.ndarray:
    """Espectrograma Mel en dB, de forma ``(n_mels, frames)``."""
    mel = librosa.feature.melspectrogram(
        y=clip.signal,
        sr=clip.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=n_mels,
    )
    return librosa.power_to_db(mel, ref=np.max)
