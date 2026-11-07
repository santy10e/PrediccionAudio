import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_classifier.audio_io import AudioLoadError, load_audio
from audio_classifier.config import DEFAULT_FEATURE_CONFIG, LABEL_ANOMALY, LABEL_NORMAL
from audio_classifier.dataset import discover_dataset, parse_filename
from audio_classifier.features import extract_features


def _wav_bytes(signal: np.ndarray, sr: int, subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def _tone(sr: int, seconds: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)


class TestLoadAudio:
    def test_mono_target_rate(self):
        clip = load_audio(_wav_bytes(_tone(16_000), 16_000))
        assert clip.signal.ndim == 1
        assert clip.sample_rate == 16_000
        assert clip.signal.dtype == np.float32
        assert clip.duration == pytest.approx(1.0, abs=1e-3)

    def test_stereo_is_downmixed(self):
        stereo = np.stack([_tone(16_000), _tone(16_000)], axis=1)
        clip = load_audio(_wav_bytes(stereo, 16_000))
        assert clip.signal.ndim == 1
        assert clip.original_channels == 2

    def test_resamples_to_target_rate(self):
        clip = load_audio(_wav_bytes(_tone(44_100), 44_100))
        assert clip.original_sample_rate == 44_100
        assert clip.sample_rate == DEFAULT_FEATURE_CONFIG.sample_rate
        assert clip.duration == pytest.approx(1.0, abs=1e-2)

    def test_file_like_read_twice(self):
        buf = io.BytesIO(_wav_bytes(_tone(16_000), 16_000))
        first = load_audio(buf)
        second = load_audio(buf)  # el puntero se reinicia
        np.testing.assert_array_equal(first.signal, second.signal)

    def test_invalid_bytes_raise(self):
        with pytest.raises(AudioLoadError):
            load_audio(b"esto no es audio")

    def test_too_short_raises(self):
        with pytest.raises(AudioLoadError, match="corto"):
            load_audio(_wav_bytes(_tone(16_000, seconds=0.01), 16_000))


class TestFeatures:
    def test_fixed_length_regardless_of_format(self):
        mono = extract_features(load_audio(_wav_bytes(_tone(16_000, 2.0), 16_000)))
        stereo_hi = extract_features(
            load_audio(_wav_bytes(np.stack([_tone(48_000, 0.5)] * 2, axis=1), 48_000))
        )
        assert mono.shape == stereo_hi.shape == (DEFAULT_FEATURE_CONFIG.n_features,)
        assert np.all(np.isfinite(mono))

    def test_silence_is_finite(self):
        feats = extract_features(load_audio(_wav_bytes(np.zeros(16_000), 16_000)))
        assert np.all(np.isfinite(feats))


class TestDataset:
    @pytest.mark.parametrize(
        ("name", "label", "section"),
        [
            ("section_00_source_test_normal_0062.wav", LABEL_NORMAL, "00"),
            ("section_02_source_test_anomaly_0003.wav", LABEL_ANOMALY, "02"),
            ("section_00_target_test_anomaly_0000.wav", LABEL_ANOMALY, "00"),
        ],
    )
    def test_parse_filename(self, name, label, section):
        item = parse_filename(Path(name))
        assert item is not None
        assert (item.label, item.section) == (label, section)

    @pytest.mark.parametrize("name", ["unknown_01.wav", "grabacion.wav", "section_xx_normal.wav"])
    def test_unlabeled_files_are_ignored(self, name):
        assert parse_filename(Path(name)) is None

    def test_discover_dataset(self, tmp_path):
        for name in ["section_00_source_test_normal_0001.wav", "section_00_source_test_anomaly_0001.wav"]:
            sf.write(tmp_path / name, _tone(16_000), 16_000)
        (tmp_path / "notas.txt").write_text("x")
        sf.write(tmp_path / "sin_etiqueta.wav", _tone(16_000), 16_000)
        found = discover_dataset(tmp_path)
        assert sorted(f.label for f in found) == [LABEL_ANOMALY, LABEL_NORMAL]
