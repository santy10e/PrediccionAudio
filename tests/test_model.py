import joblib
import numpy as np
import pytest
import soundfile as sf

from audio_classifier.audio_io import load_audio
from audio_classifier.config import LABEL_ANOMALY, LABEL_NORMAL
from audio_classifier.model import (
    IncompatibleModelError,
    ModelNotFoundError,
    load_model,
    train_model,
)


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    """Normal = tono puro grave; anomalía = tono agudo con ruido."""
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    sr = 16_000
    t = np.arange(sr) / sr
    for section in ("00", "01"):
        for i in range(10):
            normal = 0.4 * np.sin(2 * np.pi * (200 + 5 * i) * t)
            anomaly = 0.4 * np.sin(2 * np.pi * (3000 + 50 * i) * t) + 0.2 * rng.standard_normal(sr)
            sf.write(root / f"section_{section}_source_test_normal_{i:04d}.wav", normal, sr)
            sf.write(root / f"section_{section}_source_test_anomaly_{i:04d}.wav", anomaly, sr)
    return root


@pytest.fixture(scope="module")
def bundle(synthetic_dataset):
    return train_model(synthetic_dataset, test_size=0.25)


def test_report_metrics(bundle):
    r = bundle.report
    assert r.n_samples == 40
    assert r.n_train + r.n_test == 40
    assert r.class_counts == {LABEL_NORMAL: 20, LABEL_ANOMALY: 20}
    assert r.accuracy == pytest.approx(1.0)
    assert r.roc_auc == pytest.approx(1.0)
    assert set(r.per_section_accuracy) == {"Sección 00", "Sección 01"}
    assert len(r.feature_importances) == bundle.feature_config.n_features


def test_prediction(bundle, synthetic_dataset):
    clip = load_audio(synthetic_dataset / "section_00_source_test_anomaly_0003.wav")
    pred = bundle.predict_clip(clip)
    assert pred.label == LABEL_ANOMALY
    assert pred.is_anomaly
    assert 0.5 <= pred.confidence <= 1.0
    assert pred.confidence == pytest.approx(pred.anomaly_probability)


def test_progress_callback(synthetic_dataset):
    calls = []
    train_model(synthetic_dataset, progress=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (40, 40)


def test_save_and_load_roundtrip(bundle, tmp_path):
    path = bundle.save(tmp_path / "nested" / "model.joblib")
    loaded = load_model(path)
    assert loaded.trained_at == bundle.trained_at
    x = np.zeros(bundle.feature_config.n_features, dtype=np.float32)
    assert loaded.predict_features(x) == bundle.predict_features(x)


def test_missing_model(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "nope.joblib")


def test_legacy_model_rejected(tmp_path):
    path = tmp_path / "legacy.pkl"
    joblib.dump({"old": "format"}, path)
    with pytest.raises(IncompatibleModelError):
        load_model(path)


def test_single_class_dataset_fails(tmp_path):
    sf.write(tmp_path / "section_00_source_test_normal_0001.wav", np.zeros(16_000), 16_000)
    with pytest.raises(ValueError, match="anomaly"):
        train_model(tmp_path)


def test_empty_dataset_fails(tmp_path):
    with pytest.raises(ValueError, match="No se encontraron"):
        train_model(tmp_path)
