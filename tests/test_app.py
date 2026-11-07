"""Pruebas de humo de la interfaz con ``streamlit.testing``."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from audio_classifier.config import MODEL_PATH

APP = str(Path(__file__).resolve().parent.parent / "app.py")

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="requiere el modelo entrenado")


def test_app_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) >= 3


def test_sample_analysis_shows_result():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.segmented_control[0].set_value("sample").run()
    assert not at.exception
    assert at.selectbox[0].options
    markdown = " ".join(m.value for m in at.markdown)
    assert "Resultado" in markdown
    assert ("Funcionamiento normal" in markdown) or ("Anomalía detectada" in markdown)
