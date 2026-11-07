"""Entrenamiento, evaluación, persistencia e inferencia del clasificador."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .audio_io import AudioClip, load_audio
from .config import (
    DEFAULT_FEATURE_CONFIG,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    LABELS,
    MODEL_PATH,
    TRAIN_DIR,
    FeatureConfig,
)
from .dataset import LabeledFile, discover_dataset
from .features import extract_features

BUNDLE_VERSION = 2
ProgressCallback = Callable[[int, int], None]


class ModelNotFoundError(FileNotFoundError):
    """No existe un modelo entrenado en la ruta indicada."""


class IncompatibleModelError(RuntimeError):
    """El archivo de modelo no es compatible con esta versión de la aplicación."""


@dataclass(frozen=True)
class Prediction:
    label: str
    anomaly_probability: float

    @property
    def is_anomaly(self) -> bool:
        return self.label == LABEL_ANOMALY

    @property
    def confidence(self) -> float:
        p = self.anomaly_probability
        return p if self.is_anomaly else 1.0 - p


@dataclass
class TrainingReport:
    n_samples: int
    n_train: int
    n_test: int
    class_counts: dict[str, int]
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    per_section_accuracy: dict[str, float]
    feature_importances: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelBundle:
    estimator: RandomForestClassifier
    feature_config: FeatureConfig
    report: TrainingReport
    trained_at: str
    sklearn_version: str
    bundle_version: int = BUNDLE_VERSION

    # ---------------------------------------------------------------- inference
    def predict_features(self, features: np.ndarray) -> Prediction:
        proba = self.estimator.predict_proba(features.reshape(1, -1))[0]
        p_anomaly = float(proba[list(self.estimator.classes_).index(LABEL_ANOMALY)])
        label = LABEL_ANOMALY if p_anomaly >= 0.5 else LABEL_NORMAL
        return Prediction(label=label, anomaly_probability=p_anomaly)

    def predict_clip(self, clip: AudioClip) -> Prediction:
        return self.predict_features(extract_features(clip, self.feature_config))

    # -------------------------------------------------------------- persistence
    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path


def load_model(path: Path = MODEL_PATH) -> ModelBundle:
    if not path.exists():
        raise ModelNotFoundError(f"No se encontró el modelo en {path}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # pickle de otra versión, archivo corrupto, etc.
        raise IncompatibleModelError(f"No se pudo cargar el modelo: {exc}") from exc
    if not isinstance(bundle, ModelBundle) or bundle.bundle_version != BUNDLE_VERSION:
        raise IncompatibleModelError("El modelo guardado tiene un formato antiguo. Vuelve a entrenarlo.")
    return bundle


# ---------------------------------------------------------------------- training
def _featurize(
    files: list[LabeledFile], config: FeatureConfig, progress: ProgressCallback | None
) -> np.ndarray:
    rows = []
    for i, item in enumerate(files, start=1):
        rows.append(extract_features(load_audio(item.path, config), config))
        if progress:
            progress(i, len(files))
    return np.vstack(rows)


def _new_estimator(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )


def train_model(
    data_dir: Path = TRAIN_DIR,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    test_size: float = 0.2,
    seed: int = 42,
    progress: ProgressCallback | None = None,
) -> ModelBundle:
    """Entrena el clasificador.

    1. Extrae características de todos los archivos etiquetados.
    2. Evalúa sobre un *hold-out* estratificado por sección y clase.
    3. Reentrena el modelo final con el 100 % de los datos.
    """
    files = discover_dataset(data_dir)
    if not files:
        raise ValueError(f"No se encontraron audios etiquetados en {data_dir}")

    y = np.array([f.label for f in files])
    sections = np.array([f.section for f in files])
    missing = set(LABELS) - set(y)
    if missing:
        raise ValueError(f"Faltan ejemplos de la(s) clase(s): {', '.join(sorted(missing))}")

    X = _featurize(files, config, progress)
    strata = np.char.add(np.char.add(sections, "_"), y)
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, stratify=strata, random_state=seed
    )

    clf = _new_estimator(seed).fit(X[idx_train], y[idx_train])
    y_pred = clf.predict(X[idx_test])
    p_anom = clf.predict_proba(X[idx_test])[:, list(clf.classes_).index(LABEL_ANOMALY)]
    y_test = y[idx_test]

    per_section = {
        f"Sección {s}": float(accuracy_score(y_test[mask], y_pred[mask]))
        for s in sorted(set(sections))
        if (mask := sections[idx_test] == s).any()
    }

    final = _new_estimator(seed).fit(X, y)
    report = TrainingReport(
        n_samples=len(y),
        n_train=len(idx_train),
        n_test=len(idx_test),
        class_counts={lbl: int((y == lbl).sum()) for lbl in LABELS},
        accuracy=float(accuracy_score(y_test, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, pos_label=LABEL_ANOMALY, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, pos_label=LABEL_ANOMALY, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, pos_label=LABEL_ANOMALY, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test == LABEL_ANOMALY, p_anom)),
        confusion_matrix=confusion_matrix(y_test, y_pred, labels=list(LABELS)).tolist(),
        per_section_accuracy=per_section,
        feature_importances=final.feature_importances_.tolist(),
    )
    return ModelBundle(
        estimator=final,
        feature_config=config,
        report=report,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sklearn_version=sklearn.__version__,
    )
