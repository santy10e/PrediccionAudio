"""Descubrimiento y etiquetado del dataset de entrenamiento."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import LABEL_ANOMALY, LABEL_NORMAL, SUPPORTED_EXTENSIONS

# Convención de nombres DCASE: section_<id>_<dominio>_<split>_<normal|anomaly>_<n>.wav
_FILENAME_RE = re.compile(r"section_(?P<section>\d+)_.*?_(?P<label>normal|anomaly)_\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class LabeledFile:
    path: Path
    label: str
    section: str


def parse_filename(path: Path) -> LabeledFile | None:
    """Obtiene etiqueta y sección a partir del nombre del archivo, o ``None`` si no aplica."""
    match = _FILENAME_RE.search(path.stem)
    if not match:
        return None
    label = LABEL_NORMAL if match["label"].lower() == "normal" else LABEL_ANOMALY
    return LabeledFile(path=path, label=label, section=match["section"])


def discover_dataset(directory: Path) -> list[LabeledFile]:
    """Lista los archivos de audio etiquetados dentro de ``directory`` (recursivo)."""
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe el directorio de datos: {directory}")
    files = [
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS
    ]
    return [item for item in map(parse_filename, files) if item is not None]
