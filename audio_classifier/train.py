"""CLI de entrenamiento: ``python -m audio_classifier.train``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import MODEL_PATH, TRAIN_DIR
from .model import train_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena el clasificador de audio.")
    parser.add_argument("--data-dir", type=Path, default=TRAIN_DIR, help="Carpeta con audios etiquetados")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Ruta del modelo de salida")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción de evaluación (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    def progress(done: int, total: int) -> None:
        print(f"\rExtrayendo características {done}/{total}", end="", flush=True)

    bundle = train_model(args.data_dir, test_size=args.test_size, seed=args.seed, progress=progress)
    path = bundle.save(args.output)
    r = bundle.report
    print(
        f"\n\nModelo guardado en {path}\n"
        f"  Muestras        {r.n_samples} (train {r.n_train} / test {r.n_test})\n"
        f"  Accuracy        {r.accuracy:.3f}\n"
        f"  Balanced acc.   {r.balanced_accuracy:.3f}\n"
        f"  Precision (anom) {r.precision:.3f}\n"
        f"  Recall (anom)   {r.recall:.3f}\n"
        f"  F1 (anom)       {r.f1:.3f}\n"
        f"  ROC AUC         {r.roc_auc:.3f}"
    )
    for section, acc in r.per_section_accuracy.items():
        print(f"  {section:<15} {acc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
