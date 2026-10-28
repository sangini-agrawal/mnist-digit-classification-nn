"""Phase 6: prediction system – recognise handwritten digits in your own images.

Examples (run from the project root after `python -m src.train`):
    python -m src.predict                                  # demo on images in sample_digits/
    python -m src.predict --image my_digit.png             # your own photo or drawing
    python -m src.predict --image a.png b.jpg --model mlp  # several images, dense network
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow import keras

from . import config
from .preprocess import prepare_image


def load(which="cnn"):
    path = config.CNN_PATH if which == "cnn" else config.MLP_PATH
    if not path.exists():
        raise SystemExit(f"{path.name} not found. Train first with:  python -m src.train")
    return keras.models.load_model(path)


def predict_digit(image, model, which="cnn"):
    """image: path / PIL image / array. Returns (digit, confidence, top-3 list, 28×28 input)."""
    x = prepare_image(image)
    batch = x[np.newaxis, ..., np.newaxis] if which == "cnn" else x[np.newaxis]
    p = model.predict(batch, verbose=0)[0]
    top = np.argsort(p)[::-1][:3]
    return int(top[0]), float(p[top[0]]), [(int(k), float(p[k])) for k in top], x


def main():
    ap = argparse.ArgumentParser(description="Handwritten digit recogniser")
    ap.add_argument("--image", nargs="*", help="image file(s): PNG/JPG, any size, dark or light background")
    ap.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    a = ap.parse_args()
    files = a.image or sorted(str(p) for p in config.SAMPLE_DIR.glob("*.png"))
    if not files:
        raise SystemExit("No images given and sample_digits/ is empty.")
    model = load(a.model)
    for f in files:
        digit, conf, top3, _ = predict_digit(f, model, a.model)
        alts = ", ".join(f"{d}: {p:.1%}" for d, p in top3[1:])
        print(f"{Path(f).name:<28} → predicted digit {digit}  (confidence {conf:.1%}; next: {alts})")


if __name__ == "__main__":
    main()
