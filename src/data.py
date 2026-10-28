"""Phase 1–2: data collection and preprocessing for MNIST."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from . import config


def load_mnist():
    """Return (x_train, y_train), (x_test, y_test) as uint8 images (N, 28, 28) and integer labels.

    Uses a local copy at data/mnist.npz if present, otherwise downloads with Keras
    (about 11 MB, cached after the first run)."""
    if config.LOCAL_DATA.exists():
        d = np.load(config.LOCAL_DATA)
        return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])
    from tensorflow import keras
    return keras.datasets.mnist.load_data()


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale pixel values from 0–255 to 0–1 (float32)."""
    return (images.astype("float32") / 255.0)


def add_channel(images: np.ndarray) -> np.ndarray:
    """(N, 28, 28) -> (N, 28, 28, 1): the channel axis a convolutional layer expects."""
    return images[..., np.newaxis]


def prepare():
    """Load, normalise and split into train / validation / test sets.

    The 6,000-image validation set is a stratified sample of the training images, so
    the 10,000 test images are used only once, for the final evaluation."""
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, x_test = normalize(x_train), normalize(x_test)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=config.VALIDATION_SIZE, stratify=y_train, random_state=config.SEED)
    return {"x_train": x_tr, "y_train": y_tr.astype(int), "x_val": x_val, "y_val": y_val.astype(int),
            "x_test": x_test, "y_test": y_test.astype(int)}


def summary(y_train, y_test) -> dict:
    return {"train_images": int(len(y_train)), "test_images": int(len(y_test)),
            "train_counts": np.bincount(y_train, minlength=10).tolist(),
            "test_counts": np.bincount(y_test, minlength=10).tolist()}
