"""Photo-style test: how much does MNIST-style preprocessing matter for real photos?

Each image in sample_digits/ (except the photo-style one) is turned into 3 "photos":
the digit is drawn small, in dark ink, at a random position on a 400 x 400 grey,
noisy paper background. Each photo is then classified twice:
  * plain resize to 28 x 28 and invert (like the starter code), and
  * the MNIST-style preprocessing in src/preprocess.py.

    python -m src.photo_test          # needs models/mnist_cnn.keras (created by training)
"""
import numpy as np
from PIL import Image

from . import config
from .preprocess import prepare_image

SAMPLES = ["drawn_0.png", "drawn_1.png", "drawn_2.png", "drawn_4.png", "drawn_7.png", "drawn_8.png",
           "mnist_test_16_digit9.png", "mnist_test_45_digit5.png", "mnist_test_50_digit6.png", "starter_3.png"]


def make_photo(digit28, rng, size=400):
    """Draw a 28x28 MNIST-style digit (white on black, 0..1) as dark ink on grey paper."""
    paper = rng.uniform(150, 200)
    bg = paper + rng.normal(0, 8, (size, size))
    side = int(rng.integers(60, 111))                       # digit height 60-110 px
    ink = np.asarray(Image.fromarray((digit28 * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR),
                     dtype=np.float32) / 255
    top, left = rng.integers(0, size - side, 2)
    patch = bg[top:top + side, left:left + side]
    bg[top:top + side, left:left + side] = patch * (1 - 0.75 * ink)
    return np.clip(bg, 0, 255).astype(np.uint8)


def naive(photo):
    """Starter-code approach: resize the whole photo to 28 x 28 and invert."""
    small = np.asarray(Image.fromarray(photo).resize((28, 28), Image.BILINEAR), dtype=np.float32)
    return (255 - small) / 255.0


def build_set(seed=config.SEED, per_sample=3):
    rng = np.random.default_rng(seed)
    photos, labels = [], []
    for name in SAMPLES:
        label = int(name.split(".")[0][-1])
        digit = prepare_image(config.SAMPLE_DIR / name)
        for _ in range(per_sample):
            photos.append(make_photo(digit, rng)); labels.append(label)
    return photos, np.array(labels)


def evaluate(predict_fn, seed=config.SEED):
    """predict_fn: maps an (N, 28, 28) float array to class probabilities."""
    photos, labels = build_set(seed)
    x_naive = np.stack([naive(p) for p in photos])
    x_mnist = np.stack([prepare_image(p) for p in photos])
    res = {"n": len(labels),
           "naive_correct": int((predict_fn(x_naive).argmax(1) == labels).sum()),
           "mnist_style_correct": int((predict_fn(x_mnist).argmax(1) == labels).sum())}
    return res, photos


def main():
    from tensorflow import keras
    cnn = keras.models.load_model(config.CNN_PATH)
    res, _ = evaluate(lambda x: cnn.predict(x[..., np.newaxis], verbose=0))
    print(f"Photo-style test images: {res['n']}")
    print(f"  plain resize + invert : {res['naive_correct']} / {res['n']} correct")
    print(f"  MNIST-style preprocess: {res['mnist_style_correct']} / {res['n']} correct")


if __name__ == "__main__":
    main()
