"""Turn a user's handwritten digit image into an MNIST-style 28×28 input.

MNIST digits are white on black, fitted into a 20×20 box and centred by their centre
of mass inside a 28×28 frame. A photo or drawing must go through the same steps,
otherwise the model sees something very different from its training data."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _center_of_mass(a: np.ndarray):
    total = a.sum()
    if total == 0:
        return a.shape[0] / 2, a.shape[1] / 2
    rows, cols = np.indices(a.shape)
    return (rows * a).sum() / total, (cols * a).sum() / total


def prepare_image(source, return_steps: bool = False):
    """source: file path, PIL image or numpy array. Returns a float32 (28, 28) array in [0, 1]."""
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source.astype(np.uint8))
    else:
        img = Image.open(source)
    if img.mode in ("RGBA", "LA"):                       # transparent drawings: paste on white
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    steps = {"original": gray.copy()}

    # 1. MNIST is white digit on black: invert if the background is light
    if np.median(gray) > 127:
        gray = 255.0 - gray
    # remove the paper/background level and faint noise (thresholds adapt to the ink strength)
    gray = np.clip(gray - np.median(gray), 0, None)
    peak = np.percentile(gray, 99.9)
    if peak <= 0:
        raise ValueError("The image looks empty – no digit found.")
    gray[gray < 0.15 * peak] = 0
    steps["inverted"] = gray.copy()

    # 2. Crop to the digit's bounding box (using only clearly inked pixels)
    ys, xs = np.nonzero(gray >= 0.3 * peak)
    if len(ys) == 0:
        raise ValueError("The image looks empty – no digit found.")
    crop = np.clip(gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * (255.0 / peak), 0, 255)
    steps["cropped"] = crop.copy()

    # 3. Resize so the longer side is 20 px, keeping the aspect ratio
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((nw, nh), Image.LANCZOS), dtype=np.float32)

    # 4. Paste into a 28×28 frame, centred by centre of mass
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = small
    cy, cx = _center_of_mass(canvas)
    dy, dx = int(round(14 - cy)), int(round(14 - cx))
    canvas = np.roll(np.roll(canvas, dy, axis=0), dx, axis=1)

    out = np.clip(canvas / max(canvas.max(), 1e-6), 0, 1).astype(np.float32)
    steps["final"] = out
    return (out, steps) if return_steps else out
