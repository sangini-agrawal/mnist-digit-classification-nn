"""Charts for the notebook, report and presentation."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import config

BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlelocation": "left", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "font.family": "DejaVu Sans",
})


def _save(fig, name, out_dir=None):
    out = Path(out_dir or config.FIGURES_DIR); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"; fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); return p


def sample_grid(x, y, n=30, out_dir=None):
    cols = 10; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.3))
    for ax, img, lbl in zip(axes.ravel(), x[:n], y[:n]):
        ax.imshow(img, cmap="gray"); ax.set_title(str(lbl), loc="center", fontsize=12); ax.axis("off")
    fig.suptitle("First 30 training images with their labels", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(); return _save(fig, "01_sample_images", out_dir)


def class_distribution(y_train, y_test, out_dir=None):
    tr, te = np.bincount(y_train, minlength=10), np.bincount(y_test, minlength=10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=False)
    for ax, counts, title in [(axes[0], tr, f"Training set ({tr.sum():,} images)"), (axes[1], te, f"Test set ({te.sum():,} images)")]:
        ax.bar(range(10), counts, color=BLUE, width=0.7)
        ax.set_xticks(range(10)); ax.set_xlabel("Digit"); ax.set_ylabel("Images"); ax.set_title(title)
        ax.grid(axis="x", visible=False)
        for i, c in enumerate(counts): ax.text(i, c + counts.max() * 0.01, f"{c:,}", ha="center", va="bottom", fontsize=8.5, color=TEXT_2)
        ax.set_ylim(0, counts.max() * 1.12)
    fig.tight_layout(); return _save(fig, "02_class_distribution", out_dir)


def mean_digits(x, y, out_dir=None):
    fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
    for d, ax in enumerate(axes):
        ax.imshow(x[y == d].mean(0), cmap="magma"); ax.set_title(str(d), loc="center"); ax.axis("off")
    fig.suptitle("The 'average' image of each digit (mean of all training images)", x=0.01, ha="left", fontweight="bold", fontsize=12.5, y=1.08)
    return _save(fig, "03_mean_digits", out_dir)


def pixel_histogram(x, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    vals = (x[:2000].ravel() * 255).astype(int)
    ax.hist(vals, bins=52, color=BLUE)
    ax.set_yscale("log"); ax.set_xlabel("Pixel value (0 = black, 255 = white)"); ax.set_ylabel("Number of pixels (log scale)")
    zero = (vals == 0).mean()
    ax.set_title(f"Pixel values: {zero:.0%} of pixels are pure black background")
    return _save(fig, "04_pixel_histogram", out_dir)


def training_history(histories: dict, out_dir=None, best_epochs: dict | None = None):
    fig, axes = plt.subplots(len(histories), 2, figsize=(12, 3.9 * len(histories)), squeeze=False)
    for r, (name, h) in enumerate(histories.items()):
        ep = np.arange(1, len(h["loss"]) + 1)
        for c, key in enumerate(["accuracy", "loss"]):
            ax = axes[r, c]
            ax.plot(ep, h[key], color=BLUE, label="Training", marker="o", markersize=3)
            ax.plot(ep, h["val_" + key], color=ORANGE, label="Validation", marker="o", markersize=3)
            if best_epochs and name in best_epochs:
                ax.axvline(best_epochs[name], color=TEXT_2, ls="--", lw=1)
            ax.set_xlabel("Epoch"); ax.set_ylabel(key.capitalize()); ax.set_title(f"{name}: {key} per epoch")
            ax.legend(loc="lower right" if key == "accuracy" else "upper right")
    fig.tight_layout(); return _save(fig, "05_training_history", out_dir)


def confusion(cm, title, name, out_dir=None):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    shown = np.log1p(cm)
    ax.imshow(shown, cmap=SEQUENTIAL); ax.grid(False)
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v: ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5 if i != j else 9,
                          fontweight="bold" if i == j else "normal", color="white" if shown[i, j] > shown.max() * 0.6 else TEXT)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit"); ax.set_title(title)
    ax.spines[:].set_visible(False)
    return _save(fig, name, out_dir)


def per_class_accuracy(models: dict, out_dir=None):
    fig, ax = plt.subplots(figsize=(11, 3.9))
    names = list(models); w = 0.38; colors = [BLUE, ORANGE]
    for k, n in enumerate(names):
        vals = np.array(models[n]) * 100
        ax.bar(np.arange(10) + (k - 0.5) * w, vals, width=w - 0.04, color=colors[k], label=n)
    lo = min(min(v) for v in models.values()) * 100
    ax.set_ylim(np.floor(lo - 1.5), 100.3); ax.set_xticks(range(10)); ax.set_xlabel("Digit"); ax.set_ylabel("Correct (%)")
    ax.grid(axis="x", visible=False); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title("Accuracy for each digit on the 10,000 test images (axis starts above 0 to show differences)")
    return _save(fig, "08_per_class_accuracy", out_dir)


def prediction_grid(x, y, pred, conf, title, name, idx, out_dir=None):
    n = len(idx); cols = 8; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.85))
    for ax in axes.ravel(): ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ok = pred[i] == y[i]
        ax.imshow(x[i], cmap="gray")
        ax.set_title(f"pred {pred[i]} ({conf[i]:.0%})\ntrue {y[i]}", loc="center", fontsize=9.5,
                     color=TEXT if ok else RED, fontweight="normal" if ok else "bold")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout(); return _save(fig, name, out_dir)


def confidence_histogram(conf, correct, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(0.2, 1, 33)
    ax.hist([conf[correct], conf[~correct]], bins=bins, color=[BLUE, ORANGE], rwidth=0.9,
            label=[f"Correct ({correct.sum():,})", f"Wrong ({(~correct).sum():,})"])
    ax.set_yscale("log"); ax.set_xlabel("Model confidence (highest softmax probability)"); ax.set_ylabel("Test images (log scale)")
    ax.set_title("Wrong answers usually come with lower confidence"); ax.legend(loc="upper left")
    return _save(fig, "11_confidence_histogram", out_dir)


def model_comparison(results: dict, out_dir=None):
    names = list(results)[::-1]; vals = [results[n] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.4))
    colors = [BLUE if "network" in n.lower() else "#8f8e88" for n in names]
    ax.hlines(range(len(names)), min(vals) - 1, vals, color=GRID, lw=2)
    ax.scatter(vals, range(len(names)), color=colors, s=90, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for i, v in enumerate(vals): ax.text(v + 0.12, i, f"{v:.2f}%", va="center", fontsize=10)
    ax.set_yticks(range(len(names)), names); ax.set_xlim(min(vals) - 1, 100.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test accuracy (%) – axis starts above 0 to show differences")
    ax.set_title("Neural networks vs classical machine learning (10,000 test images)")
    return _save(fig, "12_model_comparison", out_dir)


def architecture(layers, title, name, out_dir=None):
    fig, ax = plt.subplots(figsize=(13, 3.3)); ax.axis("off"); ax.set_xlim(0, len(layers)); ax.set_ylim(0, 1)
    fills = {"in": "#f0efec", "conv": "#cde2fb", "pool": "#e6eefb", "dense": "#9ec5f4", "drop": "#f0efec", "out": "#fbd5c4", "flat": "#f0efec"}
    for i, (kind, t, s) in enumerate(layers):
        x = i + 0.07
        ax.add_patch(plt.Rectangle((x, 0.18), 0.86, 0.64, facecolor=fills[kind], edgecolor="#256abf", linewidth=1.1))
        ax.text(x + 0.43, 0.62, t, ha="center", va="center", fontweight="bold", fontsize=10.5)
        ax.text(x + 0.43, 0.4, s, ha="center", va="center", fontsize=8.8, color=TEXT_2, linespacing=1.35)
        if i < len(layers) - 1:
            ax.annotate("", xy=(i + 1.07, 0.5), xytext=(i + 0.93, 0.5), arrowprops=dict(arrowstyle="-|>", color=TEXT_2, lw=1.3))
    ax.set_title(title, loc="left"); return _save(fig, name, out_dir)


def preprocessing_steps(steps: dict, pred_text: str, out_dir=None):
    keys = [("original", "1. Your image"), ("inverted", "2. Grayscale,\nwhite on black"), ("cropped", "3. Cropped to\nthe digit"), ("final", "4. 28×28, centred\nlike MNIST")]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]})
    for ax, (k, t) in zip(axes, keys):
        ax.imshow(steps[k], cmap="gray"); ax.set_title(t, loc="center", fontsize=10.5); ax.axis("off")
    axes[4].axis("off"); axes[4].text(0.05, 0.5, pred_text, fontsize=12, va="center", fontweight="bold", linespacing=1.6)
    fig.suptitle("Preparing a user's handwritten digit for the model", x=0.01, ha="left", fontweight="bold", fontsize=12.5, y=1.04)
    return _save(fig, "13_preprocessing_pipeline", out_dir)
