"""End-to-end pipeline: load data, train the dense network and the CNN, evaluate, benchmark, save.

Run from the project root:
    python -m src.train                 # both networks + baselines (≈ 5–10 min on CPU)
    python -m src.train --model mlp     # only the dense network
    python -m src.train --no-baselines  # skip the classical ML benchmarks
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config, data, evaluate, visualize
from .model import build_cnn, build_mlp


def set_seed(seed: int = config.SEED):
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def train_one(name, model, x_tr, y_tr, x_val, y_val, epochs, patience):
    print(f"\n=== Training {name} ===")
    model.summary()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)
    t0 = time.time()
    hist = model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=epochs,
                     batch_size=config.BATCH_SIZE, callbacks=[stop], verbose=2)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return h, time.time() - t0


def evaluate_model(name, model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    probs = model.predict(x_test, verbose=0)
    pred = probs.argmax(1)
    m = evaluate.compute_metrics(y_test, pred)
    m["test_loss"] = float(loss)
    print(f"{name}: test accuracy {acc:.4f}, loss {loss:.4f}, errors {m['errors']} of {len(y_test)}")
    print(evaluate.text_report(y_test, pred))
    return m, probs


def main(which="both", baselines=True):
    set_seed()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    d = data.prepare()
    print(f"Train {d['x_train'].shape}  Validation {d['x_val'].shape}  Test {d['x_test'].shape}")
    (raw_x, raw_y), _ = data.load_mnist()

    visualize.sample_grid(raw_x, raw_y)
    visualize.class_distribution(np.concatenate([d["y_train"], d["y_val"]]), d["y_test"])
    visualize.mean_digits(d["x_train"], d["y_train"])
    visualize.pixel_histogram(d["x_train"])

    results, histories, best_epochs, probs_all = {}, {}, {}, {}
    specs = []
    if which in ("both", "mlp"):
        specs.append(("Dense neural network", build_mlp, lambda x: x, config.MLP_EPOCHS, config.MLP_PATIENCE, config.MLP_PATH))
    if which in ("both", "cnn"):
        specs.append(("Convolutional neural network", build_cnn, data.add_channel, config.CNN_EPOCHS, config.CNN_PATIENCE, config.CNN_PATH))

    for name, builder, shape, epochs, patience, path in specs:
        set_seed()
        model = builder()
        h, secs = train_one(name, model, shape(d["x_train"]), d["y_train"], shape(d["x_val"]), d["y_val"], epochs, patience)
        m, probs = evaluate_model(name, model, shape(d["x_test"]), d["y_test"])
        best = int(np.argmin(h["val_loss"]) + 1)
        m.update(params=int(model.count_params()), epochs_trained=len(h["loss"]), best_epoch=best,
                 train_seconds=round(secs, 1), history=h)
        results[name], histories[name], best_epochs[name], probs_all[name] = m, h, best, probs
        model.save(path)
        print(f"Saved {path}")

    visualize.training_history(histories, best_epochs=best_epochs)
    x_test, y_test = d["x_test"], d["y_test"]
    for k, (name, m) in enumerate(results.items()):
        tag = "mlp" if "Dense" in name else "cnn"
        visualize.confusion(m["confusion_matrix"], f"Confusion matrix – {name.lower()}",
                            f"{'06' if tag == 'mlp' else '07'}_confusion_matrix_{tag}")
    if len(results) == 2:
        visualize.per_class_accuracy({n: m["per_class_accuracy"] for n, m in results.items()})

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    probs = probs_all[best_name]; pred = probs.argmax(1); conf = probs.max(1)
    visualize.prediction_grid(x_test, y_test, pred, conf, f"Predictions on the first 24 test images ({best_name.lower()})",
                              "09_sample_predictions", list(range(24)))
    wrong = np.where(pred != y_test)[0]
    visualize.prediction_grid(x_test, y_test, pred, conf, f"Examples of mistakes ({best_name.lower()})",
                              "10_misclassified_examples", list(wrong[:24]))
    visualize.confidence_histogram(conf, pred == y_test)

    comparison = {n: m["accuracy"] for n, m in results.items()}
    if baselines:
        X_tr = d["x_train"].reshape(len(d["x_train"]), -1); X_te = x_test.reshape(len(x_test), -1)
        for name, clf in evaluate.baseline_models().items():
            t0 = time.time(); clf.fit(X_tr, d["y_train"]); acc = float((clf.predict(X_te) == y_test).mean())
            comparison[name] = acc
            print(f"{name}: test accuracy {acc:.4f} ({time.time() - t0:.0f}s)")
    visualize.model_comparison(dict(sorted(comparison.items(), key=lambda kv: -kv[1])))

    h1, h2 = config.MLP_UNITS; f1, f2 = config.CNN_FILTERS
    visualize.architecture([("in", "Input", "28×28 image"), ("flat", "Flatten", "784 values"),
                            ("dense", "Dense", f"{h1} neurons\nReLU"), ("drop", "Dropout", f"{config.MLP_DROPOUT:.0%}"),
                            ("dense", "Dense", f"{h2} neurons\nReLU"), ("out", "Output", "10 neurons\nSoftmax")],
                           "Dense neural network (main model)", "14_architecture_mlp")
    visualize.architecture([("in", "Input", "28×28×1"), ("conv", "Conv2D", f"{f1} filters 3×3\nReLU → 26×26"),
                            ("pool", "MaxPool", "2×2 → 13×13"), ("conv", "Conv2D", f"{f2} filters 3×3\nReLU → 11×11"),
                            ("pool", "MaxPool", "2×2 → 5×5"), ("flat", "Flatten", "1,600 values"),
                            ("drop", "Dropout", f"{config.CNN_DROPOUT:.0%}"), ("out", "Output", "10 neurons\nSoftmax")],
                           "Convolutional neural network (improved model)", "15_architecture_cnn")

    summary = {"data": data.summary(np.concatenate([d["y_train"], d["y_val"]]), y_test),
               "split": {"train": int(len(d["y_train"])), "validation": int(len(d["y_val"])), "test": int(len(y_test))},
               "hyperparameters": {"batch_size": config.BATCH_SIZE, "learning_rate": config.LEARNING_RATE,
                                   "mlp_units": list(config.MLP_UNITS), "mlp_dropout": config.MLP_DROPOUT,
                                   "mlp_max_epochs": config.MLP_EPOCHS, "mlp_patience": config.MLP_PATIENCE,
                                   "cnn_filters": list(config.CNN_FILTERS), "cnn_dropout": config.CNN_DROPOUT,
                                   "cnn_max_epochs": config.CNN_EPOCHS, "cnn_patience": config.CNN_PATIENCE, "seed": config.SEED},
               "models": results, "comparison": comparison}
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved metrics -> {config.METRICS_PATH}\nFigures -> {config.FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["both", "mlp", "cnn"], default="both")
    ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args()
    main(a.model, not a.no_baselines)
