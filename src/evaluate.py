"""Phase 5: testing and evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

from . import config


def compute_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    per_class_acc = (cm.diagonal() / cm.sum(1)).tolist()
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = sorted(((int(off[i, j]), i, j) for i in range(10) for j in range(10) if off[i, j]), reverse=True)[:5]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "errors": int((np.asarray(y_true) != np.asarray(y_pred)).sum()),
        "per_class_accuracy": per_class_acc,
        "per_class_f1": f1_score(y_true, y_pred, average=None).tolist(),
        "confusion_matrix": cm.tolist(),
        "top_confusions": [{"true": i, "pred": j, "count": n} for n, i, j in pairs],
    }


def text_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)


def baseline_models():
    """Classical machine-learning models used as benchmarks (input: flattened 784-pixel vectors)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    return {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=config.SEED),
        "K-Nearest Neighbours (k=3)": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=config.SEED),
    }
