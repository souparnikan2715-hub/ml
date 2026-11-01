"""
evaluation.py - security-oriented metrics.  Nothing here is hard-coded; every number
is computed from real predictions.

Two views of the same predictions
---------------------------------
* Multiclass view : Normal, DoS, DDoS, PortScan ... (per-class report + confusion matrix)
* Binary view     : Normal (0) vs Attack (1)  -> Attack is the "positive" class.
    - False Positive (FP) = normal traffic wrongly flagged as attack (analyst workload)
    - False Negative (FN) = attack traffic that was missed (security risk)

Averaging for multiclass metrics
--------------------------------
* macro    : simple average over classes (rare classes count as much as common ones)
* weighted : average weighted by class size (reflects overall behaviour)
"""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from preprocessing import NORMAL_LABEL, to_binary


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def evaluate_predictions(y_true, y_pred, classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = list(classes)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    bt, bp = to_binary(y_true), to_binary(y_pred)
    tn, fp, fn, tp = confusion_matrix(bt, bp, labels=[0, 1]).ravel()

    metrics = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        # Binary (Normal vs Attack) view
        "binary_accuracy": float(accuracy_score(bt, bp)),
        "binary_precision": float(precision_score(bt, bp, zero_division=0)),
        "binary_recall": float(recall_score(bt, bp, zero_division=0)),
        "binary_f1": float(f1_score(bt, bp, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "binary_confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "per_class": {k: {m: float(v) for m, v in d.items()} for k, d in report.items()
                      if isinstance(d, dict)},
    }
    metrics["selection_score"] = selection_score(metrics)
    return metrics


def selection_score(m):
    """
    Model-selection score used on VALIDATION data only:
        0.40 * attack recall + 0.40 * attack F1 + 0.20 * (1 - false positive rate)
    Attack detection (recall / F1) matters most, false positives are also penalised.
    """
    return float(0.40 * m["binary_recall"] + 0.40 * m["binary_f1"] + 0.20 * (1.0 - m["false_positive_rate"]))


def json_safe(obj):
    """Recursively convert numpy types / NaN so the object can be sent as JSON."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
