"""Regression and classification diagnostics."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_curve,
)

from .train import TASK_REGRESSION, TrainResult


@dataclass
class SegmentError:
    segment: str
    value: str
    count: int
    mae: float


@dataclass
class ExamplePrediction:
    actual: float
    predicted: float
    residual: float


@dataclass
class Diagnostics:
    mae: float
    rmse: float
    r2: float
    mape: float
    segment_errors: list[SegmentError] = field(default_factory=list)
    examples_best: list[ExamplePrediction] = field(default_factory=list)
    examples_worst: list[ExamplePrediction] = field(default_factory=list)
    residual_chart: str = ""
    verdict_tone: str = "okay"
    verdict_text: str = ""


@dataclass
class ClassExample:
    actual: str
    predicted: str
    confidence: float


@dataclass
class ClassDiagnostics:
    accuracy: float
    f1_macro: float
    roc_auc: float | None = None
    confusion_matrix_image: str = ""
    per_class: list[dict[str, Any]] = field(default_factory=list)
    class_balance: dict[str, int] = field(default_factory=dict)
    roc_curve: list[tuple[float, float]] = field(default_factory=list)
    high_confidence_wrong: list[ClassExample] = field(default_factory=list)
    verdict_text: str = ""


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def regression_diagnostics(result: TrainResult) -> Diagnostics:
    y_true = np.asarray(result.y_test, dtype=float)
    y_pred = np.asarray(result.y_pred, dtype=float)
    residuals = y_true - y_pred
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(residuals) / denom) * 100) if np.isfinite(denom).any() else float("nan")

    segs: list[SegmentError] = []
    if result.X_test is not None:
        for col in result.categorical_features[:3]:
            if col not in result.X_test.columns:
                continue
            frame = pd.DataFrame({"value": result.X_test[col].astype(str), "abs": np.abs(residuals)})
            for value, grp in frame.groupby("value"):
                if len(grp) >= 5:
                    segs.append(SegmentError(col, value, int(len(grp)), float(grp["abs"].mean())))
        segs.sort(key=lambda s: s.mae, reverse=True)

    order = np.argsort(np.abs(residuals))
    best_examples = [ExamplePrediction(float(y_true[i]), float(y_pred[i]), float(residuals[i])) for i in order[:5]]
    worst_examples = [ExamplePrediction(float(y_true[i]), float(y_pred[i]), float(residuals[i])) for i in order[-5:][::-1]]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(y_pred, residuals, s=16, alpha=0.55)
    ax.axhline(0, color="#222", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals")
    chart = fig_to_data_uri(fig)

    tone = "good" if r2 >= 0.65 else "okay" if r2 >= 0.25 else "weak"
    verdict = f"Best model explains R2={r2:.3f} with MAE={mae:.4g}."
    return Diagnostics(mae, rmse, r2, mape, segs[:12], best_examples, worst_examples, chart, tone, verdict)


def classification_diagnostics(result: TrainResult) -> ClassDiagnostics:
    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.y_pred)
    names = result.class_names or [str(i) for i in sorted(set(y_true).union(set(y_pred)))]
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))) if result.class_names else None)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cm_image = fig_to_data_uri(fig)

    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_class = [
        {"class": names[i] if i < len(names) else str(i), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
        for i in range(len(p))
    ]
    balance = {names[int(k)] if result.class_names and int(k) < len(names) else str(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))}

    roc_points: list[tuple[float, float]] = []
    if result.y_proba is not None and len(names) == 2:
        try:
            fpr, tpr, _ = roc_curve(y_true, result.y_proba[:, 1])
            roc_points = [(float(x), float(y)) for x, y in zip(fpr, tpr)]
        except Exception:
            pass

    wrong: list[ClassExample] = []
    if result.y_proba is not None:
        conf = result.y_proba.max(axis=1)
        bad_idx = np.where(y_true != y_pred)[0]
        ranked = sorted(bad_idx, key=lambda i: conf[i], reverse=True)[:8]
        for i in ranked:
            wrong.append(
                ClassExample(
                    actual=names[int(y_true[i])] if result.class_names else str(y_true[i]),
                    predicted=names[int(y_pred[i])] if result.class_names else str(y_pred[i]),
                    confidence=float(conf[i]),
                )
            )
    verdict = f"Best classifier reached accuracy={acc:.3f} and f1_macro={f1:.3f}."
    return ClassDiagnostics(acc, f1, None, cm_image, per_class, balance, roc_points, wrong, verdict)

