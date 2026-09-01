"""Training, model banks, split logic, and feature importance."""

from __future__ import annotations

import math
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .preprocess import find_time_column

TASK_REGRESSION = "regression"
TASK_BINARY = "binary_classification"
TASK_MULTICLASS = "multiclass_classification"


@dataclass
class ModelRun:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    seconds: float


@dataclass
class TrainResult:
    task_type: str
    target: str
    best_model: str
    best_pipeline: Pipeline
    leaderboard: list[ModelRun]
    feature_importance: list[tuple[str, float]] = field(default_factory=list)
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | np.ndarray | None = None
    y_test: pd.Series | np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_proba: np.ndarray | None = None
    split_boundary: str = "random"
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    verdict_text: str = ""
    verdict_tone: str = "okay"
    beats_baseline: bool = False


def _normalize_task(task_type: str | None) -> str | None:
    if not task_type:
        return None
    task = task_type.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "binary": TASK_BINARY,
        "binary_classification": TASK_BINARY,
        "multiclass": TASK_MULTICLASS,
        "multiclass_classification": TASK_MULTICLASS,
        "classification": TASK_MULTICLASS,
        "regression": TASK_REGRESSION,
    }
    return aliases.get(task, task)


def detect_task_type(y: pd.Series, forced: str | None = None) -> str:
    forced_norm = _normalize_task(forced)
    if forced_norm in {TASK_REGRESSION, TASK_BINARY, TASK_MULTICLASS}:
        return forced_norm
    s = y.dropna()
    unique = int(s.nunique(dropna=True))
    if unique == 2:
        return TASK_BINARY
    numeric = pd.api.types.is_numeric_dtype(s)
    if numeric:
        float_like = pd.api.types.is_float_dtype(s) and not np.allclose(s.dropna() % 1, 0)
        if unique > 15 or float_like:
            return TASK_REGRESSION
    if 2 < unique <= 15:
        return TASK_MULTICLASS
    return TASK_REGRESSION


def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


_ID_LIKE_RE = re.compile(r"(^|[_\s-])(id|uuid|guid|vault_id|_id)([_\s-]|$)|index|no$", re.IGNORECASE)


def _is_id_like(name: str) -> bool:
    lower = name.lower().strip()
    if lower in {"id", "_id", "uuid", "guid", "vault_id", "row_id", "record_id", "index", "idx", "no", "sno", "sr_no"}:
        return True
    return bool(_ID_LIKE_RE.search(lower))


def _is_time_like(name: str, series: pd.Series) -> bool:
    lower = name.lower()
    if any(k in lower for k in ("year", "date", "month", "quarter", "time", "week", "day")):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        nums = pd.to_numeric(series, errors="coerce").dropna()
        return len(nums) > 0 and float(nums.between(1900, 2100).mean()) >= 0.85
    return False


def _feature_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    # ID-like columns are excluded outright: they carry no generalizable signal,
    # and letting trees split on them produces models that memorize row order.
    features = [
        c
        for c in df.columns
        if c != target
        and df[c].notna().any()
        and not _is_id_like(c)
        # a column with a unique value per row is an identifier in disguise —
        # EXCEPT time axes (year/date), which are legitimate trend features
        and not (
            df[c].nunique(dropna=True) == len(df)
            and len(df) > 20
            and not _is_time_like(c, df[c])
        )
    ]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot())]),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def _find_entity_column(df: pd.DataFrame, exclude: set[str]) -> str | None:
    """Find a panel/entity column — a repeating categorical that identifies a physical
    unit (stall, school, block, vehicle) appearing across multiple rows. Random-splitting
    on data like this leaks: the same unit lands in both train and test and the model
    merely memorizes its rows, producing fake R2=1.0."""
    best, best_score = None, 1.0
    n = len(df)
    for c in df.columns:
        if c in exclude:
            continue
        if _is_id_like(c) or _is_time_like(c, df[c]):
            continue
        n_uniq = df[c].nunique(dropna=True)
        if n_uniq < 2 or n_uniq > n / 2:
            continue
        repeat = n / n_uniq  # avg rows per group
        if 2.0 <= repeat <= n / 3:
            if repeat > best_score:
                best_score = repeat
                best = c
    return best


def _time_split(df: pd.DataFrame, target: str, holdout_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    # exact-duplicate rows are pure leakage: remove them before any split
    if df.duplicated().any():
        df = df.drop_duplicates()
    X = df.drop(columns=[target])
    y = df[target]
    time_col = find_time_column(df)
    if not time_col:
        # no time axis: use a group split whenever a repeating entity column exists,
        # so a physical unit never straddles train/test; otherwise random split.
        entity = _find_entity_column(df, {target})
        if entity:
            from sklearn.model_selection import GroupShuffleSplit

            groups = df[entity].values
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
            return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], f"group:{entity}"
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test, "random"

    dates = pd.to_datetime(df[time_col], errors="coerce")
    valid = dates.notna()
    if valid.sum() < 10 or len(df) < 30:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test, "random"

    max_date = dates.max()
    boundary = max_date - pd.DateOffset(months=holdout_months + 1)
    train_mask = dates <= boundary
    test_mask = dates > boundary
    if test_mask.sum() < 10 or test_mask.sum() > len(df) / 2:
        order = dates.sort_values().index
        cut = max(1, int(len(order) * 0.8))
        train_idx = order[:cut]
        test_idx = order[cut:]
        return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx], f"chrono-{cut}/{len(order)}"
    return X.loc[train_mask], X.loc[test_mask], y.loc[train_mask], y.loc[test_mask], boundary.strftime("%Y-%m")


def _regression_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        # naive baseline first: if nothing beats this, the fancy models add no value
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=100.0, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=120, max_depth=18, n_jobs=-1, random_state=0),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=120, max_depth=18, n_jobs=-1, random_state=0),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=400, max_depth=8, learning_rate=0.08, random_state=0),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBRegressor"] = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            n_jobs=-1,
            random_state=0,
            tree_method="hist",
            objective="reg:squarederror",
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        models["LGBMRegressor"] = LGBMRegressor(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.08,
            n_jobs=-1,
            random_state=0,
            verbose=-1,
        )
    except Exception:
        pass
    return models


def _classification_models(task_type: str) -> dict[str, Any]:
    models: dict[str, Any] = {
        # majority-class baseline for an honest reference point
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=120, max_depth=18, n_jobs=-1, random_state=0),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, random_state=0),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBClassifier"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            n_jobs=-1,
            random_state=0,
            tree_method="hist",
            eval_metric="logloss" if task_type == TASK_BINARY else "mlogloss",
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["LGBMClassifier"] = LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.08,
            n_jobs=-1,
            random_state=0,
            verbose=-1,
        )
    except Exception:
        pass
    return models


def _metrics(task_type: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, float]:
    if task_type == TASK_REGRESSION:
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(rmse),
            "R2": float(r2_score(y_true, y_pred)),
        }
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if task_type == TASK_BINARY and y_proba is not None:
        try:
            probs = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        except Exception:
            pass
    return metrics


def _predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    try:
        if hasattr(pipe, "predict_proba"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
                return pipe.predict_proba(X)
    except Exception:
        return None
    return None


def _sort_key(task_type: str, run: ModelRun) -> float:
    m = run.metrics
    if task_type == TASK_REGRESSION:
        return m.get("RMSE", float("inf"))
    if task_type == TASK_BINARY:
        return -(m.get("roc_auc", m.get("acc", 0.0)))
    return -(m.get("f1", m.get("acc", 0.0)))


def _make_pipeline(model: Any, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(numeric, categorical)), ("model", model)])


def _add_stack_model(
    task_type: str,
    leaderboard: list[ModelRun],
    model_bank: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    y_test: np.ndarray | pd.Series,
) -> ModelRun | None:
    nonlinear = [r for r in leaderboard if r.name not in {"Ridge", "Lasso", "LogisticRegression"} and not r.name.startswith("Baseline")]
    nonlinear = sorted(nonlinear, key=lambda r: _sort_key(task_type, r))[:3]
    if len(nonlinear) < 2:
        return None
    estimators = [(r.name.lower(), clone(model_bank[r.name])) for r in nonlinear if r.name in model_bank]
    if len(estimators) < 2:
        return None
    model = (
        StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), n_jobs=-1)
        if task_type == TASK_REGRESSION
        else StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1)
    )
    pipe = _make_pipeline(model, numeric, categorical)
    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
    proba = _predict_proba(pipe, X_test)
    return ModelRun("StackedTop3", pipe, _metrics(task_type, np.asarray(y_test), pred, proba), time.time() - start)


def _print_model_line(task_type: str, run: ModelRun) -> None:
    if task_type == TASK_REGRESSION:
        print(
            f"[{run.name}] MAE={run.metrics.get('MAE', float('nan')):.4g} "
            f"RMSE={run.metrics.get('RMSE', float('nan')):.4g} "
            f"R2={run.metrics.get('R2', float('nan')):.4g} ({run.seconds:.1f}s)"
        )
    else:
        print(
            f"[{run.name}] roc_auc={run.metrics.get('roc_auc', float('nan')):.4g} "
            f"acc={run.metrics.get('acc', float('nan')):.4g} "
            f"f1={run.metrics.get('f1', float('nan')):.4g} ({run.seconds:.1f}s)"
        )


def _compute_importance(
    best: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    task_type: str,
) -> list[tuple[str, float]]:
    if len(X_test) == 0:
        return []
    sample_n = min(2000, len(X_test))
    Xs = X_test.sample(sample_n, random_state=0) if len(X_test) > sample_n else X_test
    if isinstance(y_test, pd.Series):
        ys = np.asarray(y_test.loc[Xs.index])
    else:
        ys = np.asarray(pd.Series(y_test, index=X_test.index).loc[Xs.index])
    scoring = "r2" if task_type == TASK_REGRESSION else "accuracy"
    print("top features:")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            result = permutation_importance(best, Xs, ys, n_repeats=5, random_state=0, scoring=scoring, n_jobs=-1)
        pairs = sorted(zip(Xs.columns, result.importances_mean), key=lambda x: x[1], reverse=True)[:10]
        for name, value in pairs[:8]:
            print(f"- {name}: {value:.4g}")
        return [(str(name), float(value)) for name, value in pairs]
    except Exception as exc:
        print(f"- skipped: {exc}")
        return []


def train_models(df: pd.DataFrame, target: str, task_type: str | None = None) -> TrainResult:
    if target not in df.columns:
        raise ValueError(f"target column not found: {target}")
    clean = df.dropna(subset=[target]).copy()
    detected = detect_task_type(clean[target], task_type)
    if detected == TASK_REGRESSION:
        clean[target] = pd.to_numeric(clean[target], errors="coerce")
        clean = clean.dropna(subset=[target])

    numeric, categorical = _feature_columns(clean, target)
    print(f"features: {len(numeric)} numeric + {len(categorical)} categorical = {len(numeric) + len(categorical)} total.")
    X_train, X_test, y_train_raw, y_test_raw, boundary = _time_split(clean, target)
    print(f"split: {boundary} ({len(X_train)} train / {len(X_test)} test).")

    label_encoder: LabelEncoder | None = None
    class_names: list[str] = []
    y_train: np.ndarray | pd.Series = y_train_raw
    y_test: np.ndarray | pd.Series = y_test_raw
    if detected != TASK_REGRESSION:
        label_encoder = LabelEncoder()
        all_y = pd.concat([pd.Series(y_train_raw), pd.Series(y_test_raw)], ignore_index=True).astype(str)
        label_encoder.fit(all_y)
        y_train = label_encoder.transform(pd.Series(y_train_raw).astype(str))
        y_test = label_encoder.transform(pd.Series(y_test_raw).astype(str))
        class_names = label_encoder.classes_.astype(str).tolist()

    model_bank = _regression_models() if detected == TASK_REGRESSION else _classification_models(detected)
    leaderboard: list[ModelRun] = []
    for name, model in model_bank.items():
        try:
            pipe = _make_pipeline(model, numeric, categorical)
            start = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
            proba = _predict_proba(pipe, X_test)
            run = ModelRun(name, pipe, _metrics(detected, np.asarray(y_test), pred, proba), time.time() - start)
            leaderboard.append(run)
            _print_model_line(detected, run)
        except Exception as exc:
            print(f"[{name}] skipped: {exc}")

    stack = None
    try:
        stack = _add_stack_model(detected, leaderboard, model_bank, numeric, categorical, X_train, X_test, y_train, y_test)
    except Exception as exc:
        print(f"[StackedTop3] skipped: {exc}")
    if stack:
        leaderboard.append(stack)
        _print_model_line(detected, stack)
    if not leaderboard:
        raise RuntimeError("no models trained successfully")

    leaderboard.sort(key=lambda r: _sort_key(detected, r))
    best = leaderboard[0]
    baseline = next((r for r in leaderboard if r.name.startswith("Baseline")), None)

    # honest verdicts: compare the winner to the naive baseline and flag
    # small-sample runs. A model that can't beat "predict the average" has no
    # value, and 26-row test sets deserve an exploratory warning, not a
    # confident headline.
    beats_baseline = False
    if baseline is not None:
        if detected == TASK_REGRESSION:
            beats_baseline = _sort_key(detected, best) < _sort_key(detected, baseline) - 1e-9
        else:
            beats_baseline = _sort_key(detected, best) < _sort_key(detected, baseline) - 1e-9

    r2_val = float(best.metrics.get("R2", float("nan"))) if detected == TASK_REGRESSION else float("nan")
    n_train = len(X_train)
    if not beats_baseline and baseline is not None:
        if detected == TASK_REGRESSION:
            verdict_text = (
                f"NOT USABLE: no model beat the naive mean baseline (best {best.name} RMSE={best.metrics.get('RMSE', float('nan')):.4g} "
                f"vs baseline {baseline.metrics.get('RMSE', float('nan')):.4g}). The available features carry no predictive signal for "
                f"{target}. Treat any numbers below as descriptive statistics only."
            )
        else:
            verdict_text = (
                f"NOT USABLE: no model beat the naive majority-class baseline (best {best.name} "
                f"acc={best.metrics.get('accuracy', float('nan')):.3f} vs baseline {baseline.metrics.get('accuracy', float('nan')):.3f}). "
                f"The features add no discrimination for {target} over just predicting the majority class."
            )
        verdict_tone = "unusable"
    elif detected == TASK_REGRESSION and r2_val == r2_val and r2_val < 0.1:
        verdict_text = (
            f"WEAK: best model barely explains variance (R2={r2_val:.3f}, MAE={best.metrics.get('MAE', float('nan')):.4g}). "
            f"Marginal improvement over a naive guess — do not rely on these predictions."
        )
        verdict_tone = "weak"
    elif n_train < 30:
        verdict_text = (
            f"EXPLORATORY: trained on only {n_train} rows. Metrics are unstable at this sample size and the split leaves very few "
            f"test rows. Directionally interesting at most — do not use for decisions."
        )
        verdict_tone = "exploratory"
    else:
        verdict_text = (
            f"USABLE with caution: best model {best.name} beats the naive baseline "
            f"({best.metrics.get('RMSE', float('nan')):.4g} vs {baseline.metrics.get('RMSE', float('nan')):.4g} RMSE)"
            if baseline is not None
            else f"Best model explains R2={r2_val:.3f} with MAE={best.metrics.get('MAE', float('nan')):.4g}."
        )
        if detected == TASK_REGRESSION and r2_val == r2_val:
            verdict_text += f" R2={r2_val:.3f} on held-out data."
        verdict_tone = "good" if (detected != TASK_REGRESSION or r2_val >= 0.5) else "okay"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        y_pred = best.pipeline.predict(X_test)
    y_proba = _predict_proba(best.pipeline, X_test)
    importance = _compute_importance(best.pipeline, X_test, y_test, detected)
    return TrainResult(
        task_type=detected,
        target=target,
        best_model=best.name,
        best_pipeline=best.pipeline,
        leaderboard=leaderboard,
        feature_importance=importance,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        y_proba=y_proba,
        split_boundary=boundary,
        numeric_features=numeric,
        categorical_features=categorical,
        class_names=class_names,
        verdict_text=verdict_text,
        verdict_tone=verdict_tone,
        beats_baseline=beats_baseline,
    )
