from __future__ import annotations

import hashlib
import json
from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["ar_1", "ar_2", "intraday_1", "return_5", "return_20", "vol_5", "vol_20",
            "range_1", "range_5", "volume_ratio", "ma_gap"]
CANDIDATES = {"AR1": ("intraday_1",), "Logistic_C0.1": tuple(FEATURES), "Logistic_C1": tuple(FEATURES)}
VERSION = hashlib.sha256(json.dumps({"features": FEATURES, "candidates": list(CANDIDATES),
                                     "label": "next_index_open_to_close", "threshold": 0.5,
                                     "selection_boundary": "target_date_lte_2026-06-30",
                                     "ar_baseline": "linear_ar1_residual_normal_cdf"},
                                    sort_keys=True).encode()).hexdigest()[:12]


def make_research_frame(index: pd.DataFrame) -> pd.DataFrame:
    """Row t contains only information known after close t; label is session t+1."""
    close = index.close.astype(float)
    ret = np.log(close / close.shift(1))
    intraday = np.log(close / index.open.astype(float))
    frame = pd.DataFrame(index=index.index)
    frame["ar_1"] = ret
    frame["ar_2"] = ret.shift(1)
    frame["intraday_1"] = intraday
    frame["return_5"] = ret.rolling(5).sum()
    frame["return_20"] = ret.rolling(20).sum()
    frame["vol_5"] = ret.rolling(5).std()
    frame["vol_20"] = ret.rolling(20).std()
    frame["range_1"] = (index.high - index.low) / close
    frame["range_5"] = frame.range_1.rolling(5).mean()
    frame["volume_ratio"] = index.volume / index.volume.rolling(20).mean()
    frame["ma_gap"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    frame["target_date"] = pd.Series(index.index, index=index.index).shift(-1)
    frame["target_return"] = intraday.shift(-1)
    frame["target_up"] = np.where(frame.target_return.notna(), (frame.target_return > 0).astype(int), np.nan)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def _model(name: str):
    c = 0.1 if name == "Logistic_C0.1" else 1.0
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=42))


def predict_at(frame: pd.DataFrame, signal_date: pd.Timestamp, name: str,
               shuffled: bool = False) -> float:
    # At close t, t-1 is the latest signal row whose next-day outcome is known.
    train = frame.loc[frame.index < signal_date].dropna(subset=["target_up"])
    if len(train) < 250 or train.target_up.nunique() != 2:
        raise ValueError("训练样本不足250行或只含单一类别")
    columns = list(CANDIDATES[name])
    x = train[columns]
    if name == "AR1":
        target = train.target_return.to_numpy(float)
        if shuffled:
            target = np.random.default_rng(20261023).permutation(target)
        estimator = LinearRegression().fit(x, target)
        expected = float(estimator.predict(frame.loc[[signal_date], columns])[0])
        residual_std = float(np.std(target - estimator.predict(x), ddof=2))
        if not np.isfinite(residual_std) or residual_std <= 0:
            return 0.5
        return float(np.clip(0.5 * (1 + erf(expected / (residual_std * sqrt(2)))), 1e-6, 1 - 1e-6))
    y = train.target_up.astype(int).to_numpy()
    if shuffled:
        y = np.random.default_rng(20261023).permutation(y)
    estimator = _model(name).fit(x, y)
    return float(estimator.predict_proba(frame.loc[[signal_date], columns])[0, 1])


def rolling_predictions(frame: pd.DataFrame, start: str, end: str, names: list[str]) -> pd.DataFrame:
    rows = []
    dates = frame.loc[start:end].index
    for signal_date in dates:
        target_date = frame.at[signal_date, "target_date"]
        if pd.isna(target_date):
            continue
        for name in names:
            probability = predict_at(frame, signal_date, name)
            rows.append({"signal_date": signal_date, "target_date": target_date,
                         "model": name, "up_probability": probability,
                         "actual_up": int(frame.at[signal_date, "target_up"]),
                         "actual_return": float(frame.at[signal_date, "target_return"]),
                         "version": VERSION})
    return pd.DataFrame(rows)


def metrics(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {}
    y = rows.actual_up.astype(int).to_numpy()
    p = rows.up_probability.to_numpy(float)
    pred = (p >= 0.5).astype(int)
    result = {"n": len(rows), "accuracy": float(accuracy_score(y, pred)),
              "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
              "brier": float(brier_score_loss(y, p)), "up_rate": float(y.mean())}
    if len(np.unique(y)) == 2:
        observed, forecast = calibration_curve(y, p, n_bins=5, strategy="quantile")
        result["calibration"] = [{"forecast": float(a), "observed": float(b)}
                                  for a, b in zip(forecast, observed)]
    return result


def select_model(predictions: pd.DataFrame, start: str, end: str) -> tuple[str, dict]:
    validation = predictions[predictions.target_date.between(start, end)]
    scores = {name: metrics(group) for name, group in validation.groupby("model")}
    if len(scores) != len(CANDIDATES):
        raise ValueError("候选模型验证不完整")
    # Selection uses forecast quality only; trading results cannot tune this rule.
    winner = sorted(scores, key=lambda n: (-scores[n]["balanced_accuracy"],
                                          scores[n]["brier"], n))[0]
    return winner, scores
