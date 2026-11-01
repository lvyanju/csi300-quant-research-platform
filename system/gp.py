"""Typed genetic programming for next-session intraday direction.

Numeric trees combine lagged, scale-free features; Boolean trees make decisions.
All inputs on row t are known at close t. Label/return belongs to session t+1.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .model import make_research_frame

FEATURES = ("ar_1", "ar_2", "intraday_1", "return_5", "return_20", "vol_5",
            "vol_20", "range_1", "range_5", "volume_ratio", "ma_gap")
NUM_OPS = ("add", "sub", "min", "max")
BOOL_OPS = ("gt", "lt", "and", "or", "not")
Tree = tuple[Any, ...]


@dataclass(frozen=True)
class Config:
    train_days: int = 125
    predict_days: int = 20
    population: int = 100
    generations: int = 20
    max_depth: int = 5
    seed: int = 20261026
    one_way_cost: float = 0.0005
    features: tuple[str, ...] = FEATURES


def make_frame(index: pd.DataFrame) -> pd.DataFrame:
    frame = make_research_frame(index).copy()
    # Relative inputs make scale-free comparisons; all data are known at close t.
    frame["volume_ratio"] = frame["volume_ratio"] - 1.0
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(FEATURES))


def tree_type(tree: Tree) -> str:
    return "num" if tree[0] in ("f", "k", *NUM_OPS) else "bool"


def depth(tree: Tree) -> int:
    return 1 if tree[0] in ("f", "k") else 1 + max(depth(x) for x in tree[1:])


def size(tree: Tree) -> int:
    return 1 + sum(size(x) for x in tree[1:] if isinstance(x, tuple))


def valid(tree: Tree, features: tuple[str, ...] = FEATURES) -> bool:
    op = tree[0]
    if op == "f":
        return len(tree) == 2 and tree[1] in features
    if op == "k":
        return len(tree) == 2 and isinstance(tree[1], (int, float))
    if op in NUM_OPS:
        return len(tree) == 3 and all(tree_type(x) == "num" and valid(x, features) for x in tree[1:])
    if op in ("gt", "lt"):
        return len(tree) == 3 and all(tree_type(x) == "num" and valid(x, features) for x in tree[1:])
    if op in ("and", "or"):
        return len(tree) == 3 and all(tree_type(x) == "bool" and valid(x, features) for x in tree[1:])
    if op == "not":
        return len(tree) == 2 and tree_type(tree[1]) == "bool" and valid(tree[1], features)
    return False


def random_tree(kind: str, rng: np.random.Generator, remaining: int,
                features: tuple[str, ...] = FEATURES) -> Tree:
    if kind == "num":
        if remaining <= 1 or rng.random() < 0.58:
            return ("f", features[int(rng.integers(len(features)))]) if rng.random() < 0.85 else ("k", float(rng.choice((-0.02, -0.005, 0.0, 0.005, 0.02, 1.0))))
        op = NUM_OPS[int(rng.integers(len(NUM_OPS)))]
        return (op, random_tree("num", rng, remaining - 1, features), random_tree("num", rng, remaining - 1, features))
    if remaining <= 2 or rng.random() < 0.62:
        op = "gt" if rng.random() < 0.5 else "lt"
        return (op, random_tree("num", rng, remaining - 1, features), random_tree("num", rng, remaining - 1, features))
    op = BOOL_OPS[2 + int(rng.integers(3))]
    if op == "not":
        return (op, random_tree("bool", rng, remaining - 1, features))
    return (op, random_tree("bool", rng, remaining - 1, features), random_tree("bool", rng, remaining - 1, features))


def evaluate(tree: Tree, data: dict[str, np.ndarray]) -> np.ndarray:
    op = tree[0]
    if op == "f":
        return data[tree[1]]
    if op == "k":
        return np.full(len(next(iter(data.values()))), tree[1], dtype=float)
    args = [evaluate(x, data) for x in tree[1:]]
    with np.errstate(over="ignore", invalid="ignore"):
        if op == "add": return np.clip(args[0] + args[1], -10, 10)
        if op == "sub": return np.clip(args[0] - args[1], -10, 10)
        if op == "min": return np.minimum(args[0], args[1])
        if op == "max": return np.maximum(args[0], args[1])
        if op == "gt": return args[0] > args[1]
        if op == "lt": return args[0] < args[1]
        if op == "and": return args[0] & args[1]
        if op == "or": return args[0] | args[1]
        if op == "not": return ~args[0]
    raise ValueError(op)


def paths(tree: Tree, kind: str, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    found = [prefix] if tree_type(tree) == kind else []
    for i, child in enumerate(tree[1:], 1):
        if isinstance(child, tuple):
            found.extend(paths(child, kind, prefix + (i,)))
    return found


def at(tree: Tree, path: tuple[int, ...]) -> Tree:
    for i in path:
        tree = tree[i]
    return tree


def replace(tree: Tree, path: tuple[int, ...], replacement: Tree) -> Tree:
    if not path:
        return replacement
    items = list(tree)
    items[path[0]] = replace(items[path[0]], path[1:], replacement)
    return tuple(items)


def vary(parent: Tree, other: Tree, rng: np.random.Generator, max_depth: int,
         features: tuple[str, ...] = FEATURES) -> Tree:
    kind = "bool" if rng.random() < 0.65 else "num"
    locations = paths(parent, kind)
    place = locations[int(rng.integers(len(locations)))]
    if rng.random() < 0.72:
        donor_paths = paths(other, kind)
        replacement = at(other, donor_paths[int(rng.integers(len(donor_paths)))])
    else:
        replacement = random_tree(kind, rng, max(1, max_depth - len(place)), features)
    child = replace(parent, place, replacement)
    return child if depth(child) <= max_depth and valid(child, features) else parent


def components(tree: Tree, data: dict[str, np.ndarray], returns: np.ndarray, cost: float) -> np.ndarray:
    signal = evaluate(tree, data).astype(bool)
    # Include entry/exit transitions; the last row is marked to cash at the end.
    changes = np.count_nonzero(np.diff(np.r_[False, signal, False]))
    daily = signal * returns
    daily = daily.astype(float).copy()
    daily[1:] -= (signal[1:] != signal[:-1]) * cost
    daily[0] -= signal[0] * cost
    daily[-1] -= signal[-1] * cost
    equity = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    max_dd = float(np.max(1 - equity / peak))
    total = float(equity[-1] - 1)
    std = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float(np.mean(daily) / std * np.sqrt(252)) if std > 1e-12 else 0.0
    refdown = total / max(max_dd, 0.01)
    accuracy = float(np.mean(signal == (returns > 0)))
    # Comparable scales specified before seeing evaluation outcomes.
    standardized = np.array([(np.clip(sharpe, -3, 3) - 1.5) / 0.5 + 1,
                             (np.clip(refdown, -3, 5) - 1.0) + 1,
                             (accuracy - 0.515) / 0.015 + 1])
    return standardized - 0.002 * size(tree) - 0.0001 * changes


def fit(train: pd.DataFrame, config: Config, seed: int, random_weights: bool = True) -> Tree:
    if not config.features or set(config.features) - set(FEATURES) or len(set(config.features)) != len(config.features):
        raise ValueError("特征集合为空、重复或不在允许范围内")
    if len(train) != config.train_days or train.target_return.isna().any():
        raise ValueError("训练窗长度或标签不完整")
    rng = np.random.default_rng(seed)
    data = {name: train[name].to_numpy(float) for name in config.features}
    returns = train.target_return.to_numpy(float)
    population = [random_tree("bool", rng, config.max_depth, config.features) for _ in range(config.population)]
    cache: dict[Tree, np.ndarray] = {}
    best: Tree | None = None
    best_base = -np.inf
    for _ in range(config.generations):
        weights = np.full(3, 3.0) + (rng.random(3) if random_weights else 0.0)
        scores = []
        for tree in population:
            if tree not in cache:
                cache[tree] = components(tree, data, returns, config.one_way_cost)
            scores.append(float(np.dot(weights, cache[tree])))
        order = np.argsort(scores)[::-1]
        # Select final rule by the stable base objective, not a lucky random draw.
        for i in order[:5]:
            base = float(np.sum(cache[population[int(i)]]))
            if base > best_base:
                best, best_base = population[int(i)], base
        next_pop = [population[int(i)] for i in order[:4]]
        while len(next_pop) < config.population:
            pick = rng.integers(config.population, size=3)
            parent = population[int(max(pick, key=lambda j: scores[int(j)]))]
            pick2 = rng.integers(config.population, size=3)
            other = population[int(max(pick2, key=lambda j: scores[int(j)]))]
            next_pop.append(vary(parent, other, rng, config.max_depth, config.features))
        population = next_pop
    assert best is not None
    return best


def rolling(frame: pd.DataFrame, start: str, end: str, config: Config,
            random_weights: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = frame.index[(frame.index >= pd.Timestamp(start)) & (frame.target_date <= pd.Timestamp(end))]
    rows, windows = [], []
    for block, offset in enumerate(range(0, len(dates), config.predict_days)):
        block_dates = dates[offset:offset + config.predict_days]
        first = block_dates[0]
        # Label for a signal row before `first` is known at the close of `first`.
        train = frame.loc[frame.index < first].dropna(subset=["target_return"]).tail(config.train_days)
        if len(train) != config.train_days:
            raise ValueError(f"{first.date()}之前不足{config.train_days}个已完成标签")
        if train.target_date.max() > first:
            raise AssertionError("训练使用未来标签")
        seed = config.seed + block
        rule = fit(train, config, seed, random_weights)
        data = {name: frame.loc[block_dates, name].to_numpy(float) for name in config.features}
        signals = evaluate(rule, data).astype(bool)
        rule_text = json.dumps(rule, ensure_ascii=False)
        rule_id = hashlib.sha256(rule_text.encode()).hexdigest()[:12]
        windows.append({"block": block, "train_start": train.index[0], "train_end": train.index[-1],
                        "predict_start": block_dates[0], "predict_end": block_dates[-1],
                        "rule_id": rule_id, "rule": rule_text, "seed": seed})
        for signal_date, signal in zip(block_dates, signals):
            target = frame.at[signal_date, "target_date"]
            rows.append({"signal_date": signal_date, "target_date": target,
                         "model": f"STGP_{'RMO' if random_weights else 'FMO'}_{config.train_days}_{config.predict_days}",
                         "signal_up": bool(signal), "up_probability": float(signal),
                         "actual_up": int(frame.at[signal_date, "target_up"]),
                         "actual_return": float(frame.at[signal_date, "target_return"]),
                         "rule_id": rule_id})
    return pd.DataFrame(rows), pd.DataFrame(windows)
