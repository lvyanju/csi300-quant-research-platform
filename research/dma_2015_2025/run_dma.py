"""沪深300 DMA(10,50,10) 日线择时研究；指数单位仅为模拟记账。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/hs300_daily_source.csv"
OUT = ROOT / "outputs"
START, END = pd.Timestamp("2015-01-01"), pd.Timestamp("2025-12-31")
INITIAL = 100_000.0
COMMISSION, SLIPPAGE = 0.0003, 0.0002  # 每笔成交按名义本金收取


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE, parse_dates=["date"])
    needed = {"date", "open", "high", "low", "close", "volume"}
    if not needed.issubset(raw.columns):
        raise ValueError(f"缺少字段：{needed - set(raw.columns)}")
    raw = raw.sort_values("date").set_index("date")
    if raw.index.has_duplicates or raw.index.isna().any() or not raw.index.is_monotonic_increasing:
        raise ValueError("日期重复、缺失或乱序")
    if not np.isfinite(raw[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)).all():
        raise ValueError("行情存在非有限值")
    if (raw[["open", "high", "low", "close"]] <= 0).any().any() or (raw.volume < 0).any():
        raise ValueError("行情存在非法价格或成交量")
    if ((raw.high < raw[["open", "close", "low"]].max(axis=1)) |
        (raw.low > raw[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("OHLC 不一致")
    period = raw.loc[START:END]
    if period.empty or period.index.min().year != 2015 or period.index.max().year != 2025:
        raise ValueError("2015—2025 区间不完整")
    expected = {year for year in range(2015, 2026)}
    if set(period.index.year) != expected or period.groupby(period.index.year).size().min() < 200:
        raise ValueError("目标区间有年份缺失或交易日明显不足")
    return raw


def make_signals(raw: pd.DataFrame) -> pd.DataFrame:
    # 均线使用当日及以前的收盘价；shift(1) 保证本日开盘只使用前一交易日信息。
    ddd = raw.close.rolling(10, min_periods=10).mean() - raw.close.rolling(50, min_periods=50).mean()
    ama = ddd.rolling(10, min_periods=10).mean()
    signal = (ddd > ama).where(ddd.notna() & ama.notna()).shift(1)
    result = raw.loc[START:END, ["open", "close"]].copy()
    result["dma"] = ddd.reindex(result.index)
    result["ama"] = ama.reindex(result.index)
    result["prior_close_signal"] = signal.reindex(result.index)
    if result.prior_close_signal.isna().any():
        raise ValueError("缺少 2014 年预热数据，无法在 2015 年首日形成信号")
    return result


def simulate(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    cash, units = INITIAL, 0.0
    rows = []
    for date, bar in frame.iterrows():
        target_long = bool(bar.prior_close_signal) if strategy == "DMA" else True
        action, cost = "hold", 0.0
        if target_long and units == 0:
            # 指数单位可以取小数，仅为可复现的价格指数代理，不是可交易证券。
            execution = float(bar.open) * (1 + SLIPPAGE)
            units = cash / (execution * (1 + COMMISSION))
            cost = units * execution * COMMISSION + units * float(bar.open) * SLIPPAGE
            cash = 0.0
            action = "buy"
        elif not target_long and units > 0:
            execution = float(bar.open) * (1 - SLIPPAGE)
            proceeds = units * execution
            cost = proceeds * COMMISSION + units * float(bar.open) * SLIPPAGE
            cash = proceeds * (1 - COMMISSION)
            units = 0.0
            action = "sell"
        rows.append({"date": date, "strategy": strategy, "prior_close_signal": target_long,
                     "open": float(bar.open), "close": float(bar.close), "action": action,
                     "cash": cash, "index_units": units, "transaction_cost": cost,
                     "equity": cash + units * float(bar.close)})
    return pd.DataFrame(rows)


def metrics(ledger: pd.DataFrame) -> dict:
    equity = pd.concat([pd.Series([INITIAL]), ledger.equity.reset_index(drop=True)], ignore_index=True)
    daily = equity.pct_change().dropna()
    annualized = float((equity.iloc[-1] / INITIAL) ** (252 / len(ledger)) - 1)
    vol = float(daily.std(ddof=1) * np.sqrt(252))
    return {"days": len(ledger), "net_return": float(equity.iloc[-1] / INITIAL - 1),
            "annualized_return": annualized, "max_drawdown": float((equity / equity.cummax() - 1).min()),
            "annualized_volatility": vol,
            "sharpe_zero_risk_free": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 else None,
            "trades": int(ledger.action.isin(["buy", "sell"]).sum()),
            "total_transaction_cost": float(ledger.transaction_cost.sum()),
            "ending_equity": float(equity.iloc[-1])}


def annual_returns(ledger: pd.DataFrame) -> dict:
    result = {}
    prior = INITIAL
    for year, group in ledger.groupby(ledger.date.dt.year):
        ending = float(group.equity.iloc[-1])
        result[str(year)] = float(ending / prior - 1)
        prior = ending
    return result


def main() -> None:
    raw = load_data()
    signals = make_signals(raw)
    OUT.mkdir(exist_ok=True)
    ledgers = {name: simulate(signals, name) for name in ("DMA", "buy_and_hold")}
    signals.to_csv(OUT / "dma_signals.csv", index_label="date")
    for name, ledger in ledgers.items():
        ledger.to_csv(OUT / f"{name}_ledger.csv", index=False)
    result = {"source_file": str(SOURCE.relative_to(ROOT)),
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "index_code": "sh000300", "start": str(signals.index.min().date()),
              "end": str(signals.index.max().date()), "trading_days": len(signals),
              "method": "SMA(close,10)-SMA(close,50); AMA=SMA(DMA,10); long iff DMA>AMA",
              "execution": "previous close signal, next open execution, long or cash, fractional index units",
              "commission_per_side": COMMISSION, "slippage_per_side": SLIPPAGE,
              "initial_cash": INITIAL,
              "strategies": {name: {"metrics": metrics(ledger), "annual_returns": annual_returns(ledger)}
                             for name, ledger in ledgers.items()}}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
