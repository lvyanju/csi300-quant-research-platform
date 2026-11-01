from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(predictions: pd.DataFrame, etf: pd.DataFrame, dividends: pd.DataFrame,
             initial_cash: float = 100000.0, commission_rate: float = 0.0003,
             min_commission: float = 5.0, slippage_rate: float = 0.0002,
             threshold: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Long/cash, next-open execution, 100-share lots, T+1 selling."""
    if not etf.index.is_monotonic_increasing or etf.index.has_duplicates:
        raise ValueError("ETF交易日无序或重复")
    signals = predictions.set_index("target_date").up_probability
    if signals.index.has_duplicates or not signals.index.isin(etf.index).all():
        raise ValueError("预测目标日期与ETF行情不一致")
    cash, shares, buy_day, receivable = float(initial_cash), 0, None, 0.0
    records = []
    rights = {}
    for date in signals.index.sort_values():
        row = etf.loc[date]
        opening, closing = float(row.open), float(row.close)
        paid = 0.0
        for _, event in dividends.loc[dividends.pay_date.eq(date)].iterrows():
            paid += rights.get(event.ex_date, 0.0)
        cash += paid
        receivable -= paid
        accrued = 0.0
        for _, event in dividends.loc[dividends.ex_date.eq(date)].iterrows():
            # The right belongs to shares held at the prior record-day close.
            accrued += shares * float(event.cash_per_share)
            rights[event.ex_date] = accrued
        receivable += accrued
        target_long = float(signals.loc[date]) >= threshold
        action, quantity, commission, slippage = "hold", 0, 0.0, 0.0
        if not target_long and shares and buy_day < date:
            quantity = shares
            slippage = quantity * opening * slippage_rate
            proceeds = quantity * opening - slippage
            commission = max(min_commission, proceeds * commission_rate)
            cash += proceeds - commission
            shares = 0
            action = "sell"
        elif target_long and shares == 0:
            unit = opening * (1 + slippage_rate + commission_rate)
            quantity = int(cash // (100 * unit)) * 100
            while quantity > 0:
                slippage = quantity * opening * slippage_rate
                commission = max(min_commission, quantity * opening * commission_rate)
                if quantity * opening + slippage + commission <= cash:
                    break
                quantity -= 100
            if quantity > 0:
                cash -= quantity * opening + slippage + commission
                shares = quantity
                buy_day = date
                action = "buy"
            else:
                slippage = commission = 0.0
        records.append({"date": date, "signal_date": predictions.loc[predictions.target_date.eq(date),
                                                                         "signal_date"].iloc[0],
                        "up_probability": float(signals.loc[date]), "etf_open": opening,
                        "etf_close": closing, "action": action, "quantity": quantity,
                        "cash": cash, "shares": shares, "commission": commission,
                        "slippage": slippage, "dividend_accrued": accrued,
                        "dividend_paid": paid, "receivable": receivable,
                        "equity": cash + shares * closing + receivable})
    ledger = pd.DataFrame(records)
    if ledger.empty:
        raise ValueError("没有可回测的预测")
    summary = summarize(ledger, initial_cash)
    return ledger, summary


def summarize(ledger: pd.DataFrame, initial_cash: float) -> dict:
    equity = pd.concat([pd.Series([initial_cash]), ledger.equity.reset_index(drop=True)], ignore_index=True)
    daily = equity.pct_change().dropna()
    drawdown = equity / equity.cummax() - 1
    volatility = float(daily.std(ddof=1) * np.sqrt(252)) if len(daily) > 1 else 0.0
    return {"days": len(ledger), "net_return": float(equity.iloc[-1] / initial_cash - 1),
            "max_drawdown": float(drawdown.min()),
            "sharpe_zero_rf": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
            if len(daily) > 1 and daily.std(ddof=1) > 0 else None,
            "trades": int(ledger.action.isin(["buy", "sell"]).sum()),
            "turnover_notional": float((ledger.quantity * ledger.etf_open).sum() / initial_cash),
            "total_cost": float((ledger.commission + ledger.slippage).sum()),
            "total_dividend": float(ledger.dividend_accrued.sum()), "final_equity": float(equity.iloc[-1])}


def compare(predictions: pd.DataFrame, etf: pd.DataFrame, dividends: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ledger, strategy = simulate(predictions, etf, dividends)
    _, doubled = simulate(predictions, etf, dividends, commission_rate=0.0006,
                          min_commission=10, slippage_rate=0.0004)
    always = predictions.copy()
    always["up_probability"] = 1.0
    _, buy_hold = simulate(always, etf, dividends)
    result = {"strategy": strategy, "cost_2x": doubled,
              "buy_and_hold": buy_hold, "cash_return": 0.0,
              "excess_vs_buy_hold": strategy["net_return"] - buy_hold["net_return"]}
    return ledger, result
