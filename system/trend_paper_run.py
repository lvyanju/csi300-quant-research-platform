"""Yearly walk-forward study of the Sun–Wei hybrid trend method."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .trend_paper_data import (OUTPUT_DIR, TrendConfig, load_dataset, make_samples,
                               segment_prices, validate_config)
from .trend_paper_model import (classification_metrics, fit_head, require_torch)

MODELS = ('hybrid', 'lstm', 'cnn')
YEARS = (2016, 2017, 2018)


def live_input(prices: np.ndarray, config: TrendConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Build one prediction input using prices observable at the signal close."""
    recent = np.asarray(prices[-500:], dtype=float)
    segments = segment_prices(recent, config.error_pct, config.min_segment_bars)
    if len(segments) < 4:
        return None
    previous = segments[-4:]
    trend = np.array([[[s.slope, s.duration_hours] for s in previous]], dtype=np.float32)
    raw = []
    for segment in previous:
        window = recent[max(0, segment.end - 4):segment.end + 1]
        raw.append(np.pad(window, (5 - len(window), 0), mode='edge'))
    raw = np.asarray(raw, dtype=np.float32)
    return trend, np.array([raw - raw[0, 0]], dtype=np.float32)


def prospective_inputs(bars: pd.DataFrame, year: int, config: TrendConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = bars.close.to_numpy(float)
    year_indices = np.flatnonzero(bars.timestamp.dt.year.to_numpy() == year)
    origins, trend, raw = [], [], []
    for index in year_indices[::config.evaluation_step_bars]:
        if index < 500 or index + 1 >= len(bars):
            continue
        value = live_input(prices[:index + 1], config)
        if value is None:
            continue
        origins.append(index)
        trend.append(value[0][0])
        raw.append(value[1][0])
    return np.asarray(origins, dtype=int), np.asarray(trend, dtype=np.float32), np.asarray(raw, dtype=np.float32)


def future_trend_labels(bars: pd.DataFrame, origins: np.ndarray, config: TrendConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ex-post labels only; never passed to a model before its prediction."""
    prices = bars.close.to_numpy(float)
    segments = segment_prices(prices, config.error_pct, config.min_segment_bars)
    starts = np.asarray([s.start for s in segments], dtype=int)
    directions, durations, valid = [], [], []
    for index in origins:
        position = int(np.searchsorted(starts, index, side='right'))
        if position >= len(segments) or bars.timestamp.iloc[segments[position].end].year != bars.timestamp.iloc[index].year:
            directions.append(0)
            durations.append(np.nan)
            valid.append(False)
        else:
            segment = segments[position]
            directions.append(int(segment.slope > 0))
            durations.append(segment.duration_hours)
            valid.append(True)
    return np.asarray(directions, dtype=int), np.asarray(durations, dtype=float), np.asarray(valid, dtype=bool)


def trade_simulation(bars: pd.DataFrame, predictions: pd.DataFrame, config: TrendConfig) -> tuple[pd.DataFrame, dict]:
    """One-notional long/short proxy, close-to-close execution and close-observed stops."""
    prices = bars.close.to_numpy(float)
    timestamps = bars.timestamp.to_numpy()
    trades, next_free = [], -1
    for row in predictions.itertuples():
        start = int(row.bar_index)
        if start < next_free:
            continue
        direction = 1 if row.predicted_up else -1
        hold = int(np.clip(round(float(row.duration_hours) * 12), 1, config.max_hold_bars))
        year = pd.Timestamp(timestamps[start]).year
        end = min(start + hold, len(bars) - 1)
        while end > start and pd.Timestamp(timestamps[end]).year != year:
            end -= 1
        if end <= start:
            continue
        stopped = False
        for point in range(start + 1, end + 1):
            gross = direction * (prices[point] / prices[start] - 1)
            if gross <= -config.stop_loss:
                end = point
                stopped = True
                break
        same_day = pd.Timestamp(timestamps[start]).date() == pd.Timestamp(timestamps[end]).date()
        fee = 0.000023 + (0.00046 if same_day else 0.000023)
        net = direction * (prices[end] / prices[start] - 1) - fee
        trades.append({'signal_time': str(timestamps[start]), 'exit_time': str(timestamps[end]),
                       'direction': direction, 'planned_bars': hold, 'actual_bars': end - start,
                       'stopped': stopped, 'gross_return': direction * (prices[end] / prices[start] - 1),
                       'fee_return': fee, 'net_return': net})
        next_free = end + 1
    ledger = pd.DataFrame(trades)
    if ledger.empty:
        raise ValueError('该年度没有可模拟的交易')
    equity = np.cumprod(1 + ledger.net_return.to_numpy(float))
    peak = np.maximum.accumulate(np.r_[1, equity])[1:]
    ret = ledger.net_return.to_numpy(float)
    standard = float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0
    summary = {'trades': len(ledger), 'net_return': float(equity[-1] - 1),
               'max_drawdown': float(np.min(equity / peak - 1)),
               'sharpe_per_trade': float(np.mean(ret) / standard) if standard > 0 else None,
               'stops': int(ledger.stopped.sum()), 'total_fees_return': float(ledger.fee_return.sum())}
    return ledger, summary


def buy_and_hold_proxy(bars: pd.DataFrame, year: int) -> dict:
    """One-notional long IF continuous-price comparator for the same year."""
    part = bars.loc[bars.timestamp.dt.year.eq(year)]
    prices = part.close.to_numpy(float)
    if len(prices) < 2:
        raise ValueError('买入持有对照缺少年度行情')
    relative = prices / prices[0] - 1
    equity = prices / prices[0]
    peak = np.maximum.accumulate(equity)
    return {'net_return': float(relative[-1] - 0.000046),
            'max_drawdown': float(np.min(equity / peak - 1)),
            'entry_time': str(part.timestamp.iloc[0]), 'exit_time': str(part.timestamp.iloc[-1]),
            'note': '连续价格的一倍名义本金多头代理；扣开平仓代理费，不含真实换月、保证金与滑点。'}


def list_runs() -> list[dict]:
    if not OUTPUT_DIR.exists():
        return []
    result = []
    for path in sorted(OUTPUT_DIR.glob('*/summary.json'), reverse=True):
        try:
            result.append(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            continue
    return result


def run_reproduction(dataset_id: str, config: TrendConfig = TrendConfig()) -> dict:
    validate_config(config)
    bars, metadata = load_dataset(dataset_id)
    if not metadata['coverage']['complete_for_paper']:
        raise ValueError('须提供覆盖2013—2018每年足量交易日的IF五分钟数据，才运行论文逐年复现')
    require_torch()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    result = {'id': run_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'paper': '孙宏鑫、魏先华：基于趋势学习的混合神经网络股指期货预测研究',
              'dataset_id': dataset_id, 'source_sha256': metadata['source_sha256'],
              'normalized_sha256': metadata['normalized_sha256'], 'config': asdict(config),
              'years': {}, 'status': 'completed',
              'caveat': '论文方法复现；合约连续化、分段阈值与部分网络细节由本实现明确设定。'
                        '方向/时长标签由测试后行情定义，仅用于事后评分；预测输入仅使用信号时刻前行情。'
                        '模拟为连续价格的一倍名义本金、多空、收盘成交、收盘观察止损代理，不是可执行期货订单。'}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.trend-run-', dir=OUTPUT_DIR) as folder:
        stage = Path(folder)
        all_predictions = []
        for year in YEARS:
            history = bars[(bars.timestamp.dt.year >= year - config.train_years) &
                           (bars.timestamp.dt.year < year)]
            if history.empty:
                raise ValueError(f'{year}缺少前三年训练数据')
            samples = make_samples(history.close.to_numpy(float), config)
            if len(samples['direction']) < 100:
                raise ValueError(f'{year}训练趋势样本不足100')
            origins, x_trend, x_raw = prospective_inputs(bars, year, config)
            if len(origins) < 30:
                raise ValueError(f'{year}可评价的因果预测点不足30')
            y_direction, y_duration, valid = future_trend_labels(bars, origins, config)
            origins, x_trend, x_raw = origins[valid], x_trend[valid], x_raw[valid]
            y_direction, y_duration = y_direction[valid], y_duration[valid]
            result['years'][str(year)] = {'train_first': str(history.timestamp.iloc[0]),
                                          'train_last': str(history.timestamp.iloc[-1]),
                                          'train_samples': len(samples['direction']),
                                          'buy_and_hold_proxy': buy_and_hold_proxy(bars, year), 'models': {}}
            for kind in MODELS:
                direction_head = fit_head(samples, kind, 'direction', config)
                duration_head = fit_head(samples, kind, 'duration', config)
                probability = direction_head.predict(x_trend, x_raw)
                duration = duration_head.predict(x_trend, x_raw)
                predictions = pd.DataFrame({'bar_index': origins, 'signal_time': bars.timestamp.iloc[origins].to_numpy(),
                                            'model': kind, 'up_probability': probability,
                                            'threshold': direction_head.threshold,
                                            'predicted_up': probability >= direction_head.threshold,
                                            'duration_hours': duration, 'actual_next_trend_up': y_direction,
                                            'actual_next_trend_hours': y_duration})
                all_predictions.append(predictions)
                trades, trading = trade_simulation(bars, predictions, config)
                trades.to_csv(stage / f'{year}_{kind}_trades.csv', index=False)
                classification = classification_metrics(y_direction, probability, direction_head.threshold)
                duration_rmse = float(np.sqrt(np.mean((duration - y_duration) ** 2)))
                result['years'][str(year)]['models'][kind] = {
                    'direction': classification, 'duration_rmse_hours': duration_rmse,
                    'duration_validation_rmse_hours': duration_head.validation_loss,
                    'trading': trading}
        pd.concat(all_predictions, ignore_index=True).to_csv(stage / 'predictions.csv', index=False)
        (stage / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(stage, OUTPUT_DIR / run_id)
    return result
