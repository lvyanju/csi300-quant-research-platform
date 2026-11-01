from __future__ import annotations
import json
import os
import shutil
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
from .backtest import compare
from .data import align_assets, fetch_asset, load_dividends, load_snapshot
from .gp import Config, FEATURES, make_frame, rolling, fit, evaluate
from .model import metrics, predict_at
from .forward import is_pre_open_forecast

ROOT = Path(__file__).resolve().parents[1]
DATA, OUTPUT = ROOT / 'data', ROOT / 'outputs'
VALIDATION = ('2026-01-01', '2026-06-30')
EVALUATION = ('2026-07-01', '2026-10-23')

def update_data(end=None):
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    if end is None:
        end = now.date().isoformat() if (now.hour, now.minute) >= (15, 30) else (now.date() - pd.Timedelta(days=1)).isoformat()
    DATA.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='market-stage-', dir=DATA) as folder:
        staging = Path(folder)
        for asset in ('index', 'etf'):
            fetch_asset(asset, '2023-07-01', end, staging)
        index, _ = load_snapshot(staging, 'index')
        etf, _ = load_snapshot(staging, 'etf')
        index, etf = align_assets(index, etf)
        load_dividends(DATA / '510300_dividends.csv', etf.index)
        backups = {}
        files = [f'{asset}.{suffix}' for asset in ('index', 'etf') for suffix in ('csv', 'json')]
        try:
            for name in files:
                target = DATA / name
                if target.exists():
                    backup = staging / f'{name}.backup'
                    shutil.copy2(target, backup)
                    backups[name] = backup
                os.replace(staging / name, target)
        except OSError:
            for name, backup in backups.items():
                if backup.exists():
                    os.replace(backup, DATA / name)
            raise
    return end

def _prepare():
    index, imeta = load_snapshot(DATA, 'index')
    etf, emeta = load_snapshot(DATA, 'etf')
    index, etf = align_assets(index, etf)
    return make_frame(index), etf, load_dividends(DATA / '510300_dividends.csv', etf.index), {'index': imeta, 'etf': emeta}

def _baseline(frame, start, end, name='Logistic_C0.1'):
    rows = []
    dates = frame.index[(frame.index >= pd.Timestamp(start)) & (frame.target_date <= pd.Timestamp(end))]
    for date in dates:
        p = predict_at(frame, date, name)
        rows.append({'signal_date': date, 'target_date': frame.at[date, 'target_date'], 'model': name,
                     'signal_up': p >= 0.5, 'up_probability': p, 'actual_up': int(frame.at[date, 'target_up']),
                     'actual_return': float(frame.at[date, 'target_return'])})
    return pd.DataFrame(rows)

def _score(rows, etf, dividends, prefix):
    ledger, trading = compare(rows, etf, dividends)
    ledger.to_csv(OUTPUT / f'{prefix}_etf_ledger.csv', index=False)
    return {'forecast': metrics(rows), 'trading': trading}

def run(config=Config()):
    if (config.train_days, config.predict_days) != (125, 20):
        raise ValueError('主实验窗口预注册为125/20')
    frame, etf, dividends, metadata = _prepare()
    OUTPUT.mkdir(exist_ok=True)
    results = {}
    for label, random_weights in (('rmo', True), ('fmo', False)):
        rows, windows = rolling(frame, VALIDATION[0], EVALUATION[1], config, random_weights)
        rows.to_csv(OUTPUT / f'{label}_predictions.csv', index=False)
        windows.to_csv(OUTPUT / f'{label}_windows.csv', index=False)
        results[label] = {'windows': len(windows)}
        for period, bounds in (('validation', VALIDATION), ('retrospective', EVALUATION)):
            part = rows[rows.target_date.between(*bounds)]
            results[label][period] = _score(part, etf, dividends, f'{label}_{period}')
    rows = _baseline(frame, VALIDATION[0], EVALUATION[1])
    rows.to_csv(OUTPUT / 'hw3_baseline_predictions.csv', index=False)
    results['hw3_baseline'] = {}
    for period, bounds in (('validation', VALIDATION), ('retrospective', EVALUATION)):
        results['hw3_baseline'][period] = _score(rows[rows.target_date.between(*bounds)], etf, dividends, f'hw3_baseline_{period}')
    ar_rows = _baseline(frame, VALIDATION[0], EVALUATION[1], 'AR1')
    ar_rows.to_csv(OUTPUT / 'ar1_predictions.csv', index=False)
    results['ar1'] = {}
    for period, bounds in (('validation', VALIDATION), ('retrospective', EVALUATION)):
        results['ar1'][period] = _score(ar_rows[ar_rows.target_date.between(*bounds)], etf, dividends,
                                        f'ar1_{period}')
    latest = frame.index[-1]
    scheduled_dates = frame.index[frame.index >= pd.Timestamp(VALIDATION[0])]
    block = (len(scheduled_dates) - 1) // config.predict_days
    block_start = scheduled_dates[block * config.predict_days]
    train_latest = frame.loc[frame.index < block_start].dropna(subset=['target_return']).tail(config.train_days)
    rule_latest = fit(train_latest, config, config.seed + block, True)
    next_signal = bool(evaluate(rule_latest, {f: frame.loc[[latest], f].to_numpy(float) for f in FEATURES})[0])
    result = {'latest_experimental_signal': {'signal_date': str(latest.date()), 'direction': '上涨' if next_signal else '下跌或持平',
              'signal_up': next_signal, 'probability_available': False, 'rule': rule_latest,
              'block': block, 'block_start': str(block_start.date()),
              'snapshot_fresh_for_forward_record': is_pre_open_forecast(latest, datetime.now(ZoneInfo('Asia/Shanghai')))}, 'method': 'typed_gp_random_multiobjective_125_20', 'config': config.__dict__,
              'target': 'next_session_index_open_to_close_direction', 'validation': VALIDATION, 'retrospective': EVALUATION,
              'input_features': FEATURES, 'external_market_factor_available': False,
              'snapshot_sha256': {k: v['sha256'] for k, v in metadata.items()}, 'results': results,
              'note': '历史已在第三次作业中可见；2026-07以后是回顾性比较，不是新盲测。GP布尔信号不是校准概率。'}
    (OUTPUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return result
