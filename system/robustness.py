"""Retrospective sensitivity audit. Never used to overwrite the main 125/20 result."""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from .backtest import compare
from .gp import Config, rolling
from .model import metrics
from .pipeline import EVALUATION, OUTPUT, VALIDATION, _prepare

WINDOWS = ((80, 10), (125, 20), (200, 20))
SEEDS = (20261026, 20261027, 20261028, 20261029, 20261030)


def run_sensitivity(base: Config = Config()) -> dict:
    """Audit 3 window choices x 5 seeds on the same historical data.

    This is diagnostic only: later data were already inspected in prior homework.
    """
    frame, etf, dividends, metadata = _prepare()
    OUTPUT.mkdir(exist_ok=True)
    records = []
    for train_days, predict_days in WINDOWS:
        for seed in SEEDS:
            cfg = replace(base, train_days=train_days, predict_days=predict_days, seed=seed)
            rows, windows = rolling(frame, VALIDATION[0], EVALUATION[1], cfg, True)
            for period, bounds in (('validation', VALIDATION), ('retrospective', EVALUATION)):
                part = rows[rows.target_date.between(*bounds)].copy()
                score = metrics(part)
                _, trades = compare(part, etf, dividends)
                records.append({'train_days': train_days, 'predict_days': predict_days,
                                'seed': seed, 'period': period, 'n': score['n'],
                                'accuracy': score['accuracy'],
                                'balanced_accuracy': score['balanced_accuracy'],
                                'etf_net_return': trades['strategy']['net_return'],
                                'etf_max_drawdown': trades['strategy']['max_drawdown'],
                                'trades': trades['strategy']['trades'],
                                'window_count': len(windows)})
    detail = pd.DataFrame(records)
    detail.to_csv(OUTPUT / 'robustness_runs.csv', index=False)
    summary = (detail.groupby(['period', 'train_days', 'predict_days'], as_index=False)
               .agg(seeds=('seed', 'nunique'), accuracy_mean=('accuracy', 'mean'),
                    accuracy_min=('accuracy', 'min'), accuracy_max=('accuracy', 'max'),
                    balanced_accuracy_mean=('balanced_accuracy', 'mean'),
                    etf_return_mean=('etf_net_return', 'mean'),
                    etf_return_min=('etf_net_return', 'min'),
                    etf_return_max=('etf_net_return', 'max'),
                    positive_etf_runs=('etf_net_return', lambda x: int((x > 0).sum()))))
    summary.to_csv(OUTPUT / 'robustness_summary.csv', index=False)
    # This ranking uses validation only. It is descriptive because the later data
    # were already visible when the audit was designed.
    validation = summary[summary.period.eq('validation')]
    best = validation.sort_values(['balanced_accuracy_mean', 'etf_return_mean'], ascending=False).iloc[0]
    result = {'grid': WINDOWS, 'seeds': SEEDS, 'validation_best_by_mean_balanced_accuracy':
              {'train_days': int(best.train_days), 'predict_days': int(best.predict_days)},
              'source_sha256': {k: v['sha256'] for k, v in metadata.items()},
              'warning': '诊断性历史敏感性分析：后段数据此前已被查看；选出的窗口不能称为真正样本外最优。',
              'rows': len(detail)}
    (OUTPUT / 'robustness_protocol.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
