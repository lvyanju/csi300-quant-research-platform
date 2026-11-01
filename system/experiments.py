"""Isolated, reproducible historical experiments for the course platform."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .backtest import compare
from .gp import Config, FEATURES, rolling
from .model import metrics
from .pipeline import EVALUATION, OUTPUT, VALIDATION, _prepare

EXPERIMENTS = OUTPUT / 'experiments'
METHODS = ('RMO', 'FMO')


def validate(config: Config, methods: tuple[str, ...]) -> None:
    if not config.features or len(set(config.features)) != len(config.features) or set(config.features) - set(FEATURES):
        raise ValueError('请选择至少一个允许的特征，且不可重复')
    if not 60 <= config.train_days <= 250 or not 5 <= config.predict_days <= 40:
        raise ValueError('训练窗口须为60—250日，应用窗口须为5—40日')
    if not 8 <= config.population <= 200 or not 2 <= config.generations <= 40:
        raise ValueError('种群须为8—200，代数须为2—40')
    if not 3 <= config.max_depth <= 7 or not 0 <= config.seed <= 2_000_000_000:
        raise ValueError('树深须为3—7，种子须为非负整数')
    if not 0 <= config.one_way_cost <= 0.01:
        raise ValueError('代理单边成本须在0—1%之间')
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError('请选择RMO、FMO之一或两者')


def list_experiments() -> list[dict]:
    if not EXPERIMENTS.exists():
        return []
    results = []
    for path in sorted(EXPERIMENTS.glob('*/summary.json'), reverse=True):
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
            item['path'] = str(path.parent)
            results.append(item)
        except (ValueError, OSError, KeyError):
            continue
    return results


def run_experiment(config: Config, methods: tuple[str, ...] = ('RMO', 'FMO')) -> dict:
    validate(config, methods)
    frame, etf, dividends, metadata = _prepare()
    experiment_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    result = {
        'id': experiment_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'target': 'next_session_index_open_to_close_direction',
        'config': asdict(config), 'methods': methods,
        'periods': {'validation': VALIDATION, 'retrospective': EVALUATION},
        'source_sha256': {key: value['sha256'] for key, value in metadata.items()},
        'results': {},
        'warning': '两段行情均已被查看；动态试验属于回顾性研究，不构成新的前瞻或独立样本外验证。',
    }
    with tempfile.TemporaryDirectory(prefix='.experiment-', dir=EXPERIMENTS) as tmp:
        stage = Path(tmp)
        for method in methods:
            predictions, windows = rolling(frame, VALIDATION[0], EVALUATION[1], config,
                                           random_weights=(method == 'RMO'))
            predictions.to_csv(stage / f'{method.lower()}_predictions.csv', index=False)
            windows.to_csv(stage / f'{method.lower()}_windows.csv', index=False)
            result['results'][method] = {'windows': len(windows)}
            for period, bounds in (('validation', VALIDATION), ('retrospective', EVALUATION)):
                part = predictions[predictions.target_date.between(*bounds)]
                if part.empty:
                    raise ValueError(f'{period}无可评价样本')
                ledger, trading = compare(part, etf, dividends)
                ledger.to_csv(stage / f'{method.lower()}_{period}_etf_ledger.csv', index=False)
                result['results'][method][period] = {'forecast': metrics(part), 'trading': trading}
        (stage / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(stage, EXPERIMENTS / experiment_id)
    return result
