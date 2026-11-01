"""Fast daily forecast from a frozen primary model and an experimental GP rule."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .data import align_assets, load_snapshot
from .forward import is_pre_open_forecast, record_and_reconcile
from .gp import Config, FEATURES, evaluate, fit, make_frame
from .model import VERSION, predict_at

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUTPUT = ROOT / 'outputs'
FROZEN = ROOT / 'config' / 'frozen_selection.json'
ANCHOR = pd.Timestamp('2026-01-01')


def _frozen_primary() -> dict:
    frozen = json.loads(FROZEN.read_text(encoding='utf-8'))
    if frozen.get('version') != VERSION or frozen.get('winner') != 'Logistic_C0.1':
        raise ValueError('主模型冻结配置与当前代码不一致，拒绝生成预测')
    return frozen


def _experimental_signal(frame: pd.DataFrame, latest: pd.Timestamp, config: Config) -> dict:
    dates = frame.index[frame.index >= ANCHOR]
    if len(dates) == 0 or dates[-1] != latest:
        raise ValueError('最新日期不在滚动窗口范围内')
    block = (len(dates) - 1) // config.predict_days
    block_start = dates[block * config.predict_days]
    train = frame.loc[frame.index < block_start].dropna(subset=['target_return']).tail(config.train_days)
    if len(train) != config.train_days or train.target_date.max() > block_start:
        raise ValueError('GP训练窗口不足或包含未来标签')
    rule = fit(train, config, config.seed + block, True)
    values = {name: frame.loc[[latest], name].to_numpy(float) for name in FEATURES}
    signal = bool(evaluate(rule, values)[0])
    rule_text = json.dumps(rule, ensure_ascii=False)
    return {'model': 'STGP_RMO_125_20', 'signal_up': signal,
            'direction': '上涨' if signal else '下跌或持平',
            'rule_id': hashlib.sha256(rule_text.encode()).hexdigest()[:12],
            'block': block, 'block_start': str(block_start.date()),
            'train_start': str(train.index[0].date()), 'train_end': str(train.index[-1].date()),
            'seed': config.seed + block}


def forecast(now: datetime | None = None, config: Config = Config()) -> dict:
    if (config.train_days, config.predict_days) != (125, 20):
        raise ValueError('日常预测使用已登记的125/20窗口')
    frozen = _frozen_primary()
    index, index_meta = load_snapshot(DATA, 'index')
    etf, etf_meta = load_snapshot(DATA, 'etf')
    index, etf = align_assets(index, etf)
    frame = make_frame(index)
    latest = frame.index[-1]
    current = now or datetime.now(ZoneInfo('Asia/Shanghai'))
    fresh = is_pre_open_forecast(latest, current)
    OUTPUT.mkdir(exist_ok=True)
    result = {'status': 'ready' if fresh else 'stale_data',
              'generated_at': current.isoformat(), 'data_as_of': str(latest.date()),
              'target': '沪深300下一交易日开盘至收盘方向',
              'primary_model': frozen['winner'], 'primary_version': frozen['version'],
              'index_sha256': index_meta['sha256'], 'etf_sha256': etf_meta['sha256'],
              'primary': None, 'experimental': None,
              'note': '旧行情不会发布为当前预测。STGP仅为实验信号，不提供校准概率。'}
    signals = []
    if fresh:
        probability = predict_at(frame, latest, frozen['winner'])
        primary_up = probability >= 0.5
        primary = {'model': frozen['winner'], 'up_probability': probability,
                   'signal_up': primary_up, 'direction': '上涨' if primary_up else '下跌或持平'}
        experimental = _experimental_signal(frame, latest, config)
        result['primary'], result['experimental'] = primary, experimental
        for item in (primary, experimental):
            signal = {'signal_date': str(latest.date()), 'model': item['model'],
                      'signal_up': item['signal_up'], 'index_sha256': index_meta['sha256']}
            if 'up_probability' in item:
                signal['up_probability'] = item['up_probability']
                signal['version'] = frozen['version']
            else:
                signal['rule_id'] = item['rule_id']
                signal['seed'] = item['seed']
            signals.append(signal)
    record_and_reconcile(OUTPUT, index, signals, now=current)
    (OUTPUT / 'latest_forecast.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
