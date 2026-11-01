"""Daily-data transfer of the hybrid trend architecture; not an IF paper replication."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss

from .backtest import compare
from .data import align_assets, load_dividends, load_snapshot
from .model import make_research_frame
from .pipeline import DATA, OUTPUT, EVALUATION, VALIDATION
from .trend_paper_data import TrendConfig, segment_prices
from .trend_paper_model import classification_metrics, fit_head, require_torch

DAILY_OUTPUT = OUTPUT / 'trend_daily'
TRAIN_END = pd.Timestamp('2025-12-31')
KINDS = ('hybrid', 'lstm', 'cnn')


@dataclass(frozen=True)
class DailyTrendConfig:
    error_pct: float = 0.01
    min_segment_days: int = 4
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 0.005
    hidden_size: int = 100
    seed: int = 20261031
    shuffle_training: bool = False

    def network_config(self) -> TrendConfig:
        return TrendConfig(error_pct=self.error_pct, min_segment_bars=self.min_segment_days,
                           epochs=self.epochs, batch_size=self.batch_size,
                           learning_rate=self.learning_rate, hidden_size=self.hidden_size,
                           seed=self.seed, shuffle_training=self.shuffle_training)


def daily_input(prices: np.ndarray, config: DailyTrendConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Feature vector at the current close; argument must be an observed prefix."""
    observed = np.asarray(prices[-500:], dtype=float)
    segments = segment_prices(observed, config.error_pct, config.min_segment_days)
    if len(segments) < 4:
        return None
    recent = segments[-4:]
    trend = np.asarray([[s.slope, s.duration_hours * 12] for s in recent], dtype=np.float32)
    raw = []
    for segment in recent:
        window = observed[max(0, segment.end - 4):segment.end + 1]
        raw.append(np.pad(window, (5 - len(window), 0), mode='edge'))
    raw = np.asarray(raw, dtype=np.float32)
    return trend, raw - raw[0, 0]


def _future_duration(prices: np.ndarray, config: DailyTrendConfig) -> tuple[np.ndarray, np.ndarray]:
    segments = segment_prices(prices, config.error_pct, config.min_segment_days)
    starts = np.asarray([s.start for s in segments], dtype=int)
    ends = np.asarray([s.end for s in segments], dtype=int)
    duration = np.full(len(prices), np.nan)
    target_end = np.full(len(prices), -1, dtype=int)
    for i in range(len(prices)):
        location = int(np.searchsorted(starts, i, side='right'))
        if location < len(segments):
            duration[i] = ends[location] - starts[location]
            target_end[i] = ends[location]
    return duration, target_end


def prepare_daily_samples(index: pd.DataFrame, config: DailyTrendConfig) -> tuple[dict, pd.DataFrame]:
    """Training labels never cross 2025-12-31; 2026 labels are evaluation-only."""
    frame = make_research_frame(index)
    prices = index.close.to_numpy(float)
    train_last = int(np.searchsorted(index.index.to_numpy(), np.datetime64(TRAIN_END), side='right')) - 1
    train_duration, train_target_end = _future_duration(prices[:train_last + 1], config)
    eval_duration, eval_target_end = _future_duration(prices, config)
    training = {'trend': [], 'raw': [], 'direction': [], 'duration': [], 'signal_date': []}
    evaluation = []
    for position in range(100, len(index) - 1):
        date = index.index[position]
        if date not in frame.index or pd.isna(frame.at[date, 'target_date']):
            continue
        value = daily_input(prices[:position + 1], config)
        if value is None:
            continue
        next_date = frame.at[date, 'target_date']
        direction = int(frame.at[date, 'target_up'])
        if next_date <= TRAIN_END:
            if position > train_last or train_target_end[position] < 0:
                continue
            training['trend'].append(value[0])
            training['raw'].append(value[1])
            training['direction'].append(direction)
            training['duration'].append(train_duration[position])
            training['signal_date'].append(date)
        elif date >= pd.Timestamp(VALIDATION[0]) and next_date <= pd.Timestamp(EVALUATION[1]):
            evaluation.append({'position': position, 'signal_date': date, 'target_date': next_date,
                               'trend': value[0], 'raw': value[1], 'actual_up': direction,
                               'actual_return': float(frame.at[date, 'target_return']),
                               'actual_duration_days': eval_duration[position]
                               if eval_target_end[position] > position else np.nan})
    if len(training['direction']) < 100 or len(evaluation) < 100:
        raise ValueError('日线趋势样本不足以完成训练及2026年分期评价')
    samples = {key: np.asarray(values, dtype=np.float32)
               for key, values in training.items() if key != 'signal_date'}
    samples['signal_date'] = training['signal_date']
    return samples, pd.DataFrame(evaluation)


def list_daily_runs() -> list[dict]:
    if not DAILY_OUTPUT.exists():
        return []
    result = []
    for path in sorted(DAILY_OUTPUT.glob('*/summary.json'), reverse=True):
        try:
            result.append(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            continue
    return result


def _external_torch_python() -> Path | None:
    candidate = Path(os.environ.get('TREND_TORCH_PYTHON', str(Path.home() / 'venvs/pytorch/bin/python')))
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def training_environment() -> dict:
    try:
        torch = require_torch()
        return {'ready': True, 'python': 'platform_environment', 'torch_version': torch.__version__}
    except RuntimeError:
        python = _external_torch_python()
        if python is None:
            return {'ready': False, 'reason': '没有找到PyTorch训练环境'}
        try:
            checked = subprocess.run([str(python), '-c', 'import torch; print(torch.__version__)'],
                                     capture_output=True, text=True, timeout=20, check=True)
            return {'ready': True, 'python': str(python), 'torch_version': checked.stdout.strip()}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return {'ready': False, 'reason': f'{python}无法加载PyTorch'}


def _external_predictions(training: dict, x_trend: np.ndarray, x_raw: np.ndarray,
                          config: DailyTrendConfig, stage: Path) -> tuple[dict, dict]:
    python = _external_torch_python()
    if python is None:
        raise RuntimeError('当前环境缺少PyTorch，且未找到独立训练环境；可设置TREND_TORCH_PYTHON')
    with tempfile.TemporaryDirectory(prefix='.torch-input-', dir=stage) as folder:
        folder = Path(folder)
        input_path, output_path, config_path = folder / 'input.npz', folder / 'output.npz', folder / 'config.json'
        np.savez_compressed(input_path, trend=training['trend'], raw=training['raw'],
                            direction=training['direction'], duration=training['duration'],
                            eval_trend=x_trend, eval_raw=x_raw)
        config_path.write_text(json.dumps(asdict(config.network_config())), encoding='utf-8')
        process = subprocess.run([str(python), '-m', 'system.trend_torch_worker', str(input_path),
                                  str(output_path), str(config_path)],
                                 cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                 text=True, timeout=3600, check=False)
        if process.returncode != 0:
            raise RuntimeError(f'独立PyTorch训练失败：{process.stderr[-1200:]}')
        metadata = json.loads(process.stdout.strip().splitlines()[-1])
        with np.load(output_path, allow_pickle=False) as data:
            predictions = {name: data[name].copy() for name in data.files}
    return predictions, metadata


def run_daily_transfer(config: DailyTrendConfig = DailyTrendConfig()) -> dict:
    if not 0.0001 <= config.error_pct <= 0.03 or not 3 <= config.min_segment_days <= 30:
        raise ValueError('日线趋势分割参数不在允许范围内')
    # The model validates the remaining network settings before creating output.
    from .trend_paper_data import validate_config
    validate_config(config.network_config())
    index, index_meta = load_snapshot(DATA, 'index')
    etf, etf_meta = load_snapshot(DATA, 'etf')
    index, etf = align_assets(index, etf)
    dividends = load_dividends(DATA / '510300_dividends.csv', etf.index)
    training, evaluation = prepare_daily_samples(index, config)
    try:
        require_torch()
        local_torch = True
    except RuntimeError:
        if _external_torch_python() is None:
            raise
        local_torch = False
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    result = {'id': run_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'completed', 'study_type': 'daily_trend_method_transfer',
              'target': 'next_session_index_open_to_close_direction',
              'auxiliary_target': 'next_price_trend_duration_trading_days',
              'config': asdict(config),
              'source': {'index': index_meta['source'], 'etf': etf_meta['source']},
              'source_sha256': {'index': index_meta['sha256'], 'etf': etf_meta['sha256']},
              'training': {'last_label_date': str(TRAIN_END.date()), 'samples': len(training['direction']),
                           'internal_validation': 'chronological_last_20_percent'},
              'periods': {'validation': VALIDATION, 'retrospective': EVALUATION},
              'results': {},
              'warning': '这是沪深300指数日线方法迁移，不是IF五分钟期货论文原口径复现。'
                         '2026年后段已被查看，不能作为新的前瞻盲测。'
                         '趋势时长只作辅助预测，ETF账本仅使用次日方向。'}
    x_trend = np.stack(evaluation.trend.to_numpy()).astype(np.float32)
    x_raw = np.stack(evaluation.raw.to_numpy()).astype(np.float32)
    DAILY_OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.daily-trend-', dir=DAILY_OUTPUT) as folder:
        stage = Path(folder)
        external_outputs = None
        if not local_torch:
            external_outputs, environment = _external_predictions(training, x_trend, x_raw, config, stage)
            result['training']['environment'] = environment
        else:
            result['training']['environment'] = {'python': 'platform_environment', 'backend': 'PyTorch'}
        for kind in KINDS:
            if external_outputs is None:
                direction_head = fit_head(training, kind, 'direction', config.network_config())
                duration_head = fit_head(training, kind, 'duration', config.network_config())
                probabilities = direction_head.predict(x_trend, x_raw)
                durations = duration_head.predict(x_trend, x_raw)
                threshold = direction_head.threshold
            else:
                probabilities = external_outputs[f'{kind}_probability']
                durations = external_outputs[f'{kind}_duration']
                threshold = float(external_outputs[f'{kind}_threshold'])
            predictions = evaluation[['signal_date', 'target_date', 'actual_up', 'actual_return',
                                      'actual_duration_days']].copy()
            predictions['model'] = kind
            predictions['up_probability'] = probabilities
            predictions['threshold'] = threshold
            predictions['signal_up'] = probabilities >= threshold
            predictions['predicted_duration_days'] = durations
            predictions.to_csv(stage / f'{kind}_predictions.csv', index=False)
            result['results'][kind] = {}
            for period, bounds in (('validation', VALIDATION), ('retrospective', EVALUATION)):
                part = predictions[predictions.target_date.between(*bounds)].copy()
                if part.empty:
                    raise ValueError(f'{period}缺少评价样本')
                y = part.actual_up.to_numpy(int)
                p = part.up_probability.to_numpy(float)
                direction = classification_metrics(y, p, threshold)
                direction['balanced_accuracy'] = float(balanced_accuracy_score(y, part.signal_up.to_numpy(int)))
                direction['brier'] = float(brier_score_loss(y, p))
                direction['signal_up_rate'] = float(part.signal_up.mean())
                direction['actual_up_rate'] = float(part.actual_up.mean())
                valid = part.actual_duration_days.notna()
                duration_rmse = float(np.sqrt(np.mean((part.loc[valid, 'predicted_duration_days'] -
                                                      part.loc[valid, 'actual_duration_days']) ** 2))) if valid.any() else None
                trading_input = part.copy()
                trading_input['up_probability'] = trading_input.signal_up.astype(float)
                ledger, trading = compare(trading_input, etf, dividends)
                ledger.to_csv(stage / f'{kind}_{period}_etf_ledger.csv', index=False)
                result['results'][kind][period] = {'direction': direction, 'duration_rmse_days': duration_rmse,
                                                   'duration_n': int(valid.sum()), 'trading': trading}
        (stage / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(stage, DAILY_OUTPUT / run_id)
    return result
