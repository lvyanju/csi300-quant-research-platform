"""Validated five-minute IF continuous-contract input and trend samples."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'if_5m'
OUTPUT_DIR = ROOT / 'outputs' / 'trend_paper'


@dataclass(frozen=True)
class TrendConfig:
    error_pct: float = 0.002
    min_segment_bars: int = 4
    train_years: int = 3
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 0.005
    hidden_size: int = 100
    seed: int = 20261031
    evaluation_step_bars: int = 12
    max_hold_bars: int = 72
    stop_loss: float = 0.006
    shuffle_training: bool = False


def validate_config(config: TrendConfig) -> None:
    if not 0.0001 <= config.error_pct <= 0.03 or not 3 <= config.min_segment_bars <= 30:
        raise ValueError('分段误差或最短趋势长度不在允许范围内')
    if config.train_years != 3:
        raise ValueError('论文逐年实验使用前三年训练')
    if not 1 <= config.epochs <= 100 or not 16 <= config.batch_size <= 1024:
        raise ValueError('轮次或批量不在允许范围内')
    if not 1e-5 <= config.learning_rate <= 0.05 or not 16 <= config.hidden_size <= 256:
        raise ValueError('学习率或隐藏单元数不在允许范围内')
    if not 1 <= config.evaluation_step_bars <= 72 or not 1 <= config.max_hold_bars <= 144:
        raise ValueError('预测间隔或最大持仓长度不在允许范围内')
    if not 0 < config.stop_loss <= 0.05 or not 0 <= config.seed <= 2_000_000_000:
        raise ValueError('止损或随机种子不在允许范围内')


def _read_bytes(content: bytes, suffix: str) -> pd.DataFrame:
    if suffix.lower() == '.csv':
        return pd.read_csv(io.BytesIO(content))
    if suffix.lower() == '.parquet':
        return pd.read_parquet(io.BytesIO(content))
    raise ValueError('仅支持CSV或Parquet')


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'timestamp', 'close', 'contract'}
    if not required.issubset(frame):
        raise ValueError(f'缺少字段：{sorted(required - set(frame))}')
    bars = frame.loc[:, ['timestamp', 'close', 'contract']].copy()
    bars['timestamp'] = pd.to_datetime(bars.timestamp, errors='raise')
    if bars.timestamp.dt.tz is not None:
        raise ValueError('时间戳须为Asia/Shanghai本地时间，文件中不带时区')
    bars['close'] = pd.to_numeric(bars.close, errors='raise')
    bars['contract'] = bars.contract.astype(str).str.upper().str.strip()
    if bars.empty or bars.timestamp.isna().any() or bars.close.isna().any():
        raise ValueError('存在空行情或无效值')
    if not bars.timestamp.is_monotonic_increasing or bars.timestamp.duplicated().any():
        raise ValueError('时间戳须严格升序且无重复')
    if (bars.close <= 0).any() or not np.isfinite(bars.close.to_numpy()).all():
        raise ValueError('收盘价须为正的有限数')
    if not bars.contract.str.match(r'^IF[A-Z0-9_]*$').all():
        raise ValueError('合约标识须为IF系列')
    stamp = bars.timestamp.dt
    minutes = stamp.hour * 60 + stamp.minute
    if (stamp.dayofweek >= 5).any() or (stamp.minute % 5 != 0).any() or (stamp.second != 0).any():
        raise ValueError('仅允许工作日五分钟整点K线')
    # Include both historical 09:15/15:15 futures sessions and later 09:30/15:00 sessions.
    session = ((minutes >= 9 * 60 + 15) & (minutes <= 11 * 60 + 30)) | (
        (minutes >= 13 * 60) & (minutes <= 15 * 60 + 15))
    if not session.all():
        raise ValueError('存在期货日盘交易时段外记录')
    daily_count = bars.groupby(stamp.date).size()
    if (daily_count < 20).any() or (daily_count > 54).any():
        raise ValueError('至少一个交易日五分钟K线数量异常')
    # A very large jump usually signals an unadjusted contract roll or bad price.
    jumps = bars.close.pct_change().abs()
    if (jumps > 0.15).any():
        raise ValueError('相邻价格跳变超过15%，请核对连续合约调整与数据质量')
    return bars.reset_index(drop=True)


def coverage(bars: pd.DataFrame) -> dict:
    years = bars.timestamp.dt.year.value_counts().to_dict()
    days = bars.groupby(bars.timestamp.dt.year).timestamp.apply(lambda s: s.dt.date.nunique()).to_dict()
    expected = list(range(2013, 2019))
    return {'years': {str(y): int(years.get(y, 0)) for y in expected},
            'trading_days': {str(y): int(days.get(y, 0)) for y in expected},
            'complete_for_paper': all(years.get(y, 0) >= 8000 and days.get(y, 0) >= 200 for y in expected),
            'first': str(bars.timestamp.iloc[0]), 'last': str(bars.timestamp.iloc[-1]),
            'rows': len(bars), 'contract_switches': int(bars.contract.ne(bars.contract.shift()).sum() - 1)}


def import_bars(content: bytes, filename: str, source: str, adjustment: str) -> dict:
    if not source.strip() or not adjustment.strip():
        raise ValueError('须记录数据来源和连续合约调整说明')
    if len(content) > 250_000_000:
        raise ValueError('文件超过250MB，请改用本地导入命令')
    bars = validate_bars(_read_bytes(content, Path(filename).suffix))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {'source': source.strip(), 'adjustment': adjustment.strip(), 'source_filename': Path(filename).name,
                'source_sha256': hashlib.sha256(content).hexdigest(), 'imported_at_utc': datetime.now(timezone.utc).isoformat(),
                'coverage': coverage(bars), 'schema': ['timestamp', 'close', 'contract'],
                'price_basis': 'supplier_adjusted_if_continuous_five_minute_close'}
    with tempfile.TemporaryDirectory(prefix='.if-stage-', dir=DATA_DIR) as folder:
        stage = Path(folder)
        bars.to_csv(stage / 'bars.csv', index=False)
        metadata['normalized_sha256'] = hashlib.sha256((stage / 'bars.csv').read_bytes()).hexdigest()
        (stage / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        # Publish the pair only after both files have been written and checked.
        version = hashlib.sha256((metadata['source_sha256'] + source.strip() + adjustment.strip()).encode()).hexdigest()[:16]
        target = DATA_DIR / version
        if not target.exists():
            os.replace(stage, target)
        else:
            metadata = json.loads((target / 'metadata.json').read_text(encoding='utf-8'))
    return {'id': version, **metadata}


def list_datasets() -> list[dict]:
    if not DATA_DIR.exists():
        return []
    result = []
    for meta_path in sorted(DATA_DIR.glob('*/metadata.json'), reverse=True):
        try:
            item = json.loads(meta_path.read_text(encoding='utf-8'))
            item['id'] = meta_path.parent.name
            result.append(item)
        except (OSError, ValueError):
            continue
    return result


def load_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict]:
    if not dataset_id or '/' in dataset_id or '\\' in dataset_id or dataset_id.startswith('.'):
        raise ValueError('数据集ID无效')
    root = DATA_DIR / dataset_id
    raw = (root / 'bars.csv').read_bytes()
    meta = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    if hashlib.sha256(raw).hexdigest() != meta['normalized_sha256']:
        raise ValueError('导入后的五分钟数据校验值不一致')
    bars = validate_bars(pd.read_csv(io.BytesIO(raw)))
    if coverage(bars) != meta['coverage']:
        raise ValueError('数据覆盖元信息与行情不一致')
    return bars, meta


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    slope: float
    duration_hours: float


def segment_prices(prices: np.ndarray, error_pct: float = 0.002,
                   min_bars: int = 4) -> list[Segment]:
    """Keogh-style top-down linear segmentation of a *known historical prefix*."""
    values = np.asarray(prices, dtype=float)
    if len(values) < min_bars or not np.isfinite(values).all() or (values <= 0).any():
        return []
    bounds = []
    stack = [(0, len(values) - 1)]
    while stack:
        lo, hi = stack.pop()
        x = np.arange(hi - lo + 1)
        fitted = values[lo] + (values[hi] - values[lo]) * x / max(1, hi - lo)
        residual = np.abs(values[lo:hi + 1] - fitted) / max(values[lo], 1e-8)
        split = int(np.argmax(residual)) + lo
        if residual.max() > error_pct and split - lo + 1 >= min_bars and hi - split + 1 >= min_bars:
            stack.extend([(split, hi), (lo, split)])
        else:
            bounds.append((lo, hi))
    bounds.sort()
    return [Segment(lo, hi, float((values[hi] - values[lo]) / (hi - lo) / values[lo]),
                    float((hi - lo) * 5 / 60)) for lo, hi in bounds if hi > lo]


def make_samples(prices: np.ndarray, config: TrendConfig) -> dict[str, np.ndarray]:
    segments = segment_prices(prices, config.error_pct, config.min_segment_bars)
    trend_inputs, raw_inputs, direction, duration, origin = [], [], [], [], []
    for i in range(4, len(segments)):
        previous = segments[i - 4:i]
        target = segments[i]
        trend_inputs.append([[s.slope, s.duration_hours] for s in previous])
        windows = []
        for s in previous:
            window = prices[max(0, s.end - 4):s.end + 1]
            windows.append(np.pad(window, (5 - len(window), 0), mode='edge'))
        raw = np.asarray(windows, dtype=float)
        raw_inputs.append(raw - raw[0, 0])
        direction.append(int(target.slope > 0))
        duration.append(target.duration_hours)
        origin.append(previous[-1].end)
    return {'trend': np.asarray(trend_inputs, dtype=np.float32).reshape(-1, 4, 2),
            'raw': np.asarray(raw_inputs, dtype=np.float32).reshape(-1, 4, 5),
            'direction': np.asarray(direction, dtype=np.float32),
            'duration': np.asarray(duration, dtype=np.float32),
            'origin': np.asarray(origin, dtype=int)}
