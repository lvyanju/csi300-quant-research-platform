"""Append-only prospective signal records and later outcome reconciliation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def is_pre_open_forecast(signal_date: pd.Timestamp, now: datetime) -> bool:
    """Conservative freshness rule: latest close or next weekday before 09:30."""
    session = pd.Timestamp(signal_date).normalize()
    today = pd.Timestamp(now.date())
    if today == session:
        return (now.hour, now.minute) >= (15, 30)
    next_weekday = session + pd.offsets.BDay(1)
    if session < today < next_weekday:
        return True
    if today == next_weekday:
        return (now.hour, now.minute) < (9, 30)
    return False


def record_and_reconcile(output: Path, index: pd.DataFrame, signals: list[dict],
                         now: datetime | None = None) -> bool:
    """First record for each (signal date, model) is immutable on later reruns."""
    path = output / 'forward_signals.jsonl'
    old = []
    if path.exists():
        old = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    keys = {(item['signal_date'], item['model']) for item in old}
    current = now or datetime.now(ZoneInfo('Asia/Shanghai'))
    timestamp = current.isoformat()
    additions = []
    for signal in signals:
        key = signal['signal_date'], signal['model']
        if key not in keys and is_pre_open_forecast(pd.Timestamp(signal['signal_date']), current):
            additions.append({**signal, 'recorded_at': timestamp})
            keys.add(key)
    if additions:
        with path.open('a', encoding='utf-8') as handle:
            for item in additions:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        old.extend(additions)
    records = []
    dates = index.index
    for item in old:
        later = dates[dates > pd.Timestamp(item['signal_date'])]
        if len(later):
            date = later[0]
            ret = float(index.loc[date, 'close'] / index.loc[date, 'open'] - 1)
            records.append({'signal_date': item['signal_date'], 'target_date': str(date.date()),
                            'model': item['model'], 'predicted_up': item['signal_up'],
                            'actual_up': ret > 0, 'actual_return': ret,
                            'correct': bool(item['signal_up']) == (ret > 0)})
    cols = ['signal_date', 'target_date', 'model', 'predicted_up', 'actual_up', 'actual_return', 'correct']
    pd.DataFrame(records, columns=cols).to_csv(output / 'forward_outcomes.csv', index=False)
    return bool(additions)
