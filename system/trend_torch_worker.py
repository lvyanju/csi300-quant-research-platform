"""PyTorch-only worker for training in a separate Python environment."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .trend_paper_model import fit_head, require_torch


def main(input_path: Path, output_path: Path, config_path: Path) -> None:
    torch = require_torch()
    with np.load(input_path, allow_pickle=False) as data:
        training = {name: data[name] for name in ('trend', 'raw', 'direction', 'duration')}
        eval_trend = data['eval_trend']
        eval_raw = data['eval_raw']
    config = SimpleNamespace(**json.loads(config_path.read_text(encoding='utf-8')))
    outputs: dict[str, np.ndarray] = {}
    for kind in ('hybrid', 'lstm', 'cnn'):
        direction = fit_head(training, kind, 'direction', config)
        duration = fit_head(training, kind, 'duration', config)
        outputs[f'{kind}_probability'] = direction.predict(eval_trend, eval_raw)
        outputs[f'{kind}_duration'] = duration.predict(eval_trend, eval_raw)
        outputs[f'{kind}_threshold'] = np.asarray(direction.threshold, dtype=float)
    np.savez_compressed(output_path, **outputs)
    print(json.dumps({'torch_version': torch.__version__, 'python': sys.executable}))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('usage: python -m system.trend_torch_worker input.npz output.npz config.json')
    main(*(Path(value) for value in sys.argv[1:]))
