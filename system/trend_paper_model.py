"""CNN, LSTM and fused CNN+LSTM heads for trend direction and duration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .trend_paper_data import TrendConfig


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('论文混合网络需要额外安装PyTorch：python -m pip install torch') from exc
    return torch


def build_network(kind: str, hidden_size: int):
    torch = require_torch()
    nn = torch.nn
    if kind not in ('hybrid', 'lstm', 'cnn'):
        raise ValueError('模型须为hybrid、lstm或cnn')

    class TrendNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
            self.trend_map = nn.Linear(hidden_size, 32)
            self.conv = nn.Sequential(nn.Conv1d(4, 20, 1), nn.ReLU(),
                                      nn.Conv1d(20, 20, 1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            self.price_map = nn.Linear(20, 32)
            self.dropout = nn.Dropout(0.2)
            self.head = nn.Linear(32, 1)

        def forward(self, trend, raw):
            if self.kind in ('hybrid', 'lstm'):
                trend_hidden = self.trend_map(self.lstm(trend)[0][:, -1, :])
            if self.kind in ('hybrid', 'cnn'):
                raw_hidden = self.price_map(self.conv(raw).squeeze(-1))
            if self.kind == 'hybrid':
                combined = trend_hidden + raw_hidden
            else:
                combined = trend_hidden if self.kind == 'lstm' else raw_hidden
            return self.head(self.dropout(torch.relu(combined))).squeeze(-1)

    return TrendNet()


def fbeta_precision_weighted(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.4) -> float:
    truth = np.asarray(y_true, dtype=bool)
    prediction = np.asarray(y_pred, dtype=bool)
    true_positive = int(np.sum(truth & prediction))
    precision = true_positive / int(np.sum(prediction)) if prediction.any() else 0.0
    recall = true_positive / int(np.sum(truth)) if truth.any() else 0.0
    denominator = beta * beta * precision + recall
    return float((1 + beta * beta) * precision * recall / denominator) if denominator else 0.0


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Search the paper's 0..0.9 grid on training validation only."""
    candidates = np.round(np.arange(0, 0.901, 0.05), 2)
    return float(max(candidates, key=lambda t: (fbeta_precision_weighted(y_true, probability >= t), -abs(t - 0.5))))


@dataclass
class FittedHead:
    network: object
    kind: str
    task: str
    trend_mean: np.ndarray
    trend_std: np.ndarray
    raw_std: float
    threshold: float | None
    validation_loss: float

    def predict(self, trend: np.ndarray, raw: np.ndarray) -> np.ndarray:
        torch = require_torch()
        self.network.eval()
        x_trend = (np.asarray(trend, dtype=np.float32) - self.trend_mean) / self.trend_std
        x_raw = np.asarray(raw, dtype=np.float32) / self.raw_std
        with torch.no_grad():
            logits = self.network(torch.from_numpy(x_trend), torch.from_numpy(x_raw)).numpy()
        if self.task == 'direction':
            return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        return np.maximum(5 / 60, np.logaddexp(0, logits))


def fit_head(samples: dict[str, np.ndarray], kind: str, task: str,
             config: TrendConfig) -> FittedHead:
    torch = require_torch()
    if task not in ('direction', 'duration'):
        raise ValueError('任务须为direction或duration')
    n = len(samples['direction'])
    if n < 100:
        raise ValueError('至少需要100个完整趋势样本')
    cut = int(n * 0.8)
    torch.manual_seed(config.seed + (0 if task == 'direction' else 1))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    trend_mean = samples['trend'][:cut].mean(axis=(0, 1), keepdims=True)
    trend_std = np.maximum(samples['trend'][:cut].std(axis=(0, 1), keepdims=True), 1e-6)
    raw_std = max(float(samples['raw'][:cut].std()), 1e-6)
    x_trend = torch.from_numpy(((samples['trend'] - trend_mean) / trend_std).astype(np.float32))
    x_raw = torch.from_numpy((samples['raw'] / raw_std).astype(np.float32))
    target = torch.from_numpy(samples[task].astype(np.float32))
    net = build_network(kind, config.hidden_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(config.seed)
    best_state, best_loss = None, float('inf')
    for _ in range(config.epochs):
        net.train()
        order = torch.randperm(cut, generator=generator) if config.shuffle_training else torch.arange(cut)
        for ids in order.split(config.batch_size):
            output = net(x_trend[ids], x_raw[ids])
            if task == 'direction':
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target[ids])
            else:
                predicted = torch.nn.functional.softplus(output)
                loss = torch.mean((predicted - target[ids]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            output = net(x_trend[cut:], x_raw[cut:])
            if task == 'direction':
                score = torch.nn.functional.binary_cross_entropy_with_logits(output, target[cut:])
            else:
                score = torch.sqrt(torch.mean((torch.nn.functional.softplus(output) - target[cut:]) ** 2))
            value = float(score)
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().clone() for key, tensor in net.state_dict().items()}
    net.load_state_dict(best_state)
    fitted = FittedHead(net, kind, task, trend_mean.astype(np.float32), trend_std.astype(np.float32),
                        raw_std, None, best_loss)
    if task == 'direction':
        fitted.threshold = choose_threshold(samples['direction'][cut:].astype(int),
                                            fitted.predict(samples['trend'][cut:], samples['raw'][cut:]))
    return fitted


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    predicted = probability >= threshold
    return {'n': len(y_true), 'threshold': threshold,
            'accuracy': float(accuracy_score(y_true, predicted)),
            'precision': float(precision_score(y_true, predicted, zero_division=0)),
            'recall': float(recall_score(y_true, predicted, zero_division=0)),
            'f_0_4': fbeta_precision_weighted(y_true, predicted),
            'log_loss': float(-np.mean(y_true * np.log(np.clip(probability, 1e-7, 1)) +
                                        (1 - y_true) * np.log(np.clip(1 - probability, 1e-7, 1))))}
