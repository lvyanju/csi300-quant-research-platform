from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData"
ASSETS = {"index": ("sh000300", "沪深300指数"), "etf": ("sh510300", "华泰柏瑞沪深300ETF")}
FIELDS = ["date", "open", "close", "high", "low", "volume", "amount", "amplitude", "change_pct", "change", "turnover"]


def fetch_asset(asset: str, start: str, end: str, directory: Path) -> Path:
    secid, name = ASSETS[asset]
    params = {"symbol": secid, "scale": "240", "ma": "no", "datalen": "1200"}
    try:
        response = requests.get(URL, params=params, timeout=25,
                                headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        body = response.text
    except requests.RequestException:
        from subprocess import run
        from urllib.parse import urlencode
        completed = run(["curl", "-L", "-sS", "--max-time", "25", f"{URL}?{urlencode(params)}"],
                        capture_output=True, text=True, check=True)
        body = completed.stdout
    match = re.search(r"var\s+_data=\((\[.*\])\)\s*;?", body, flags=re.S)
    if not match:
        raise ValueError(f"{name}接口未返回有效日线")
    raw = json.loads(match.group(1))
    if not raw:
        raise ValueError(f"{name}接口返回空日线")
    incoming = pd.DataFrame(raw).rename(columns={"day": "date"})
    incoming["date"] = pd.to_datetime(incoming.date)
    incoming = incoming[incoming.date.between(start, end)]
    if incoming.empty:
        raise ValueError(f"{name}请求区间无行情")
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{asset}.csv"
    # Only replace a usable snapshot after the entire response passes validation.
    frame = validate_asset(incoming, asset)
    temp = csv_path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    meta = {"asset": asset, "name": name, "code": secid, "source": URL,
            "price_basis": "unadjusted_exchange_price", "start": str(frame.date.min().date()),
            "end": str(frame.date.max().date()), "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": digest, "rows": len(frame)}
    csv_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path


def validate_asset(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{asset}缺少列：{sorted(required - set(frame.columns))}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result.date, errors="raise")
    if result.date.duplicated().any() or not result.date.is_monotonic_increasing:
        raise ValueError(f"{asset}日期重复或未升序")
    if result.date.dt.dayofweek.ge(5).any():
        raise ValueError(f"{asset}存在周末交易日")
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError(f"{asset}必要行情有缺失或非数值")
    if (result[["open", "high", "low", "close"]] <= 0).any().any() or (result.volume < 0).any():
        raise ValueError(f"{asset}行情价格或成交量非法")
    if ((result.high < result[["open", "close", "low"]].max(axis=1)) |
        (result.low > result[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{asset}OHLC边界不一致")
    return result


def load_snapshot(directory: Path, asset: str) -> tuple[pd.DataFrame, dict]:
    path = directory / f"{asset}.csv"
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"{asset}缺少行情快照；请先运行更新数据")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError(f"{asset}文件校验值不一致")
    if (meta.get("code") != ASSETS[asset][0] or meta.get("source") != URL or
        meta.get("price_basis") != "unadjusted_exchange_price"):
        raise ValueError(f"{asset}代码、数据源或价格口径不符")
    frame = validate_asset(pd.read_csv(path), asset).set_index("date")
    if (len(frame) != meta.get("rows") or str(frame.index.min().date()) != meta.get("start") or
        str(frame.index.max().date()) != meta.get("end")):
        raise ValueError(f"{asset}元数据与行情区间不一致")
    return frame, meta


def align_assets(index: pd.DataFrame, etf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = max(index.index.min(), etf.index.min())
    end = min(index.index.max(), etf.index.max())
    left, right = index.loc[start:end], etf.loc[start:end]
    if left.empty or not left.index.equals(right.index):
        raise ValueError("沪深300与510300交易日不一致或存在缺口")
    return left, right


def load_dividends(path: Path, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("缺少ETF分红文件，不能生成完整回测")
    frame = pd.read_csv(path)
    if set(frame.columns) != {"record_date", "ex_date", "pay_date", "cash_per_share"}:
        raise ValueError("分红文件必须含 record_date,ex_date,pay_date,cash_per_share")
    for column in ("record_date", "ex_date", "pay_date"):
        frame[column] = pd.to_datetime(frame[column])
    frame["cash_per_share"] = pd.to_numeric(frame.cash_per_share)
    if frame.ex_date.duplicated().any() or (frame.cash_per_share <= 0).any() or (
        (frame.record_date >= frame.ex_date) | (frame.ex_date > frame.pay_date)).any():
        raise ValueError("分红日期重复或金额无效")
    if not frame.empty and not frame.ex_date.isin(dates).all():
        raise ValueError("除息日不在行情交易日内")
    if not frame.empty and not frame.record_date.isin(dates).all():
        raise ValueError("权益登记日不在行情交易日内")
    return frame
