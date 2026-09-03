from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_b3_pine as opt

CACHE_FILE = "FAST_MARKET_FEATURES.npz"
CACHE_META = "FAST_MARKET_FEATURES.json"
CACHE_SCHEMA = 1


@dataclass
class FastMarketData:
    tickers: list[str]
    execution_dates: pd.DatetimeIndex
    decision_index: np.ndarray
    exec_open: np.ndarray
    final_close: np.ndarray
    start: str
    end: str
    opens: np.ndarray
    closes: np.ndarray
    lengths: np.ndarray
    momentum_values: np.ndarray
    momentum: np.ndarray
    vol_valid: np.ndarray
    vol_period: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compute_shared_features(market, momentum_values: list[int], vol_period: int):
    M, W, N = len(momentum_values), len(market.execution_dates), len(market.tickers)
    momentum = np.full((M, W, N), np.nan, dtype=np.float64)
    vol_valid = np.zeros((W, N), dtype=np.bool_)
    m_values = np.asarray(momentum_values, dtype=np.int64)[:, None]
    for ti, ticker in enumerate(market.tickers):
        closes = market.frames[ticker]["close"].to_numpy(dtype=np.float64)
        didx = market.decision_index[ti].astype(np.int64, copy=False)
        decision_ok = didx >= 0
        vol = opt.sample_vol_positive(closes, vol_period)
        if np.any(decision_ok):
            vol_valid[decision_ok, ti] = vol[didx[decision_ok]]
        if M and W and len(closes):
            prev_idx = didx[None, :] - m_values
            valid = decision_ok[None, :] & (prev_idx >= 0)
            safe_cur = np.clip(didx, 0, len(closes) - 1)
            safe_prev = np.clip(prev_idx, 0, len(closes) - 1)
            cur = closes[safe_cur][None, :]
            prev = closes[safe_prev]
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = cur / prev - 1.0
            momentum[:, :, ti] = np.where(
                valid & (prev > 0.0) & (cur > 0.0), scores, np.nan
            )
    return momentum, vol_valid


def build_fast_cache(data_root: Path, start: str, end: str, momentum_min: int, momentum_max: int, vol_period: int) -> Path:
    market = opt.load_market(data_root, start, end)
    values = list(range(momentum_min, momentum_max + 1))
    momentum, vol_valid = compute_shared_features(market, values, vol_period)
    lengths = np.asarray([len(market.frames[t]) for t in market.tickers], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0
    opens = np.full((len(market.tickers), max_len), np.nan, dtype=np.float64)
    closes = np.full_like(opens, np.nan)
    for ti, ticker in enumerate(market.tickers):
        n = int(lengths[ti])
        opens[ti, :n] = market.frames[ticker]["open"].to_numpy(dtype=np.float64)
        closes[ti, :n] = market.frames[ticker]["close"].to_numpy(dtype=np.float64)
    path = data_root / CACHE_FILE
    np.savez(
        path,
        tickers=np.asarray(market.tickers, dtype="U16"),
        start=np.asarray([market.start], dtype="U16"),
        requested_end=np.asarray([end], dtype="U16"),
        effective_end=np.asarray([market.end], dtype="U16"),
        execution_dates=market.execution_dates.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        decision_index=market.decision_index,
        exec_open=market.exec_open,
        final_close=market.final_close,
        lengths=lengths,
        opens=opens,
        closes=closes,
        momentum_values=np.asarray(values, dtype=np.int32),
        momentum=momentum,
        vol_valid=vol_valid,
        vol_period=np.asarray([vol_period], dtype=np.int32),
    )
    meta = {
        "schema_version": CACHE_SCHEMA,
        "cache_sha256": sha256_file(path),
        "start": market.start,
        "requested_end": end,
        "effective_end": market.end,
        "momentum_min": momentum_min,
        "momentum_max": momentum_max,
        "vol_period": vol_period,
        "weeks": len(market.execution_dates),
    }
    (data_root / CACHE_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def load_fast_cache(data_root: Path, start: str, end: str, momentum_values: list[int], vol_period: int):
    path = data_root / CACHE_FILE
    meta_path = data_root / CACHE_META
    if not path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("schema_version", -1)) != CACHE_SCHEMA:
        raise RuntimeError("Fast cache schema incompatível.")
    if meta.get("cache_sha256") != sha256_file(path):
        raise RuntimeError("Fast cache SHA-256 divergente.")
    if str(meta.get("start")) != str(start) or str(meta.get("requested_end")) != str(end):
        raise RuntimeError("Fast cache pertence a outro período.")
    if int(meta.get("vol_period", -1)) != int(vol_period):
        raise RuntimeError("Fast cache usa VOL_PERIOD diferente.")

    with np.load(path, allow_pickle=False) as data:
        cached_m = data["momentum_values"].astype(np.int32, copy=False)
        requested = np.asarray(momentum_values, dtype=np.int32)
        positions = np.searchsorted(cached_m, requested)
        if len(requested) and (
            np.any(positions >= len(cached_m)) or np.any(cached_m[positions] != requested)
        ):
            raise RuntimeError("Fast cache não cobre todos os Momentum solicitados.")
        return FastMarketData(
            tickers=[str(x) for x in data["tickers"].tolist()],
            execution_dates=pd.DatetimeIndex(data["execution_dates"].astype("datetime64[ns]")),
            decision_index=data["decision_index"].astype(np.int32, copy=True),
            exec_open=data["exec_open"].astype(np.float64, copy=True),
            final_close=data["final_close"].astype(np.float64, copy=True),
            start=str(data["start"][0]),
            end=str(data["effective_end"][0]),
            opens=data["opens"].astype(np.float64, copy=True),
            closes=data["closes"].astype(np.float64, copy=True),
            lengths=data["lengths"].astype(np.int32, copy=True),
            momentum_values=requested.copy(),
            momentum=data["momentum"][positions].astype(np.float64, copy=True),
            vol_valid=data["vol_valid"].astype(np.bool_, copy=True),
            vol_period=int(data["vol_period"][0]),
        )
