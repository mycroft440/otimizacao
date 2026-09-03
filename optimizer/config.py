from __future__ import annotations

import math
from datetime import date

DEFAULT_START = "2018-01-02"
DEFAULT_END = ""
DEFAULT_GAP_MIN = 5
DEFAULT_GAP_MAX = 80
DEFAULT_SIGNAL_MIN = 2
DEFAULT_SIGNAL_MAX = 60
DEFAULT_MOMENTUM_MIN = 5
DEFAULT_MOMENTUM_MAX = 252
DEFAULT_VOL_PERIOD = 21
DEFAULT_VOL_MIN = 2
DEFAULT_VOL_MAX = 60
DEFAULT_INITIAL_CASH = 1000.0
DEFAULT_FEE_BPS = 3.25
DEFAULT_SLIPPAGE_BPS = 10.0
DEFAULT_ODD_LOT_EXTRA_BPS = 5.0
DEFAULT_SHARDS = 20

# Guard rails. They are intentionally permissive enough for stress tests while
# still catching obvious workflow_dispatch typos such as 325 bps instead of 3.25.
MAX_FEE_BPS = 100.0
MAX_SLIPPAGE_BPS = 500.0
MAX_ODD_LOT_EXTRA_BPS = 500.0
MAX_PERIOD = 5000


def required_warmup_sessions(
    gap_max: int,
    signal_max: int,
    momentum_max: int,
    vol_period: int,
    *,
    safety_sessions: int = 10,
) -> int:
    """Conservative number of pre-start sessions required by the indicator family.

    Gap needs a previous close, the Gap Ratio needs ``gap_max`` observations,
    its SMA needs ``signal_max`` valid ratios and the persistent direction state
    needs one further comparison. Momentum needs ``momentum_max`` prior closes.
    Volatility uses ``vol_period`` returns. The safety margin protects against
    off-by-one differences at the snapshot boundary.
    """
    validate_periods(gap_max, signal_max, momentum_max, vol_period)
    gap_signal = gap_max + signal_max + 2
    momentum = momentum_max + 1
    volatility = vol_period + 2
    return max(gap_signal, momentum, volatility) + int(safety_sessions)


def validate_periods(gap: int, signal: int, momentum: int, vol: int) -> None:
    values = {"gap": gap, "signal": signal, "momentum": momentum, "vol": vol}
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} precisa ser inteiro")
        if value <= 0 or value > MAX_PERIOD:
            raise ValueError(f"{name} fora do intervalo permitido: {value}")
    if vol < 2:
        raise ValueError(
            "vol precisa ser >= 2 para reproduzir a variancia amostral do Pine"
        )


def validate_run_config(
    *,
    start: str,
    end: str,
    gap_min: int,
    gap_max: int,
    signal_min: int,
    signal_max: int,
    momentum_min: int,
    momentum_max: int,
    vol_period: int,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    odd_lot_extra_bps: float,
    shard_id: int | None = None,
    shards: int | None = None,
) -> None:
    validate_periods(gap_min, signal_min, momentum_min, vol_period)
    validate_periods(gap_max, signal_max, momentum_max, vol_period)
    if gap_max < gap_min or signal_max < signal_min or momentum_max < momentum_min:
        raise ValueError("faixa de periodos invertida")

    try:
        start_date = date.fromisoformat(start)
    except Exception as exc:  # pragma: no cover - exact parser error is irrelevant
        raise ValueError(f"start invalido: {start!r}") from exc
    if end:
        try:
            end_date = date.fromisoformat(end)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"end invalido: {end!r}") from exc
        if start_date >= end_date:
            raise ValueError(f"start precisa ser anterior a end: {start} >= {end}")

    finite_values = {
        "initial_cash": initial_cash,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "odd_lot_extra_bps": odd_lot_extra_bps,
    }
    for name, value in finite_values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} precisa ser finito")
    if initial_cash <= 0:
        raise ValueError("initial_cash precisa ser > 0")
    if fee_bps < 0 or fee_bps > MAX_FEE_BPS:
        raise ValueError(f"fee_bps fora do intervalo [0,{MAX_FEE_BPS}]")
    if slippage_bps < 0 or slippage_bps > MAX_SLIPPAGE_BPS:
        raise ValueError(f"slippage_bps fora do intervalo [0,{MAX_SLIPPAGE_BPS}]")
    if odd_lot_extra_bps < 0 or odd_lot_extra_bps > MAX_ODD_LOT_EXTRA_BPS:
        raise ValueError(
            f"odd_lot_extra_bps fora do intervalo [0,{MAX_ODD_LOT_EXTRA_BPS}]"
        )

    if shards is not None:
        if shards <= 0:
            raise ValueError("shards precisa ser > 0")
        if shard_id is None or not 0 <= shard_id < shards:
            raise ValueError(f"shard_id invalido: {shard_id}/{shards}")