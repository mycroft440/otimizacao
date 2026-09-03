from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _prepare_curve(curve: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "equity"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"curve sem colunas obrigatorias: {sorted(missing)}")
    out = curve[["date", "equity"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["equity"] = pd.to_numeric(out["equity"], errors="raise")
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError("curve vazia")
    values = out["equity"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("equity precisa ser finita e nao negativa")
    return out


def year_is_complete(dates: pd.Series) -> bool:
    """Conservative completeness check without pretending to own a B3 calendar.

    A year is considered complete only when observations reach the first week of
    January and the final days of December. This deliberately rejects partial
    first/last years such as a backtest starting in July.
    """
    if dates.empty:
        return False
    first = pd.Timestamp(dates.iloc[0])
    last = pd.Timestamp(dates.iloc[-1])
    if first.year != last.year:
        raise ValueError("year_is_complete recebeu mais de um ano")
    starts_in_first_week = first.month == 1 and first.day <= 7
    ends_in_last_days = last.month == 12 and last.day >= 20
    return bool(starts_in_first_week and ends_in_last_days)


def annual_metrics(curve: pd.DataFrame, initial_cash: float) -> dict[str, object]:
    if not math.isfinite(float(initial_cash)) or initial_cash <= 0:
        raise ValueError("initial_cash precisa ser finito e > 0")
    df = _prepare_curve(curve)
    rows: list[dict[str, object]] = []
    prior_equity = float(initial_cash)
    for year in sorted(df["date"].dt.year.unique()):
        yd = df[df["date"].dt.year == year]
        start_equity = prior_equity
        end_equity = float(yd.iloc[-1]["equity"])
        ret = end_equity / start_equity - 1.0 if start_equity > 0 else float("nan")
        complete = year_is_complete(yd["date"])
        rows.append(
            {
                "year": int(year),
                "start_equity": start_equity,
                "end_equity": end_equity,
                "return": ret,
                "return_pct": ret * 100.0,
                "first_observation": yd.iloc[0]["date"].date().isoformat(),
                "last_observation": yd.iloc[-1]["date"].date().isoformat(),
                "complete_year": complete,
            }
        )
        prior_equity = end_equity

    complete_returns = np.asarray(
        [float(row["return"]) for row in rows if bool(row["complete_year"])], dtype=float
    )
    avg_complete = float(np.mean(complete_returns)) if len(complete_returns) else float("nan")
    geometric_complete = (
        float(np.prod(1.0 + complete_returns) ** (1.0 / len(complete_returns)) - 1.0)
        if len(complete_returns) and np.all(1.0 + complete_returns > 0)
        else float("nan")
    )
    final_equity = float(df.iloc[-1]["equity"])
    total_return = final_equity / initial_cash - 1.0
    return {
        "initial_cash": float(initial_cash),
        "final_equity": final_equity,
        "total_return": total_return,
        "total_return_pct": total_return * 100.0,
        "complete_years": int(sum(bool(row["complete_year"]) for row in rows)),
        "average_complete_year_return": avg_complete,
        "average_complete_year_return_pct": avg_complete * 100.0,
        "geometric_mean_complete_year_return": geometric_complete,
        "geometric_mean_complete_year_return_pct": geometric_complete * 100.0,
        "years": rows,
    }


def risk_metrics(curve: pd.DataFrame, initial_cash: float) -> dict[str, float]:
    df = _prepare_curve(curve)
    eq = df["equity"].to_numpy(dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = np.divide(eq, peak, out=np.ones_like(eq), where=peak > 0) - 1.0
    daily = pd.Series(eq).pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    elapsed_years = (df.iloc[-1]["date"] - df.iloc[0]["date"]).days / 365.2425
    final = float(eq[-1])
    total_return = final / initial_cash - 1.0
    cagr = (
        (final / initial_cash) ** (1.0 / elapsed_years) - 1.0
        if elapsed_years > 0 and final > 0 and initial_cash > 0
        else float("nan")
    )
    sample_std = float(np.std(daily, ddof=1)) if len(daily) > 1 else float("nan")
    annual_volatility = sample_std * math.sqrt(252.0) if math.isfinite(sample_std) else float("nan")
    sharpe = (
        float(np.mean(daily)) / sample_std * math.sqrt(252.0)
        if math.isfinite(sample_std) and sample_std > 0
        else float("nan")
    )
    max_drawdown = float(np.min(dd)) if len(dd) else float("nan")
    calmar = cagr / abs(max_drawdown) if math.isfinite(cagr) and max_drawdown < 0 else float("nan")
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown_close_to_close": max_drawdown,
        "annual_volatility_close_to_close": annual_volatility,
        "sharpe_rf0_close_to_close": sharpe,
        "calmar_close_to_close": calmar,
    }
