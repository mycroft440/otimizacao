from __future__ import annotations

import numpy as np

import optimize_b3_pine as opt


def simulate_momentum_batch(
    targets_batch: np.ndarray,
    gap_state: np.ndarray,
    momentum_batch: np.ndarray,
    vol_valid: np.ndarray,
    market,
    *,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    odd_lot_extra_bps: float,
) -> dict[str, np.ndarray]:
    """Vectorize independent Momentum parameter portfolios without cross-combo reductions."""
    B, P, W = targets_batch.shape
    if momentum_batch.shape[0] != B or momentum_batch.shape[1] != W:
        raise ValueError("Momentum batch shape incompatível com targets.")
    Q = B * P
    fee = fee_bps / 10000.0
    base = slippage_bps / 10000.0
    extra = odd_lot_extra_bps / 10000.0

    combo_m = np.repeat(np.arange(B, dtype=np.int32), P)
    combo_p = np.tile(np.arange(P, dtype=np.int32), B)
    cash = np.full(Q, float(initial_cash), dtype=np.float64)
    shares = np.zeros(Q, dtype=np.int64)
    holding = np.full(Q, -1, dtype=np.int16)
    trades = np.zeros(Q, dtype=np.int32)
    skipped = np.zeros(Q, dtype=np.int32)
    fees_paid = np.zeros(Q, dtype=np.float64)
    slip_paid = np.zeros(Q, dtype=np.float64)

    for w in range(W):
        target = targets_batch[:, :, w].reshape(Q).copy()
        hmask = (holding >= 0) & (target >= 0) & (holding != target)
        if np.any(hmask):
            rows = np.flatnonzero(hmask)
            b = combo_m[rows]
            p = combo_p[rows]
            h = holding[rows].astype(np.int64)
            t = target[rows].astype(np.int64)
            inc_m = momentum_batch[b, w, h]
            top_m = momentum_batch[b, w, t]
            inc_ok = (
                gap_state[p, w, h]
                & vol_valid[w, h]
                & np.isfinite(inc_m)
                & (inc_m > 0.0)
            )
            tie = inc_ok & np.isfinite(top_m) & (inc_m == top_m)
            if np.any(tie):
                target[rows[tie]] = holding[rows[tie]]

        changed = target != holding
        rows = np.flatnonzero(changed)
        if len(rows):
            projected = cash[rows].copy()
            valid = np.ones(len(rows), dtype=np.bool_)
            old = holding[rows].astype(np.int64)
            new = target[rows].astype(np.int64)

            sell_mask = old >= 0
            if np.any(sell_mask):
                rr = np.flatnonzero(sell_mask)
                raw = market.exec_open[w, old[rr]]
                ok = np.isfinite(raw) & (raw > 0.0)
                valid[rr] &= ok
                if np.any(ok):
                    q = shares[rows[rr]][ok]
                    raw_ok = raw[ok]
                    slip = opt.slip_amount(raw_ok, q, base, extra)
                    gross = raw_ok * q - slip
                    projected[rr[ok]] += gross - gross * fee

            buy_mask = new >= 0
            if np.any(buy_mask):
                rr = np.flatnonzero(buy_mask)
                raw = market.exec_open[w, new[rr]]
                ok = np.isfinite(raw) & (raw > 0.0)
                valid[rr] &= ok
                if np.any(ok):
                    qq = opt.affordable_qty(projected[rr[ok]], raw[ok], fee, base, extra)
                    valid[rr[ok]] &= qq > 0

            good_rows = rows[valid]
            bad_rows = rows[~valid]
            if len(bad_rows):
                skipped[bad_rows] += 1

            if len(good_rows):
                oldg = holding[good_rows].astype(np.int64)
                newg = target[good_rows].astype(np.int64)
                sell = oldg >= 0
                if np.any(sell):
                    gr = good_rows[sell]
                    old_asset = oldg[sell]
                    raw = market.exec_open[w, old_asset]
                    q = shares[gr]
                    slip = opt.slip_amount(raw, q, base, extra)
                    gross = raw * q - slip
                    ff = gross * fee
                    cash[gr] += gross - ff
                    fees_paid[gr] += ff
                    slip_paid[gr] += slip
                    shares[gr] = 0
                    holding[gr] = -1
                    trades[gr] += 1

                buy = newg >= 0
                if np.any(buy):
                    gr = good_rows[buy]
                    new_asset = newg[buy]
                    raw = market.exec_open[w, new_asset]
                    q = opt.affordable_qty(cash[gr], raw, fee, base, extra)
                    odd = q % 100
                    gross = raw * ((1.0 + base) * q + extra * odd)
                    ff = gross * fee
                    slip = raw * (base * q + extra * odd)
                    new_cash = cash[gr] - gross - ff
                    if np.any(new_cash < -1e-7):
                        raise RuntimeError("batch invariant: compra gerou caixa negativo")
                    cash[gr] = np.maximum(new_cash, 0.0)
                    shares[gr] = q
                    holding[gr] = new_asset.astype(np.int16)
                    fees_paid[gr] += ff
                    slip_paid[gr] += slip
                    trades[gr] += (q > 0).astype(np.int32)

        if np.any(cash < -1e-7):
            raise RuntimeError("batch invariant: caixa negativo")
        if np.any(shares < 0):
            raise RuntimeError("batch invariant: quantidade negativa")
        if np.any((holding < 0) != (shares == 0)):
            raise RuntimeError("batch invariant: holding e quantidade divergentes")

    equity = cash.copy()
    invested = holding >= 0
    if np.any(invested):
        rows = np.flatnonzero(invested)
        px = market.final_close[holding[rows].astype(np.int64)]
        if np.any(~np.isfinite(px)) or np.any(px <= 0.0):
            raise RuntimeError("batch invariant: preço final inválido")
        equity[rows] += shares[rows] * px

    shape = (B, P)
    return {
        "final_equity": equity.reshape(shape),
        "total_return": (equity / float(initial_cash) - 1.0).reshape(shape),
        "cash": cash.reshape(shape),
        "shares": shares.reshape(shape),
        "holding": holding.reshape(shape),
        "trades": trades.reshape(shape),
        "skipped": skipped.reshape(shape),
        "fees": fees_paid.reshape(shape),
        "slippage": slip_paid.reshape(shape),
    }
