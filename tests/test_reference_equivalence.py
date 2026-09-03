from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))

import fast_batch
import fast_shared
import optimize_b3_pine as opt
import optimize_b3_pine_fast as fast
import reference_engine as ref


class ReferenceEngineSyntheticTests(unittest.TestCase):
    def build_market(self):
        rng = np.random.default_rng(12345)
        dates = pd.bdate_range("2017-01-02", "2020-12-31")
        tickers = ["AAA3", "BBB3", "CCC3"]
        frames = {}
        for index, ticker in enumerate(tickers):
            returns = rng.normal(0.0003 + index * 0.00005, 0.012, len(dates))
            close = 20.0 * np.cumprod(1.0 + returns)
            overnight = rng.normal(0.0, 0.004, len(dates))
            open_ = close * (1.0 + overnight)
            frames[ticker] = pd.DataFrame({"date": dates, "open": open_, "close": close})

        master = pd.DatetimeIndex(dates)
        iso = master.isocalendar()
        key = (iso["year"].astype(str) + "-" + iso["week"].astype(str)).to_numpy()
        first = np.ones(len(master), dtype=bool)
        first[1:] = key[1:] != key[:-1]
        positions = np.flatnonzero(first & (master >= pd.Timestamp("2018-01-02")))
        positions = positions[positions > 0]
        execution_dates = master[positions]
        decision_dates = master[positions - 1]
        exec_open = np.full((len(execution_dates), len(tickers)), np.nan)
        decision_index = np.full((len(tickers), len(execution_dates)), -1, dtype=np.int32)
        final_close = np.zeros(len(tickers))
        for ti, ticker in enumerate(tickers):
            frame = frames[ticker]
            index = pd.Index(frame["date"])
            eidx = index.get_indexer(execution_dates)
            didx = index.get_indexer(decision_dates)
            exec_open[:, ti] = frame["open"].to_numpy()[eidx]
            decision_index[ti] = didx
            final_close[ti] = float(frame.iloc[-1]["close"])
        return opt.MarketData(
            tickers=tickers,
            frames=frames,
            master_dates=master,
            execution_dates=execution_dates,
            decision_dates=decision_dates,
            exec_open=exec_open,
            decision_index=decision_index,
            final_close=final_close,
            start="2018-01-02",
            end="2020-12-31",
        )

    def test_indicators_targets_and_portfolio_match(self):
        market = self.build_market()
        for g, s, m in [(5, 2, 5), (20, 9, 30), (40, 20, 63), (80, 60, 252)]:
            pairs, gap, momentum, vol = opt.precompute_shard(market, [g], [s], [m], 21)
            targets = opt.first_ranked_targets(gap, momentum[0], vol)
            slow = ref.simulate_reference(
                market, g, s, m, 21,
                initial_cash=1000.0,
                fee_bps=3.25,
                slippage_bps=10.0,
                odd_lot_extra_bps=5.0,
            )
            self.assertEqual(pairs, [(g, s)])
            self.assertTrue(np.array_equal(gap[0], slow["gap_state"]), (g, s, m, "gap"))
            self.assertTrue(np.allclose(momentum[0], slow["momentum"], equal_nan=True, rtol=1e-12, atol=1e-12), (g, s, m, "momentum"))
            self.assertTrue(np.array_equal(vol, slow["vol_valid"]), (g, s, m, "vol"))
            self.assertTrue(np.array_equal(targets[0], slow["base_targets"]), (g, s, m, "targets"))
            primary = opt.simulate_pairs(
                targets, gap, momentum[0], vol, market,
                initial_cash=1000.0,
                fee_bps=3.25,
                slippage_bps=10.0,
                odd_lot_extra_bps=5.0,
            )
            self.assertAlmostEqual(float(primary["final_equity"][0]), float(slow["final_equity"]), places=7)
            self.assertEqual(int(primary["holding"][0]), int(slow["holding"]))
            self.assertEqual(int(primary["shares"][0]), int(slow["shares"]))
            self.assertEqual(int(primary["trades"][0]), int(slow["trades"]))
            self.assertEqual(int(primary["skipped"][0]), int(slow["skipped"]))

    def test_parallel_gap_and_momentum_batch_are_bit_exact(self):
        market = self.build_market()
        gaps, signals, momentums = [5, 20], [2, 9], [5, 30, 63]
        original = opt.precompute_shard(market, gaps, signals, momentums, 21)
        old_workers = os.environ.get("B3_PRECOMPUTE_WORKERS")
        os.environ["B3_PRECOMPUTE_WORKERS"] = "3"
        try:
            accelerated = fast.fast_precompute_shard(market, gaps, signals, momentums, 21)
        finally:
            if old_workers is None:
                os.environ.pop("B3_PRECOMPUTE_WORKERS", None)
            else:
                os.environ["B3_PRECOMPUTE_WORKERS"] = old_workers
        self.assertEqual(original[0], accelerated[0])
        self.assertTrue(np.array_equal(original[1], accelerated[1]))
        self.assertTrue(np.array_equal(original[2], accelerated[2], equal_nan=True))
        self.assertTrue(np.array_equal(original[3], accelerated[3]))

        pairs, gap, momentum, vol = original
        targets = np.stack([
            opt.first_ranked_targets(gap, momentum[mi], vol)
            for mi in range(len(momentums))
        ])
        batch = fast_batch.simulate_momentum_batch(
            targets, gap, momentum, vol, market,
            initial_cash=1000.0,
            fee_bps=3.25,
            slippage_bps=10.0,
            odd_lot_extra_bps=5.0,
        )
        keys = ["final_equity", "cash", "shares", "holding", "trades", "skipped", "fees", "slippage"]
        for mi in range(len(momentums)):
            scalar = opt.simulate_pairs(
                targets[mi], gap, momentum[mi], vol, market,
                initial_cash=1000.0,
                fee_bps=3.25,
                slippage_bps=10.0,
                odd_lot_extra_bps=5.0,
            )
            for key in keys:
                if np.issubdtype(scalar[key].dtype, np.floating):
                    self.assertTrue(np.array_equal(scalar[key], batch[key][mi], equal_nan=True), (momentums[mi], key))
                else:
                    self.assertTrue(np.array_equal(scalar[key], batch[key][mi]), (momentums[mi], key))
        self.assertEqual(len(pairs), 4)

    def test_binary_fast_cache_reloads_exact_arrays(self):
        market = self.build_market()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/universes").mkdir(parents=True)
            (root / "data/candles").mkdir(parents=True)
            (root / "data/universes/fixed_40_2018.json").write_text(
                json.dumps({"tickers": market.tickers}), encoding="utf-8"
            )
            for ticker in market.tickers:
                market.frames[ticker].to_csv(
                    root / "data/candles" / f"{ticker.lower()}_1d.csv", index=False
                )
            fast_shared.build_fast_cache(root, market.start, market.end, 5, 63, 21)
            cached = fast_shared.load_fast_cache(root, market.start, market.end, [5, 30, 63], 21)
            self.assertIsNotNone(cached)
            original = opt.precompute_shard(market, [5, 20], [2, 9], [5, 30, 63], 21)
            accelerated = fast.fast_precompute_shard(cached, [5, 20], [2, 9], [5, 30, 63], 21)
            self.assertEqual(original[0], accelerated[0])
            self.assertTrue(np.array_equal(original[1], accelerated[1]))
            self.assertTrue(np.array_equal(original[2], accelerated[2], equal_nan=True))
            self.assertTrue(np.array_equal(original[3], accelerated[3]))

    def test_gap_state_uses_same_float64_accumulation_contract(self):
        rng = np.random.default_rng(20260903)
        n = 3000
        raw_gaps = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-8.0, 3.0, n)
        close = np.full(n, 1_000_000.0, dtype=np.float64)
        open_ = close.copy()
        open_[1:] = close[:-1] + raw_gaps[1:]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2010-01-04", periods=n),
            "open": open_,
            "close": close,
        })
        gaps = np.zeros(n, dtype=np.float64)
        gaps[1:] = open_[1:] - close[:-1]
        positive = np.maximum(gaps, 0.0)
        negative = np.maximum(-gaps, 0.0)
        pos_sum = opt.rolling_sum(positive, 5)
        neg_sum = opt.rolling_sum(negative, 5)
        ratio = np.full(n, np.nan, dtype=np.float64)
        valid = np.isfinite(pos_sum) & np.isfinite(neg_sum)
        ratio[valid & (neg_sum == 0.0)] = 1.0
        nz = valid & (neg_sum != 0.0)
        ratio[nz] = 100.0 * pos_sum[nz] / neg_sum[nz]
        signal = opt.rolling_mean_contiguous(ratio, 2, 4)
        expected = opt.persistent_direction_state(signal)
        actual, _momentum, _vol = ref._indicator_for_ticker(frame, 5, 2, 5, 21)
        self.assertTrue(np.array_equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
