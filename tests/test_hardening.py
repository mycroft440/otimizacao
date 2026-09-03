from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))

import config
import metrics
import optimize_b3_pine as opt
import reference_engine as ref


class ConfigTests(unittest.TestCase):
    def test_required_warmup_covers_default_maxima(self):
        value = config.required_warmup_sessions(80, 60, 252, 21)
        self.assertGreaterEqual(value, 263)
        self.assertGreaterEqual(value, 80 + 60 + 2)

    def test_rejects_invalid_financial_inputs(self):
        base = dict(
            start="2018-01-02", end="2019-01-02",
            gap_min=5, gap_max=80, signal_min=2, signal_max=60,
            momentum_min=5, momentum_max=252, vol_period=21,
            initial_cash=1000.0, fee_bps=3.25, slippage_bps=10.0,
            odd_lot_extra_bps=5.0, shard_id=0, shards=20,
        )
        config.validate_run_config(**base)
        for key, value in [
            ("initial_cash", 0.0), ("fee_bps", -1.0),
            ("slippage_bps", -1.0), ("odd_lot_extra_bps", -1.0),
        ]:
            bad = dict(base)
            bad[key] = value
            with self.assertRaises(ValueError):
                config.validate_run_config(**bad)


class MetricsTests(unittest.TestCase):
    def test_partial_first_year_is_not_complete(self):
        dates = pd.date_range("2020-07-01", "2021-12-30", freq="B")
        curve = pd.DataFrame({"date": dates, "equity": np.linspace(1000.0, 1500.0, len(dates))})
        result = metrics.annual_metrics(curve, 1000.0)
        rows = {row["year"]: row for row in result["years"]}
        self.assertFalse(rows[2020]["complete_year"])
        self.assertTrue(rows[2021]["complete_year"])
        self.assertEqual(result["complete_years"], 1)

    def test_partial_terminal_year_is_not_complete(self):
        dates = pd.date_range("2020-01-02", "2021-08-31", freq="B")
        curve = pd.DataFrame({"date": dates, "equity": np.linspace(1000.0, 1400.0, len(dates))})
        result = metrics.annual_metrics(curve, 1000.0)
        rows = {row["year"]: row for row in result["years"]}
        self.assertTrue(rows[2020]["complete_year"])
        self.assertFalse(rows[2021]["complete_year"])


class AffordableQtyPropertyTests(unittest.TestCase):
    def test_closed_form_matches_independent_reference(self):
        rng = random.Random(440)
        for _ in range(1000):
            cash = rng.uniform(1.0, 100000.0)
            raw = rng.uniform(0.05, 1000.0)
            fee = rng.uniform(0.0, 0.005)
            base = rng.uniform(0.0, 0.01)
            extra = rng.uniform(0.0, 0.01)
            got = int(opt.affordable_qty(
                np.array([cash]), np.array([raw]), fee, base, extra
            )[0])
            expected = ref.affordable_qty_reference(cash, raw, fee, base, extra)
            self.assertEqual(got, expected, (cash, raw, fee, base, extra, got, expected))
            cost = ref._buy_cost(raw, got, fee, base, extra)
            self.assertLessEqual(cost, cash + 1e-7)
            if got >= 0:
                next_cost = ref._buy_cost(raw, got + 1, fee, base, extra)
                self.assertGreater(next_cost, cash - 1e-7)


class PortfolioCostTests(unittest.TestCase):
    @staticmethod
    def market():
        execution_dates = pd.DatetimeIndex([pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-13")])
        decision_dates = pd.DatetimeIndex([pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-10")])
        frames = {
            "AAA3": pd.DataFrame({"date": decision_dates.append(execution_dates).sort_values(), "open": 50.0, "close": 50.0}),
            "BBB3": pd.DataFrame({"date": decision_dates.append(execution_dates).sort_values(), "open": 25.0, "close": 25.0}),
        }
        return opt.MarketData(
            tickers=["AAA3", "BBB3"], frames=frames,
            master_dates=decision_dates.append(execution_dates).sort_values(),
            execution_dates=execution_dates, decision_dates=decision_dates,
            exec_open=np.array([[50.0, 25.0], [60.0, 30.0]], dtype=float),
            decision_index=np.zeros((2, 2), dtype=np.int32),
            final_close=np.array([60.0, 30.0]), start="2020-01-06", end="2020-01-13",
        )

    def test_switch_with_real_costs_reconciles_reference(self):
        market = self.market()
        targets = np.array([[0, 1]], dtype=np.int16)
        gap = np.ones((1, 2, 2), dtype=bool)
        mom = np.array([[0.2, 0.1], [0.1, 0.3]], dtype=float)
        vol = np.ones((2, 2), dtype=bool)
        result = opt.simulate_pairs(
            targets, gap, mom, vol, market,
            initial_cash=1000.0, fee_bps=3.25, slippage_bps=10.0,
            odd_lot_extra_bps=5.0,
        )
        self.assertGreaterEqual(float(result["cash"][0]), 0.0)
        self.assertGreater(float(result["fees"][0]), 0.0)
        self.assertGreater(float(result["slippage"][0]), 0.0)
        self.assertEqual(int(result["holding"][0]), 1)
        self.assertEqual(int(result["trades"][0]), 3)


if __name__ == "__main__":
    unittest.main()
