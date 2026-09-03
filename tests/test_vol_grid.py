from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))

import config
import optimize_b3_pine as opt


class VolatilityGridTests(unittest.TestCase):
    def test_config_accepts_vol_period_one(self):
        config.validate_run_config(
            start="2018-01-02",
            end="2019-01-02",
            gap_min=5,
            gap_max=80,
            signal_min=2,
            signal_max=60,
            momentum_min=5,
            momentum_max=252,
            vol_period=1,
            initial_cash=1000.0,
            fee_bps=3.25,
            slippage_bps=10.0,
            odd_lot_extra_bps=5.0,
            shard_id=0,
            shards=20,
        )

    def test_vol_period_one_never_passes_positive_vol_gate(self):
        closes = np.array([10.0, 11.0, 9.0, 12.0, 12.5], dtype=np.float64)
        gate = opt.sample_vol_positive(closes, 1)
        self.assertEqual(gate.dtype, np.bool_)
        self.assertFalse(bool(np.any(gate)))

    def test_default_exhaustive_vol_range_contains_21(self):
        values = list(range(config.DEFAULT_VOL_MIN, config.DEFAULT_VOL_MAX + 1))
        self.assertEqual(values[0], 1)
        self.assertEqual(values[-1], 60)
        self.assertEqual(len(values), 60)
        self.assertIn(config.DEFAULT_VOL_PERIOD, values)


if __name__ == "__main__":
    unittest.main()
