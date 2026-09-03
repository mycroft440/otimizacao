from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))

import optimize_b3_pine as opt
import optimize_b3_pine_vol_grid as grid


class ExactVolGridAccelerationTests(unittest.TestCase):
    def _market(self):
        closes_a = np.array(
            [10.0, 10.2, 10.1, 10.4, 10.8, 10.7, 10.9, 11.2, 11.1, 11.5, 11.7, 11.6],
            dtype=np.float64,
        )
        closes_b = np.array(
            [20.0, 19.8, 20.2, 20.1, 20.4, 20.6, 20.5, 20.9, 21.0, 20.8, 21.2, 21.4],
            dtype=np.float64,
        )
        decision_index = np.array(
            [
                [1, 3, 5, 7, 9, 11],
                [1, 3, 5, 7, 9, 11],
            ],
            dtype=np.int32,
        )
        return SimpleNamespace(
            tickers=["AAA3", "BBB4"],
            execution_dates=pd.date_range("2020-01-06", periods=6, freq="7D"),
            decision_index=decision_index,
            frames={
                "AAA3": pd.DataFrame({"close": closes_a}),
                "BBB4": pd.DataFrame({"close": closes_b}),
            },
        )

    def test_all_vol_prefix_sum_cache_is_bit_exact_to_canonical_gate(self):
        market = self._market()
        periods = list(range(2, 11))
        cached = grid.compute_all_vol_valid(market, periods)
        self.assertEqual(cached.shape, (9, 6, 2))
        for vi, period in enumerate(periods):
            canonical = grid.compute_vol_valid(market, period)
            self.assertTrue(
                np.array_equal(cached[vi], canonical),
                f"VOL_PERIOD={period}",
            )

    def test_identical_vol_gates_are_grouped_only_on_exact_bytes(self):
        gates = np.array(
            [
                [[False, True], [True, True]],
                [[False, True], [True, True]],
                [[True, True], [True, True]],
                [[False, True], [True, True]],
            ],
            dtype=np.bool_,
        )
        groups = grid.group_identical_vol_gates([2, 3, 4, 5], gates)
        self.assertEqual([vols for vols, _gate in groups], [[2, 3, 5], [4]])
        self.assertTrue(np.array_equal(groups[0][1], gates[0]))
        self.assertTrue(np.array_equal(groups[1][1], gates[2]))

    def test_cached_momentum_ranking_matches_canonical_targets_with_ties(self):
        rng = np.random.default_rng(20260903)
        M, P, W, N = 8, 11, 13, 7
        momentum = rng.normal(size=(M, W, N)).astype(np.float64)
        momentum[rng.random(momentum.shape) < 0.10] = np.nan
        # Empates exatos exercitam a exigência de sort estável.
        momentum[:, :, 1] = np.where(
            rng.random((M, W)) < 0.35,
            momentum[:, :, 0],
            momentum[:, :, 1],
        )
        gap_state = rng.random((P, W, N)) < 0.55
        vol_valid = rng.random((W, N)) < 0.80

        order, positive = grid.build_momentum_rank_cache(momentum)
        for mi in range(M):
            expected = opt.first_ranked_targets(
                gap_state,
                momentum[mi],
                vol_valid,
            )
            actual = grid.first_ranked_targets_from_cache(
                gap_state,
                positive[mi],
                vol_valid,
                order[mi],
            )
            self.assertTrue(np.array_equal(actual, expected), f"momentum index {mi}")

    def test_numpy_top_k_matches_full_stable_dataframe_sort(self):
        pairs = [(g, s) for g in range(5, 9) for s in range(2, 6)]
        momentum_values = list(range(5, 12))
        P = len(pairs)
        M = len(momentum_values)
        # Inteiros convertidos para float geram muitos empates na fronteira do Top-K.
        rng = np.random.default_rng(440)
        equity = rng.integers(900, 1100, size=(M, P)).astype(np.float64)
        top_k = 31

        selected = grid.exact_top_k_indices(
            equity,
            pairs,
            momentum_values,
            top_k,
        )

        full = pd.DataFrame(
            {
                "idx": np.arange(M * P, dtype=np.int64),
                "gap_period": np.tile(
                    np.asarray([g for g, _s in pairs], dtype=np.int32),
                    M,
                ),
                "signal_period": np.tile(
                    np.asarray([s for _g, s in pairs], dtype=np.int32),
                    M,
                ),
                "momentum_period": np.repeat(
                    np.asarray(momentum_values, dtype=np.int32),
                    P,
                ),
                "final_equity": equity.reshape(-1),
            }
        ).sort_values(
            [
                "final_equity",
                "gap_period",
                "signal_period",
                "momentum_period",
            ],
            ascending=[False, True, True, True],
            kind="stable",
        )
        expected = full.head(top_k)["idx"].to_numpy(dtype=np.int64)
        self.assertTrue(np.array_equal(selected, expected))


if __name__ == "__main__":
    unittest.main()