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

import fast_shared
import optimize_b3_pine as opt
import optimize_b3_pine_fast as fast


class FastCliEquivalenceTests(unittest.TestCase):
    def _dataset(self, root: Path):
        rng = np.random.default_rng(20260903)
        dates = pd.bdate_range("2017-01-02", "2020-12-31")
        tickers = ["AAA3", "BBB3", "CCC3"]
        (root / "data/universes").mkdir(parents=True)
        (root / "data/candles").mkdir(parents=True)
        (root / "data/universes/fixed_40_2018.json").write_text(
            json.dumps({"tickers": tickers}), encoding="utf-8"
        )
        for ti, ticker in enumerate(tickers):
            returns = rng.normal(0.0002 + ti * 0.0001, 0.013, len(dates))
            close = 30.0 * np.cumprod(1.0 + returns)
            open_ = close * (1.0 + rng.normal(0.0, 0.004, len(dates)))
            pd.DataFrame({"date": dates, "open": open_, "close": close}).to_csv(
                root / "data/candles" / f"{ticker.lower()}_1d.csv", index=False
            )

    def _argv(self, root: Path, output: Path):
        return [
            "optimizer",
            "--data-root", str(root),
            "--output", str(output),
            "--start", "2018-01-02",
            "--end", "2020-12-31",
            "--gap-min", "5",
            "--gap-max", "6",
            "--signal-min", "2",
            "--signal-max", "3",
            "--momentum-min", "5",
            "--momentum-max", "7",
            "--vol-period", "21",
            "--initial-cash", "1000",
            "--fee-bps", "3.25",
            "--slippage-bps", "10",
            "--odd-lot-extra-bps", "5",
            "--shard-id", "0",
            "--shards", "1",
        ]

    def test_fast_cli_csv_is_byte_exact_to_scalar_cli_with_parallel_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._dataset(root)
            scalar_out = root / "scalar.csv"
            fast_out = root / "fast.csv"
            old_argv = sys.argv[:]
            old_batch = os.environ.get("B3_MOMENTUM_BATCH")
            old_workers = os.environ.get("B3_BATCH_WORKERS")
            try:
                sys.argv = self._argv(root, scalar_out)
                opt.main()
                fast_shared.build_fast_cache(root, "2018-01-02", "2020-12-31", 5, 7, 21)
                os.environ["B3_MOMENTUM_BATCH"] = "1"
                os.environ["B3_BATCH_WORKERS"] = "3"
                sys.argv = self._argv(root, fast_out)
                fast.main()
            finally:
                sys.argv = old_argv
                if old_batch is None:
                    os.environ.pop("B3_MOMENTUM_BATCH", None)
                else:
                    os.environ["B3_MOMENTUM_BATCH"] = old_batch
                if old_workers is None:
                    os.environ.pop("B3_BATCH_WORKERS", None)
                else:
                    os.environ["B3_BATCH_WORKERS"] = old_workers
            self.assertEqual(scalar_out.read_bytes(), fast_out.read_bytes())


if __name__ == "__main__":
    unittest.main()
