from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))
import config  # noqa: E402
import finalize_manifest  # noqa: E402
import optimize_b3_pine as opt  # noqa: E402
import optimize_b3_pine_fast as fast  # noqa: E402
import reduce_results as red  # noqa: E402


class WorkflowSupplyChainTests(unittest.TestCase):
    def test_all_official_actions_are_pinned_to_full_commit_sha(self):
        pattern = re.compile(r"uses:\s*actions/[^@\s]+@([^\s#]+)")
        sha40 = re.compile(r"^[0-9a-f]{40}$")
        failures = []
        for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            for ref in pattern.findall(text):
                if not sha40.fullmatch(ref):
                    failures.append(f"{path.name}: {ref}")
        self.assertFalse(failures, "Actions oficiais sem pin imutavel: " + ", ".join(failures))

    def test_manifest_source_identity_auto_includes_critical_tracked_files(self):
        paths = set(finalize_manifest.tracked_source_paths(ROOT))
        required = {
            "optimizer/finalize_manifest.py",
            "optimizer/audit_fast_precompute.py",
            "optimizer/audit_portfolio_management.py",
            "optimizer/b3_strategy_live_universe.json",
            "optimizer/b3_strategy_live_selection.csv",
            "optimizer/b3_strategy_live_corporate_action_overrides.json",
            "tests/test_walk_forward_guard.py",
            ".github/workflows/b3-pine-exhaustive.yml",
            ".github/workflows/b3-pine-walk-forward.yml",
        }
        self.assertFalse(required - paths, f"fontes criticas fora do manifest: {sorted(required-paths)}")


class CanonicalCliDefaultsTests(unittest.TestCase):
    def test_reference_optimizer_uses_canonical_cost_defaults(self):
        with mock.patch.object(
            sys,
            "argv",
            ["optimize_b3_pine.py", "--data-root", "x", "--output", "y"],
        ):
            args = opt.parse_args()
        self.assertEqual(args.initial_cash, config.DEFAULT_INITIAL_CASH)
        self.assertEqual(args.fee_bps, config.DEFAULT_FEE_BPS)
        self.assertEqual(args.slippage_bps, config.DEFAULT_SLIPPAGE_BPS)
        self.assertEqual(args.odd_lot_extra_bps, config.DEFAULT_ODD_LOT_EXTRA_BPS)

    def test_reducer_uses_same_canonical_cost_defaults(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "reduce_results.py",
                "--data-root",
                "x",
                "--results-dir",
                "r",
                "--output-dir",
                "o",
            ],
        ):
            args = red.parse_args()
        self.assertEqual(args.initial_cash, config.DEFAULT_INITIAL_CASH)
        self.assertEqual(args.fee_bps, config.DEFAULT_FEE_BPS)
        self.assertEqual(args.slippage_bps, config.DEFAULT_SLIPPAGE_BPS)
        self.assertEqual(args.odd_lot_extra_bps, config.DEFAULT_ODD_LOT_EXTRA_BPS)

    def test_fast_optimizer_respects_equals_form_override(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "optimize_b3_pine_fast.py",
                "--data-root=x",
                "--output=y",
                "--fee-bps=4.0",
                "--slippage-bps=12.5",
                "--odd-lot-extra-bps=7.5",
                "--initial-cash=2500",
            ],
        ):
            args = fast._hardened_parse_args()
        self.assertEqual(args.initial_cash, 2500.0)
        self.assertEqual(args.fee_bps, 4.0)
        self.assertEqual(args.slippage_bps, 12.5)
        self.assertEqual(args.odd_lot_extra_bps, 7.5)


if __name__ == "__main__":
    unittest.main()
