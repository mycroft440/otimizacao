from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))

import validate_top_oos as oos


class OOSGuardTests(unittest.TestCase):
    def make_files(self, directory: Path, training_end="2023-12-31", fee_bps=3.25):
        top = directory / "top_100.csv"
        top.write_text("gap_period,signal_period,momentum_period,vol_period\n41,15,49,21\n", encoding="utf-8")
        digest = hashlib.sha256(top.read_bytes()).hexdigest()
        manifest = directory / "MANIFEST.json"
        manifest.write_text(
            json.dumps({
                "schema_version": 2,
                "optimizer_sha": "a" * 40,
                "upstream_sha": "b" * 40,
                "optimizer_source_sha256": {
                    "optimizer/optimize_b3_pine.py": "c" * 64,
                    "optimizer/config.py": "d" * 64,
                },
                "execution": {
                    "start": "2018-01-02",
                    "end": training_end,
                    "initial_cash": 1000.0,
                    "fee_bps_per_side": fee_bps,
                    "slippage_bps_per_side": 10.0,
                    "odd_lot_extra_bps_weighted": 5.0,
                },
                "selection_provenance": {
                    "mode": "training_only",
                    "training_start": "2018-01-02",
                    "training_end": training_end,
                    "top_100_sha256": digest,
                },
            }),
            encoding="utf-8",
        )
        return top, manifest

    def expected_config(self):
        return {
            "initial_cash": 1000.0,
            "fee_bps": 3.25,
            "slippage_bps": 10.0,
            "odd_lot_extra_bps": 5.0,
        }

    def test_verified_training_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, manifest = self.make_files(Path(tmp))
            result = oos.verify_training_source(
                top, manifest, "2024-01-01", expected_config=self.expected_config()
            )
            self.assertEqual(result["training_end"], "2023-12-31")
            self.assertTrue(result["training_config_matches_oos"])
            self.assertEqual(result["training_source_file_count"], 2)

    def test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, manifest = self.make_files(Path(tmp), training_end="2024-01-01")
            with self.assertRaises(SystemExit):
                oos.verify_training_source(top, manifest, "2024-01-01")

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, manifest = self.make_files(Path(tmp))
            top.write_text(top.read_text(encoding="utf-8") + "42,16,50,21\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                oos.verify_training_source(top, manifest, "2024-01-01")

    def test_non_training_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, manifest = self.make_files(Path(tmp))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["selection_provenance"]["mode"] = "in_sample_full"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SystemExit):
                oos.verify_training_source(top, manifest, "2024-01-01")

    def test_cost_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, manifest = self.make_files(Path(tmp), fee_bps=4.0)
            with self.assertRaises(SystemExit):
                oos.verify_training_source(
                    top, manifest, "2024-01-01", expected_config=self.expected_config()
                )

    def test_missing_source_hashes_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, manifest = self.make_files(Path(tmp))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload.pop("optimizer_source_sha256")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SystemExit):
                oos.verify_training_source(top, manifest, "2024-01-01")

    def test_short_git_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, manifest = self.make_files(Path(tmp))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["optimizer_sha"] = "abc"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SystemExit):
                oos.verify_training_source(top, manifest, "2024-01-01")


if __name__ == "__main__":
    unittest.main()
