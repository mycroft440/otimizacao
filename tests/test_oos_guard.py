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
    def make_files(self, directory: Path, training_end="2023-12-31"):
        top = directory / "top_100.csv"
        top.write_text("gap_period,signal_period,momentum_period,vol_period\n41,15,49,21\n", encoding="utf-8")
        digest = hashlib.sha256(top.read_bytes()).hexdigest()
        manifest = directory / "MANIFEST.json"
        manifest.write_text(
            json.dumps({
                "optimizer_sha": "abc",
                "upstream_sha": "def",
                "execution": {"start": "2018-01-02", "end": training_end},
                "selection_provenance": {"mode": "training_only", "top_100_sha256": digest},
            }),
            encoding="utf-8",
        )
        return top, manifest

    def test_verified_training_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, manifest = self.make_files(Path(tmp))
            result = oos.verify_training_source(top, manifest, "2024-01-01")
            self.assertEqual(result["training_end"], "2023-12-31")

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


if __name__ == "__main__":
    unittest.main()
