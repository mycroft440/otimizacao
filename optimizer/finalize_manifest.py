#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--top", required=True, type=Path)
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--optimizer-sha", required=True)
    p.add_argument("--selection-mode", choices=["in_sample_full", "training_only"], required=True)
    return p.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_paths = [
        "optimizer/requirements.txt",
        "optimizer/config.py",
        "optimizer/metrics.py",
        "optimizer/optimize_b3_pine.py",
        "optimizer/optimize_b3_pine_fast.py",
        "optimizer/reference_engine.py",
        "optimizer/reduce_results.py",
        "optimizer/audit_exhaustive_grid.py",
        "optimizer/audit_results_integrity.py",
        "optimizer/audit_shard_set.py",
        "optimizer/audit_reference_engine.py",
        "optimizer/audit_snapshot.py",
        "optimizer/audit_stale_prices.py",
        "optimizer/audit_warmup.py",
        "optimizer/audit_pine_golden.py",
        "optimizer/audit_corporate_actions.py",
        "optimizer/analyze_robustness.py",
        "optimizer/build_annual_report.py",
        "optimizer/harden_best_report.py",
        "optimizer/validate_top_oos.py",
        "optimizer/walk_forward_validate.py",
        "optimizer/walk_forward_windows.json",
        "tests/test_hardening.py",
        "tests/test_reference_equivalence.py",
        "tests/test_oos_guard.py",
        "tests/test_second_review_regressions.py",
        "tests/test_workflow_supply_chain.py",
        "tests/fixtures/pine_reference/README.md",
        "README.md",
        ".github/workflows/b3-pine-exhaustive.yml",
        ".github/workflows/b3-pine-walk-forward.yml",
        ".github/workflows/hardening-ci.yml",
    ]
    source_hashes = {}
    for relative in source_paths:
        path = args.repo_root / relative
        if not path.exists():
            raise SystemExit(f"arquivo de fonte ausente ao finalizar manifest: {relative}")
        source_hashes[relative] = sha256_file(path)

    execution = manifest.get("execution") or {}
    manifest.update(
        {
            "schema_version": 2,
            "optimizer_repository": "mycroft440/otimizacao",
            "optimizer_sha": args.optimizer_sha,
            "runtime": {
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "platform": platform.platform(),
            },
            "optimizer_source_sha256": source_hashes,
            "selection_provenance": {
                "mode": args.selection_mode,
                "training_start": execution.get("start"),
                "training_end": execution.get("end"),
                "top_100_sha256": sha256_file(args.top),
            },
        }
    )
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "optimizer_sha": args.optimizer_sha,
                "selection_mode": args.selection_mode,
                "top_100_sha256": manifest["selection_provenance"]["top_100_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
