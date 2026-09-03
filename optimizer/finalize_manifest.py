#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
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


def tracked_source_paths(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            "README.md",
            "optimizer",
            "tests",
            ".github/workflows",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    paths = sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())
    if not paths:
        raise SystemExit("git ls-files nao retornou fontes do otimizador")
    required = {
        "optimizer/finalize_manifest.py",
        "optimizer/optimize_b3_pine.py",
        "optimizer/optimize_b3_pine_fast.py",
        "optimizer/audit_fast_precompute.py",
        "optimizer/audit_portfolio_management.py",
        "optimizer/b3_strategy_live_universe.json",
        "optimizer/b3_strategy_live_selection.csv",
        "optimizer/b3_strategy_live_corporate_action_overrides.json",
        ".github/workflows/b3-pine-exhaustive.yml",
    }
    missing = sorted(required - set(paths))
    if missing:
        raise SystemExit(f"fontes obrigatorias nao versionadas/nao selecionadas: {missing}")
    return paths


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_paths = tracked_source_paths(repo_root)
    source_hashes: dict[str, str] = {}
    for relative in source_paths:
        path = repo_root / relative
        if not path.is_file():
            raise SystemExit(f"arquivo versionado ausente ao finalizar manifest: {relative}")
        source_hashes[relative] = sha256_file(path)

    execution = manifest.get("execution") or {}
    manifest.update(
        {
            "schema_version": 3,
            "optimizer_repository": "mycroft440/otimizacao",
            "optimizer_sha": args.optimizer_sha,
            "runtime": {
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "platform": platform.platform(),
            },
            "source_identity": {
                "selection": "all git-tracked files under README.md, optimizer/, tests/, .github/workflows/",
                "tracked_file_count": len(source_hashes),
                "files_sha256": source_hashes,
            },
            # Backward-compatible alias for consumers of schema v2.
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
                "tracked_source_files": len(source_hashes),
                "top_100_sha256": manifest["selection_provenance"]["top_100_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
