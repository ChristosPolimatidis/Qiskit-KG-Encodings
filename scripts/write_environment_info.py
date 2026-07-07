from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE_NAMES = (
    "qiskit",
    "qiskit-aer",
    "rdflib",
    "numpy",
    "matplotlib",
    "pytest",
)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "package_versions": package_versions(),
        "git_commit": git_commit(repo_root),
        "os": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write experiment run environment info.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--repo-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_payload(repo_root), indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote environment info: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
