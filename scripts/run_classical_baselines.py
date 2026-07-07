from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classical_baselines import BASELINE_SCOPE, run_all_classical_baselines


RAW_FIELDS = [
    "task",
    "baseline_method",
    "input_size",
    "expected_value",
    "measured_value",
    "runtime_seconds",
    "notes",
    "claim_scope",
]

SUMMARY_FIELDS = [
    "task",
    "baseline_count",
    "mean_runtime_seconds",
    "methods",
    "claim_scope",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def cell_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), sort_keys=True)
    return str(value)


def write_csv(rows: list[dict[str, Any]], output_path: Path, fields: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: cell_value(row.get(field, "")) for field in fields})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        runtimes = [float(row["runtime_seconds"]) for row in task_rows]
        summary_rows.append(
            {
                "task": task,
                "baseline_count": len(task_rows),
                "mean_runtime_seconds": sum(runtimes) / len(runtimes),
                "methods": "; ".join(str(row["baseline_method"]) for row in task_rows),
                "claim_scope": BASELINE_SCOPE,
            }
        )
    return summary_rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic classical sanity baselines for toy KG tasks."
    )
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = [result.to_dict() for result in run_all_classical_baselines()]
    summary_rows = summarize(raw_rows)
    raw_path = output_dir / "classical_baselines_raw.csv"
    summary_path = output_dir / "classical_baselines_summary.csv"
    json_path = output_dir / "classical_baselines.json"

    write_csv(raw_rows, raw_path, RAW_FIELDS)
    write_csv(summary_rows, summary_path, SUMMARY_FIELDS)
    payload = {
        "settings": vars(args),
        "claim_scope": BASELINE_SCOPE,
        "raw_rows": raw_rows,
        "summary_rows": summary_rows,
        "output_files": {
            "raw": str(raw_path),
            "summary": str(summary_path),
            "json": str(json_path),
        },
    }
    json_path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print("Classical baselines complete")
    print(f"  Raw CSV: {raw_path}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  JSON: {json_path}")
    return payload


if __name__ == "__main__":
    main()
