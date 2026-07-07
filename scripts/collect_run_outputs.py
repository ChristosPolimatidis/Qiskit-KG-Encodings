from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "experiment_name",
    "encoding",
    "dataset",
    "size",
    "shots",
    "repetitions",
    "metric_name",
    "metric_value",
    "runtime_seconds",
    "output_files",
]

ARTIFACT_FIELDS = [
    "relative_path",
    "kind",
    "size_bytes",
]

SCALING_METRICS = [
    "total_runtime",
    "completed_total_runtime",
    "rdf_parse_time",
    "id_mapping_time",
    "state_preparation_time",
    "circuit_construction_time",
    "simulation_time",
    "measurement_time",
    "qubits",
    "circuit_depth",
    "gate_count",
    "logical_circuit_depth",
    "logical_gate_count",
    "decomposed_circuit_depth",
    "decomposed_gate_count",
    "transpiled_circuit_depth",
    "transpiled_gate_count",
    "phase_marked_triples",
    "phase_marked_fraction",
]

BASELINE_METRICS = [
    "expected_value",
    "measured_value",
    "runtime_seconds",
]


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".png":
        return "png"
    if suffix in {".log", ".txt"}:
        return "log"
    if suffix == ".xml":
        return "xml"
    if suffix == ".zip":
        return "zip"
    return "artifact"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def build_artifact_index(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": relative(path, run_dir),
                "kind": artifact_kind(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def load_core_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "summary" / "results_summary.csv"
    return read_csv(path)


def scaling_rows(run_dir: Path, repetitions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in sorted((run_dir / "raw" / "scaling").rglob("scaling_raw_results.csv")):
        for source in read_csv(raw_path):
            output_ref = relative(raw_path, run_dir)
            for metric_name in SCALING_METRICS:
                metric_value = parse_number(source.get(metric_name))
                if metric_value is None:
                    continue
                rows.append(
                    {
                        "experiment_name": "scaling",
                        "encoding": source.get("encoding", ""),
                        "dataset": source.get("dataset_name", ""),
                        "size": source.get("dataset_size") or source.get("num_triples", ""),
                        "shots": source.get("shots", ""),
                        "repetitions": repetitions,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "runtime_seconds": source.get("completed_total_runtime")
                        or source.get("total_runtime", ""),
                        "output_files": output_ref,
                    }
                )
    return rows


def baseline_rows(run_dir: Path, shots: int, repetitions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_path = run_dir / "raw" / "classical_baselines" / "classical_baselines_raw.csv"
    for source in read_csv(raw_path):
        output_ref = relative(raw_path, run_dir)
        for metric_name in BASELINE_METRICS:
            metric_value = parse_number(source.get(metric_name))
            if metric_value is None:
                continue
            rows.append(
                {
                    "experiment_name": source.get("task", "classical_baseline"),
                    "encoding": "classical",
                    "dataset": "running_example",
                    "size": source.get("input_size", ""),
                    "shots": shots,
                    "repetitions": repetitions,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "runtime_seconds": source.get("runtime_seconds", ""),
                    "output_files": output_ref,
                }
            )
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect run artifacts and normalize numeric summary rows."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    summary_rows = load_core_rows(run_dir)
    summary_rows.extend(scaling_rows(run_dir, args.repetitions))
    summary_rows.extend(baseline_rows(run_dir, args.shots, args.repetitions))
    artifact_rows = build_artifact_index(run_dir)

    write_csv(run_dir / "summary" / "results_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(run_dir / "summary" / "artifact_index.csv", artifact_rows, ARTIFACT_FIELDS)
    (run_dir / "json" / "artifact_index.json").write_text(
        json.dumps(artifact_rows, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Collected {len(summary_rows)} numeric summary rows.")
    print(f"Indexed {len(artifact_rows)} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
