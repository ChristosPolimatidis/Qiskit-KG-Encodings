from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tasks.amplitude_similarity import run_amplitude_similarity_task
from src.tasks.basis_lookup import run_basis_lookup_task
from src.tasks.combined_demo import run_combined_demo_task
from src.tasks.keyword_search import run_keyword_search_task
from src.tasks.link_prediction_distance import run_link_prediction_distance_task
from src.tasks.multihop_phase_kickback import run_multihop_phase_kickback_task
from src.tasks.phase_filtering import run_phase_filtering_task
from src.tasks.schema_matching_qft import run_schema_matching_qft_task


OUTPUT_FOLDERS = (
    "logs",
    "tables",
    "plots",
    "json",
    "raw",
    "summary",
    "circuits",
    "histograms",
)

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

MANIFEST_FIELDS = [
    "experiment_name",
    "category",
    "encoding",
    "dataset",
    "size",
    "repetition",
    "status",
    "runtime_seconds",
    "output_files",
    "log_file",
    "error_message",
]

TEXT_FIELDS_TO_DROP = {
    "claim_note",
    "claim_scope",
    "method_note",
    "task_description",
}

NUMERIC_METRIC_KEYS = (
    "success_probability",
    "top_score",
    "max_absolute_error",
    "estimated_similarity",
    "classical_similarity",
    "classical_distance",
    "absolute_error",
    "relative_error",
    "marked_probability_before",
    "marked_probability_after",
    "fourier_pattern_similarity",
    "exact_distribution_similarity",
    "measured_distribution_similarity",
    "negative_control_similarity",
    "phase_error",
    "shot_phase_error",
    "expected_composed_phase",
    "estimated_phase",
    "state_norm",
    "num_qubits",
    "vector_dimension",
    "dimension",
    "circuit_depth",
    "gate_count",
    "transpiled_depth",
    "transpiled_gate_count",
    "task_time_seconds",
    "preparation_time_seconds",
    "normalization_time_seconds",
    "state_preparation_time_seconds",
    "simulation_time_seconds",
    "measurement_time_seconds",
    "readout_decoding_time_seconds",
)


class ExperimentFailure(RuntimeError):
    pass


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = {name: output_dir / name for name in OUTPUT_FOLDERS}
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
    return folders


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def compact_result_payload(result: Any) -> dict[str, Any]:
    payload = json_safe(result)
    if not isinstance(payload, dict):
        return {"value": payload}
    return {
        key: value
        for key, value in payload.items()
        if key not in TEXT_FIELDS_TO_DROP
    }


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(json_safe(row.get(field)), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else row.get(field, "")
                    for field in fields
                }
            )


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return None


def extract_metrics(payload: dict[str, Any]) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    for key in NUMERIC_METRIC_KEYS:
        number = numeric_value(payload.get(key))
        if number is not None:
            metrics.append({"metric_name": key, "metric_value": number})

    for mapping_name in (
        "estimated_scores",
        "classical_scores",
        "absolute_errors",
        "probabilities",
    ):
        mapping = payload.get(mapping_name)
        if isinstance(mapping, dict):
            for key, value in sorted(mapping.items()):
                number = numeric_value(value)
                if number is not None:
                    metrics.append(
                        {
                            "metric_name": f"{mapping_name}:{key}",
                            "metric_value": number,
                        }
                    )
    return metrics


def find_counts_payload(payload: dict[str, Any]) -> dict[str, int] | None:
    for key in ("counts", "measurement_counts", "counts_x", "counts_y"):
        counts = payload.get(key)
        if isinstance(counts, dict) and all(
            isinstance(value, (int, np.integer)) for value in counts.values()
        ):
            return {str(item): int(value) for item, value in counts.items()}
    nested = payload.get("counts_by_entity")
    if isinstance(nested, dict):
        combined: dict[str, int] = {}
        for entity, counts in nested.items():
            if not isinstance(counts, dict):
                continue
            for bitstring, value in counts.items():
                if isinstance(value, (int, np.integer)):
                    combined[f"{entity}:{bitstring}"] = int(value)
        return combined or None
    return None


def plot_bar(
    values: dict[str, float],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    rotation: int = 30,
) -> None:
    if not values:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(values)
    numbers = [float(values[label]) for label in labels]
    width = max(6.5, min(14.0, 0.55 * len(labels) + 3.0))
    fig, ax = plt.subplots(figsize=(width, 4.2))
    ax.bar(labels, numbers, color="#2563eb")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=rotation)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_experiment_artifacts(
    *,
    result: Any,
    experiment: dict[str, Any],
    repetition: int,
    folders: dict[str, Path],
    output_dir: Path,
    shots: int,
    repetitions: int,
    dataset: str,
    dataset_size: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = compact_result_payload(result)
    name = str(experiment["name"])
    stem = f"{name}_rep{repetition:02d}"
    json_path = folders["json"] / f"{stem}.json"
    raw_path = folders["raw"] / f"{stem}.json"
    circuit_path = folders["circuits"] / f"{stem}_metrics.json"
    metrics_path = folders["tables"] / f"{stem}_metrics.csv"

    write_json(json_path, payload)
    write_json(raw_path, payload)

    circuit_payload = {
        "experiment_name": name,
        "repetition": repetition,
        "num_qubits": payload.get("num_qubits"),
        "circuit_depth": payload.get("circuit_depth"),
        "gate_count": payload.get("gate_count"),
        "transpiled_depth": payload.get("transpiled_depth"),
        "transpiled_gate_count": payload.get("transpiled_gate_count"),
        "backend": payload.get("backend"),
    }
    write_json(circuit_path, circuit_payload)

    metrics = extract_metrics(payload)
    output_files = [
        relative(json_path, output_dir),
        relative(raw_path, output_dir),
        relative(circuit_path, output_dir),
    ]

    counts = find_counts_payload(payload)
    if counts:
        histogram_path = folders["histograms"] / f"{stem}_counts.png"
        plot_bar(
            {key: float(value) for key, value in counts.items()},
            title=f"{name} measurement counts",
            ylabel="count",
            output_path=histogram_path,
            rotation=45,
        )
        output_files.append(relative(histogram_path, output_dir))

    output_files.append(relative(metrics_path, output_dir))
    metric_rows = [
        {
            "experiment_name": name,
            "encoding": experiment["encoding"],
            "dataset": dataset,
            "size": dataset_size,
            "shots": shots,
            "repetitions": repetitions,
            "metric_name": metric["metric_name"],
            "metric_value": metric["metric_value"],
            "runtime_seconds": payload.get("task_time_seconds", ""),
            "output_files": ";".join(output_files),
        }
        for metric in metrics
    ]
    write_csv(metrics_path, metric_rows, SUMMARY_FIELDS)
    return metric_rows, output_files


def build_experiments() -> list[dict[str, Any]]:
    return [
        {
            "name": "basis_lookup",
            "category": "basis_encoding",
            "encoding": "basis",
            "runner": lambda shots, seed, repetition: run_basis_lookup_task(
                shots=shots,
                seed_simulator=seed + repetition,
            ),
        },
        {
            "name": "amplitude_similarity",
            "category": "amplitude_encoding",
            "encoding": "amplitude",
            "runner": lambda shots, seed, repetition: run_amplitude_similarity_task(
                shots=shots,
                seed_simulator=seed + repetition,
            ),
        },
        {
            "name": "phase_filtering",
            "category": "phase_encoding",
            "encoding": "phase",
            "runner": lambda shots, seed, repetition: run_phase_filtering_task(
                index_mode="sequential",
                shots=shots,
                seed_simulator=seed + repetition,
            ),
        },
        {
            "name": "entity_matching",
            "category": "entity_matching",
            "encoding": "amplitude",
            "runner": lambda shots, seed, repetition: run_link_prediction_distance_task(
                shots=shots,
                seed_simulator=seed + repetition,
            ),
        },
        {
            "name": "keyword_search",
            "category": "keyword_search",
            "encoding": "amplitude",
            "runner": lambda shots, seed, repetition: run_keyword_search_task(
                shots=shots,
                seed_simulator=seed + repetition,
                repetitions=1,
            ),
        },
        {
            "name": "schema_matching",
            "category": "schema_matching",
            "encoding": "phase",
            "runner": lambda shots, seed, repetition: run_schema_matching_qft_task(
                shots=shots,
                seed_simulator=seed + repetition,
            ),
        },
        {
            "name": "multi_hop_phase",
            "category": "phase_accumulation",
            "encoding": "phase",
            "runner": lambda shots, seed, repetition: run_multihop_phase_kickback_task(
                shots=shots,
                seed_simulator=seed + repetition,
            ),
        },
        {
            "name": "combined_encoding",
            "category": "combined_encoding",
            "encoding": "combined",
            "runner": lambda shots, seed, repetition: run_combined_demo_task(
                index_mode="sequential",
                output_path=None,
            ),
        },
    ]


def run_one_experiment(
    *,
    experiment: dict[str, Any],
    repetition: int,
    runner: Callable[[int, int, int], Any],
    folders: dict[str, Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    output_files: list[str] = []
    try:
        result = runner(args.shots, args.seed, repetition)
        metric_rows, output_files = write_experiment_artifacts(
            result=result,
            experiment=experiment,
            repetition=repetition,
            folders=folders,
            output_dir=output_dir,
            shots=args.shots,
            repetitions=args.repetitions,
            dataset=args.dataset,
            dataset_size=args.dataset_size,
        )
        status = "success"
        error_message = ""
    except Exception as exc:
        metric_rows = []
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        failure_path = folders["logs"] / f"{experiment['name']}_rep{repetition:02d}.log"
        failure_path.write_text(error_message + "\n", encoding="utf-8")
        output_files = [relative(failure_path, output_dir)]
    runtime = time.perf_counter() - start
    manifest_row = {
        "experiment_name": experiment["name"],
        "category": experiment["category"],
        "encoding": experiment["encoding"],
        "dataset": args.dataset,
        "size": args.dataset_size,
        "repetition": repetition,
        "status": status,
        "runtime_seconds": round(runtime, 9),
        "output_files": output_files,
        "log_file": output_files[0] if status == "failed" else "",
        "error_message": error_message,
    }
    return metric_rows, manifest_row


def plot_suite_summaries(rows: list[dict[str, Any]], folders: dict[str, Path]) -> None:
    runtimes: dict[str, float] = {}
    key_metrics: dict[str, float] = {}
    for row in rows:
        name = str(row["experiment_name"])
        runtime = numeric_value(row.get("runtime_seconds"))
        if runtime is not None:
            runtimes[name] = max(runtime, runtimes.get(name, 0.0))
        metric_name = str(row.get("metric_name", ""))
        if metric_name in {
            "success_probability",
            "estimated_similarity",
            "top_score",
            "marked_probability_after",
            "measured_distribution_similarity",
            "estimated_phase",
            "state_norm",
        }:
            key_metrics[f"{name}:{metric_name}"] = float(row["metric_value"])
    plot_bar(
        runtimes,
        title="Runtime by experiment",
        ylabel="seconds",
        output_path=folders["plots"] / "runtime_by_experiment.png",
        rotation=45,
    )
    plot_bar(
        key_metrics,
        title="Selected numeric outputs",
        ylabel="value",
        output_path=folders["plots"] / "selected_metrics.png",
        rotation=60,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the KG encoding experiment suite and write numeric outputs."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--dataset",
        default="data/running_example.ttl",
        help="Dataset label/path recorded in output metadata.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=6,
        help="Dataset size recorded in output metadata.",
    )
    parser.add_argument(
        "--profile",
        choices=("light", "medium", "hard"),
        default="light",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    folders = ensure_output_dirs(output_dir)
    experiments = build_experiments()
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        for repetition in range(1, args.repetitions + 1):
            rows, manifest_row = run_one_experiment(
                experiment=experiment,
                repetition=repetition,
                runner=experiment["runner"],
                folders=folders,
                output_dir=output_dir,
                args=args,
            )
            summary_rows.extend(rows)
            manifest_rows.append(manifest_row)
            if manifest_row["status"] == "failed" and args.profile in {"light", "medium"}:
                break
        if any(
            row["status"] == "failed" and row["experiment_name"] == experiment["name"]
            for row in manifest_rows
        ) and args.profile in {"light", "medium"}:
            break

    raw_metrics_path = folders["raw"] / "experiment_metrics.csv"
    tables_metrics_path = folders["tables"] / "experiment_metrics.csv"
    summary_path = folders["summary"] / "results_summary.csv"
    manifest_json_path = folders["summary"] / "core_experiment_manifest.json"
    manifest_csv_path = folders["summary"] / "core_experiment_manifest.csv"

    write_csv(raw_metrics_path, summary_rows, SUMMARY_FIELDS)
    write_csv(tables_metrics_path, summary_rows, SUMMARY_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_json(manifest_json_path, {"experiments": manifest_rows})
    write_csv(manifest_csv_path, manifest_rows, MANIFEST_FIELDS)
    plot_suite_summaries(summary_rows, folders)

    failed = [row for row in manifest_rows if row["status"] == "failed"]
    if failed and args.profile in {"light", "medium"}:
        raise ExperimentFailure(
            f"{len(failed)} core experiment(s) failed. See {manifest_json_path}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
