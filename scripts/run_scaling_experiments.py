from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import queue
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_scaling_datasets import DATASET_CONFIGS, generate_dataset
from src.amplitude_encoding import (
    build_amplitude_encoding_artifacts,
    prepare_amplitude_state_from_normalized_vector,
)
from src.basis_encoding import (
    build_uniform_encoded_state_circuit_from_encoded_states,
    encode_triples_as_basis_states,
)
from src.id_mapper import build_encoding_context
from src.kg_parser import load_triples
from src.main import counts_to_probabilities
from src.phase_encoding import (
    apply_phase_oracle,
    build_phase_interference_from_marked_circuit,
    build_uniform_index_state,
    compute_phase_angles,
    predicate_phase_marker,
    zero_phase_marker,
)


ENCODINGS = ("basis", "amplitude", "phase")
DATASET_CATEGORIES = ("synthetic", "real")
DEFAULT_REAL_KG_DIR = Path("data/real_kgs")
REAL_KG_FILES = (
    "exampleV3.ttl",
    "productsSmall.rdf",
    "DecodedOntologies_V2.ttl",
    "Aristotle.xml",
)
SYNTHETIC_PHASE_PREDICATE = "http://example.org/p0"
PHASE_MARKER_MODES = (
    "synthetic-default",
    "first-predicate",
    "most-common-predicate",
    "none",
    "custom",
)

RAW_FIELDNAMES = [
    "dataset_category",
    "dataset_name",
    "dataset_path",
    "dataset_size",
    "num_triples",
    "num_entities",
    "num_predicates",
    "encoding",
    "repetition",
    "success",
    "status",
    "error_message",
    "total_runtime",
    "rdf_parse_time",
    "id_mapping_time",
    "state_preparation_time",
    "circuit_construction_time",
    "simulation_time",
    "measurement_time",
    "qubits",
    "circuit_depth",
    "gate_count",
    "operation_counts",
    "logical_circuit_depth",
    "logical_gate_count",
    "logical_operation_counts",
    "decomposed_circuit_depth",
    "decomposed_gate_count",
    "decomposed_operation_counts",
    "transpiled_circuit_depth",
    "transpiled_gate_count",
    "transpiled_operation_counts",
    "metric_status",
    "metric_error",
    "shots",
    "backend",
    "started_at_utc",
    "finished_at_utc",
    "entity_bit_width",
    "predicate_bit_width",
    "measurement_circuit_preparation_time",
    "decoding_time",
    "completed_total_runtime",
    "phase_marker_mode",
    "phase_requested_predicate",
    "phase_effective_predicate",
    "phase_marked_triples",
    "phase_total_triples",
    "phase_marked_fraction",
    "phase_warning",
    "max_basis_simulation_qubits",
    "max_phase_diagonal_qubits",
    "max_metric_qubits",
    "timeout_seconds",
    "weight_strategy",
    "phase_mark_predicate",
    "phase_angle",
    "decompose_reps",
    "compute_decomposed_metrics",
    "compute_transpiled_metrics",
    "rdf_format",
    "original_num_triples",
    "max_real_triples",
    "truncation_applied",
]

SUMMARY_FIELDNAMES = [
    "dataset_category",
    "dataset_name",
    "dataset_path",
    "dataset_size",
    "num_triples",
    "encoding",
    "run_count",
    "success_count",
    "failure_count",
    "statuses",
    "mean_total_runtime",
    "std_total_runtime",
    "mean_completed_total_runtime",
    "std_completed_total_runtime",
    "mean_rdf_parse_time",
    "mean_id_mapping_time",
    "mean_state_preparation_time",
    "mean_circuit_construction_time",
    "mean_simulation_time",
    "mean_measurement_time",
    "mean_qubits",
    "mean_circuit_depth",
    "mean_gate_count",
    "mean_logical_circuit_depth",
    "mean_logical_gate_count",
    "mean_decomposed_circuit_depth",
    "mean_decomposed_gate_count",
    "mean_transpiled_circuit_depth",
    "mean_transpiled_gate_count",
    "mean_phase_marked_triples",
    "mean_phase_marked_fraction",
    "shots",
    "backend",
]

PLOT_METRICS = [
    ("mean_completed_total_runtime", "Total runtime (seconds)", "total_runtime"),
    (
        "mean_state_preparation_time",
        "State preparation time (seconds)",
        "state_preparation_time",
    ),
    ("mean_simulation_time", "Simulation time (seconds)", "simulation_time"),
    ("mean_qubits", "Number of qubits", "qubits"),
    ("mean_logical_circuit_depth", "Logical circuit depth", "logical_circuit_depth"),
    ("mean_logical_gate_count", "Logical gate count", "logical_gate_count"),
    (
        "mean_decomposed_circuit_depth",
        "Decomposed circuit depth",
        "decomposed_circuit_depth",
    ),
    ("mean_decomposed_gate_count", "Decomposed gate count", "decomposed_gate_count"),
    (
        "mean_transpiled_circuit_depth",
        "Transpiled circuit depth",
        "transpiled_circuit_depth",
    ),
    ("mean_transpiled_gate_count", "Transpiled gate count", "transpiled_gate_count"),
]

ENCODING_COLORS = {
    "basis": "#2563eb",
    "amplitude": "#d97706",
    "phase": "#059669",
}
ENCODING_MARKERS = {
    "basis": "o",
    "amplitude": "s",
    "phase": "D",
}
CATEGORY_MARKERS = {
    "synthetic": "o",
    "real": "^",
}
CATEGORY_LINESTYLES = {
    "synthetic": "-",
    "real": "--",
}
PLOT_BACKGROUND = "#fbfaf7"
AXIS_BACKGROUND = "#ffffff"
GRID_COLOR = "#d8dee9"


@dataclass(frozen=True)
class DatasetSpec:
    category: str
    path: Path
    size: int | None = None


class SimulatorLimitError(RuntimeError):
    """Raised when a configured guard prevents an unsafe dense simulation."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timed_call(function, *args, **kwargs) -> tuple[Any, float]:
    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start_time


def rounded_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 9)


def dense_state_size_message(num_qubits: int) -> str:
    amplitudes = 2**num_qubits
    gib = amplitudes * 16 / (1024**3)
    return (
        f"{num_qubits} qubits require a dense state vector with "
        f"{amplitudes:,} complex amplitudes (about {gib:.2f} GiB)."
    )


def dense_unitary_size_message(num_qubits: int) -> str:
    dimension = 2**num_qubits
    gib = dimension * dimension * 16 / (1024**3)
    return (
        f"{num_qubits} qubits require a dense {dimension:,} x {dimension:,} "
        f"phase oracle matrix (about {gib:.2f} GiB)."
    )


def empty_row(
    dataset_path: Path,
    dataset_category: str,
    encoding: str,
    repetition: int,
    shots: int,
    dataset_size: int | None = None,
) -> dict[str, Any]:
    row = {field: None for field in RAW_FIELDNAMES}
    row.update(
        {
            "dataset_category": dataset_category,
            "dataset_name": dataset_path.stem,
            "dataset_path": str(dataset_path),
            "dataset_size": dataset_size,
            "encoding": encoding,
            "repetition": repetition,
            "success": False,
            "status": "not_started",
            "error_message": "",
            "shots": shots,
            "backend": "Qiskit Statevector",
            "started_at_utc": utc_timestamp(),
        }
    )
    return row


def apply_runtime_settings(
    row: dict[str, Any],
    args: argparse.Namespace,
    timeout_seconds: float | None = None,
) -> None:
    row.update(
        {
            "max_basis_simulation_qubits": getattr(
                args,
                "max_basis_simulation_qubits",
                None,
            ),
            "max_phase_diagonal_qubits": getattr(
                args,
                "max_phase_diagonal_qubits",
                None,
            ),
            "max_metric_qubits": getattr(args, "max_metric_qubits", None),
            "timeout_seconds": timeout_seconds,
            "weight_strategy": getattr(args, "weight_strategy", None),
            "phase_mark_predicate": row.get("phase_mark_predicate")
            or getattr(args, "phase_mark_predicate", None),
            "phase_requested_predicate": row.get("phase_requested_predicate")
            or getattr(args, "phase_mark_predicate", None),
            "phase_marker_mode": row.get("phase_marker_mode")
            or getattr(args, "phase_marker_mode", None),
            "phase_angle": getattr(args, "phase_angle", None),
            "decompose_reps": getattr(args, "decompose_reps", None),
            "compute_decomposed_metrics": getattr(
                args,
                "compute_decomposed_metrics",
                None,
            ),
            "compute_transpiled_metrics": getattr(
                args,
                "compute_transpiled_metrics",
                None,
            ),
            "rdf_format": getattr(args, "rdf_format", None),
            "max_real_triples": getattr(args, "max_real_triples", None),
        }
    )


def update_context_metrics(row: dict[str, Any], context) -> None:
    row.update(
        {
            "num_triples": context.triple_count,
            "num_entities": context.entity_count,
            "num_predicates": context.predicate_count,
            "entity_bit_width": context.entity_bit_width,
            "predicate_bit_width": context.predicate_bit_width,
        }
    )
    if row.get("dataset_size") is None:
        row["dataset_size"] = context.triple_count


def count_operations(circuit: QuantumCircuit) -> dict[str, int]:
    return {
        str(operation): int(count)
        for operation, count in circuit.count_ops().items()
    }


def operation_count_json(operation_counts: dict[str, int]) -> str:
    return json.dumps(operation_counts, sort_keys=True)


def metric_limit_message(metric_name: str, circuit: QuantumCircuit, limit: int) -> str:
    return (
        f"{metric_name} metrics skipped because the circuit has "
        f"{circuit.num_qubits} qubits and --max-metric-qubits is {limit}."
    )


def store_circuit_metric_prefix(
    row: dict[str, Any],
    prefix: str,
    circuit: QuantumCircuit,
) -> None:
    operation_counts = count_operations(circuit)
    row.update(
        {
            f"{prefix}_circuit_depth": circuit.depth(),
            f"{prefix}_gate_count": sum(operation_counts.values()),
            f"{prefix}_operation_counts": operation_count_json(operation_counts),
        }
    )


def update_circuit_metrics(
    row: dict[str, Any],
    circuit: QuantumCircuit,
    args: argparse.Namespace,
) -> None:
    row["qubits"] = circuit.num_qubits

    store_circuit_metric_prefix(row, "logical", circuit)
    row["circuit_depth"] = row["logical_circuit_depth"]
    row["gate_count"] = row["logical_gate_count"]
    row["operation_counts"] = row["logical_operation_counts"]

    metric_statuses: list[str] = []
    metric_errors: list[str] = []
    max_metric_qubits = getattr(args, "max_metric_qubits", None)

    if getattr(args, "compute_decomposed_metrics", True):
        if max_metric_qubits is not None and circuit.num_qubits > max_metric_qubits:
            if "metric_limit" not in metric_statuses:
                metric_statuses.append("metric_limit")
            metric_errors.append(
                metric_limit_message("Decomposed", circuit, max_metric_qubits)
            )
        else:
            try:
                decomposed = circuit.decompose(
                    reps=max(1, int(getattr(args, "decompose_reps", 1)))
                )
                store_circuit_metric_prefix(row, "decomposed", decomposed)
            except Exception as exc:
                if "error" not in metric_statuses:
                    metric_statuses.append("error")
                metric_errors.append(
                    f"Decomposed metrics failed: {type(exc).__name__}: {exc}"
                )
    else:
        if "not_requested" not in metric_statuses:
            metric_statuses.append("not_requested")

    if getattr(args, "compute_transpiled_metrics", True):
        if max_metric_qubits is not None and circuit.num_qubits > max_metric_qubits:
            if "metric_limit" not in metric_statuses:
                metric_statuses.append("metric_limit")
            metric_errors.append(
                metric_limit_message("Transpiled", circuit, max_metric_qubits)
            )
        else:
            try:
                transpiled_circuit = transpile(
                    circuit,
                    basis_gates=["u", "cx"],
                    optimization_level=0,
                )
                store_circuit_metric_prefix(row, "transpiled", transpiled_circuit)
            except Exception as exc:
                if "error" not in metric_statuses:
                    metric_statuses.append("error")
                metric_errors.append(
                    f"Transpiled metrics failed: {type(exc).__name__}: {exc}"
                )
    else:
        if "not_requested" not in metric_statuses:
            metric_statuses.append("not_requested")

    row["metric_status"] = ";".join(metric_statuses) if metric_statuses else "success"
    row["metric_error"] = "; ".join(metric_errors)


def build_basis_uniform_statevector(
    encoded_states: list[dict[str, Any]],
    num_qubits: int,
) -> Statevector:
    vector = np.zeros(2**num_qubits, dtype=complex)
    amplitude = 1 / math.sqrt(len(encoded_states))
    for encoded_state in encoded_states:
        vector[int(encoded_state["bitstring"], 2)] = amplitude
    return Statevector(vector)


def simulate_basis_statevector(
    encoded_states: list[dict[str, Any]],
    num_qubits: int,
) -> dict[str, float]:
    statevector = build_basis_uniform_statevector(
        encoded_states=encoded_states,
        num_qubits=num_qubits,
    )
    return {
        state: float(probability)
        for state, probability in sorted(statevector.probabilities_dict().items())
        if probability > 0
    }


def sample_basis_statevector_counts(
    encoded_states: list[dict[str, Any]],
    num_qubits: int,
    shots: int,
) -> dict[str, int]:
    statevector = build_basis_uniform_statevector(
        encoded_states=encoded_states,
        num_qubits=num_qubits,
    )
    return {
        state: int(count)
        for state, count in sorted(statevector.sample_counts(shots).items())
    }


def simulate_vector_probabilities(vector: np.ndarray) -> dict[str, float]:
    statevector = Statevector(np.asarray(vector, dtype=complex))
    return {
        state: float(probability)
        for state, probability in sorted(statevector.probabilities_dict().items())
        if probability > 0
    }


def sample_vector_counts(vector: np.ndarray, shots: int) -> dict[str, int]:
    statevector = Statevector(np.asarray(vector, dtype=complex))
    return {
        state: int(count)
        for state, count in sorted(statevector.sample_counts(shots).items())
    }


def sample_statevector_counts(statevector: Statevector, shots: int) -> dict[str, int]:
    return {
        state: int(count)
        for state, count in sorted(statevector.sample_counts(shots).items())
    }


def run_basis_scaling(row: dict[str, Any], context, args: argparse.Namespace) -> None:
    row["qubits"] = context.total_basis_qubits
    row["backend"] = "Qiskit Statevector"

    encoded_states, metadata_elapsed = timed_call(
        encode_triples_as_basis_states,
        context=context,
    )
    row["state_preparation_time"] = rounded_seconds(metadata_elapsed)

    if context.total_basis_qubits > args.max_basis_simulation_qubits:
        raise SimulatorLimitError(
            "Basis dense simulation skipped by guard. "
            f"{dense_state_size_message(context.total_basis_qubits)} "
            f"Configured limit: {args.max_basis_simulation_qubits} qubits."
        )

    circuit, circuit_elapsed = timed_call(
        build_uniform_encoded_state_circuit_from_encoded_states,
        encoded_states=encoded_states,
        num_qubits=context.total_basis_qubits,
    )
    row["circuit_construction_time"] = rounded_seconds(circuit_elapsed)
    update_circuit_metrics(row, circuit, args)

    _, simulation_elapsed = timed_call(
        simulate_basis_statevector,
        encoded_states,
        context.total_basis_qubits,
    )
    counts, measurement_elapsed = timed_call(
        sample_basis_statevector_counts,
        encoded_states,
        context.total_basis_qubits,
        args.shots,
    )
    _, decoding_elapsed = timed_call(counts_to_probabilities, counts)

    row["simulation_time"] = rounded_seconds(simulation_elapsed)
    row["measurement_circuit_preparation_time"] = 0.0
    row["measurement_time"] = rounded_seconds(measurement_elapsed)
    row["decoding_time"] = rounded_seconds(decoding_elapsed)


def run_amplitude_scaling(
    row: dict[str, Any],
    context,
    args: argparse.Namespace,
) -> None:
    row["backend"] = "Qiskit Statevector"

    artifacts, state_elapsed = timed_call(
        build_amplitude_encoding_artifacts,
        triples=context.triples,
        weights=None,
        strategy=args.weight_strategy,
    )
    row["state_preparation_time"] = rounded_seconds(state_elapsed)
    row["qubits"] = artifacts["num_qubits"]

    circuit, circuit_elapsed = timed_call(
        prepare_amplitude_state_from_normalized_vector,
        normalized_vector=artifacts["normalized_vector"],
        name="AmplitudeScaling",
    )
    row["circuit_construction_time"] = rounded_seconds(circuit_elapsed)
    update_circuit_metrics(row, circuit, args)

    _, simulation_elapsed = timed_call(
        simulate_vector_probabilities,
        artifacts["normalized_vector"],
    )
    counts, measurement_elapsed = timed_call(
        sample_vector_counts,
        artifacts["normalized_vector"],
        args.shots,
    )
    _, decoding_elapsed = timed_call(counts_to_probabilities, counts)

    row["simulation_time"] = rounded_seconds(simulation_elapsed)
    row["measurement_circuit_preparation_time"] = 0.0
    row["measurement_time"] = rounded_seconds(measurement_elapsed)
    row["decoding_time"] = rounded_seconds(decoding_elapsed)


def resolve_phase_marker_mode(
    dataset_category: str,
    args: argparse.Namespace,
) -> str:
    requested_predicate = getattr(args, "phase_mark_predicate", None)
    configured_mode = getattr(args, "phase_marker_mode", None)
    if configured_mode:
        return configured_mode
    if requested_predicate:
        return "custom"
    if dataset_category == "synthetic":
        return "synthetic-default"
    return "most-common-predicate"


def most_common_predicate(context) -> str | None:
    if not context.triples:
        return None
    predicate_counts = Counter(triple.predicate for triple in context.triples)
    return predicate_counts.most_common(1)[0][0]


def select_phase_mark_fn(row: dict[str, Any], context, args: argparse.Namespace):
    requested_predicate = getattr(args, "phase_mark_predicate", None)
    mode = resolve_phase_marker_mode(str(row.get("dataset_category") or ""), args)

    effective_predicate: str | None
    phase_warning = ""
    if mode == "custom":
        if not requested_predicate:
            raise ValueError(
                "--phase-marker-mode custom requires --phase-mark-predicate."
            )
        effective_predicate = requested_predicate
    elif mode == "synthetic-default":
        effective_predicate = SYNTHETIC_PHASE_PREDICATE
    elif mode == "first-predicate":
        effective_predicate = context.triples[0].predicate if context.triples else None
    elif mode == "most-common-predicate":
        effective_predicate = most_common_predicate(context)
    elif mode == "none":
        effective_predicate = None
        phase_warning = "Phase marker mode 'none'; phase oracle is a no-op baseline."
    else:
        raise ValueError(f"Unsupported phase marker mode '{mode}'.")

    row["phase_marker_mode"] = mode
    row["phase_requested_predicate"] = requested_predicate or ""
    row["phase_mark_predicate"] = requested_predicate or ""
    row["phase_effective_predicate"] = effective_predicate or ""
    row["phase_total_triples"] = context.triple_count
    row["phase_warning"] = phase_warning

    if effective_predicate:
        return predicate_phase_marker(
            predicate_uri=effective_predicate,
            phase_value=args.phase_angle,
        )
    return zero_phase_marker


def run_phase_scaling(row: dict[str, Any], context, args: argparse.Namespace) -> None:
    row["backend"] = "Qiskit Statevector"
    mark_fn = select_phase_mark_fn(row, context, args)

    state_start = time.perf_counter()
    initial_state = build_uniform_index_state(context.triple_count)
    phase_angles, marked_triples = compute_phase_angles(context=context, mark_fn=mark_fn)
    state_elapsed = time.perf_counter() - state_start

    num_qubits = int(math.log2(len(initial_state)))
    row["qubits"] = num_qubits
    row["phase_marked_triples"] = len(marked_triples)
    row["phase_total_triples"] = context.triple_count
    row["phase_marked_fraction"] = (
        round(len(marked_triples) / context.triple_count, 9)
        if context.triple_count
        else None
    )
    if len(marked_triples) == 0 and row.get("phase_marker_mode") != "none":
        row["phase_warning"] = (
            "No triples matched the phase predicate; phase oracle is a no-op "
            "for this dataset."
        )
    row["state_preparation_time"] = rounded_seconds(state_elapsed)

    if num_qubits > args.max_phase_diagonal_qubits:
        raise SimulatorLimitError(
            "Phase dense-oracle simulation skipped by guard. "
            f"{dense_unitary_size_message(num_qubits)} "
            f"Configured limit: {args.max_phase_diagonal_qubits} qubits."
        )

    circuit_start = time.perf_counter()
    before_circuit = QuantumCircuit(num_qubits, name="PhaseMarkedScaling")
    before_circuit.initialize(initial_state, before_circuit.qubits)
    apply_phase_oracle(circuit=before_circuit, phase_angles=phase_angles)
    after_circuit = build_phase_interference_from_marked_circuit(before_circuit)
    circuit_elapsed = time.perf_counter() - circuit_start

    row["circuit_construction_time"] = rounded_seconds(circuit_elapsed)
    update_circuit_metrics(row, after_circuit, args)

    before_statevector, before_simulation_elapsed = timed_call(
        Statevector.from_instruction,
        before_circuit,
    )
    after_statevector, after_simulation_elapsed = timed_call(
        Statevector.from_instruction,
        after_circuit,
    )

    before_counts, before_measure = timed_call(
        sample_statevector_counts,
        before_statevector,
        args.shots,
    )
    after_counts, after_measure = timed_call(
        sample_statevector_counts,
        after_statevector,
        args.shots,
    )
    _, before_decoding = timed_call(counts_to_probabilities, before_counts)
    _, after_decoding = timed_call(counts_to_probabilities, after_counts)

    row["simulation_time"] = rounded_seconds(
        before_simulation_elapsed + after_simulation_elapsed
    )
    row["measurement_circuit_preparation_time"] = 0.0
    row["measurement_time"] = rounded_seconds(before_measure + after_measure)
    row["decoding_time"] = rounded_seconds(before_decoding + after_decoding)


def run_one_configuration(
    dataset_path: Path,
    dataset_category: str,
    encoding: str,
    repetition: int,
    args: argparse.Namespace,
    dataset_size: int | None = None,
) -> dict[str, Any]:
    row = empty_row(
        dataset_path=dataset_path,
        dataset_category=dataset_category,
        encoding=encoding,
        repetition=repetition,
        shots=args.shots,
        dataset_size=dataset_size,
    )
    apply_runtime_settings(row, args)
    run_start = time.perf_counter()

    try:
        try:
            triples, parse_elapsed = timed_call(
                load_triples,
                file_path=dataset_path,
                rdf_format=args.rdf_format,
            )
        except Exception as exc:
            row["success"] = False
            row["status"] = "parse_error"
            row["error_message"] = f"{type(exc).__name__}: {exc}"
            return row

        original_num_triples = len(triples)
        max_real_triples = getattr(args, "max_real_triples", None)
        row["original_num_triples"] = original_num_triples
        row["max_real_triples"] = max_real_triples
        row["truncation_applied"] = False
        if (
            dataset_category == "real"
            and max_real_triples is not None
            and original_num_triples > max_real_triples
        ):
            triples = sorted(
                triples,
                key=lambda triple: (
                    triple.subject,
                    triple.predicate,
                    triple.object,
                ),
            )[:max_real_triples]
            row["truncation_applied"] = True

        context, mapping_elapsed = timed_call(
            build_encoding_context,
            triples=triples,
            fixed_thesis_mapping=False,
            dataset_path=dataset_path,
        )
        row["rdf_parse_time"] = rounded_seconds(parse_elapsed)
        row["id_mapping_time"] = rounded_seconds(mapping_elapsed)
        update_context_metrics(row, context)

        if encoding == "basis":
            run_basis_scaling(row, context, args)
        elif encoding == "amplitude":
            run_amplitude_scaling(row, context, args)
        elif encoding == "phase":
            run_phase_scaling(row, context, args)
        else:
            raise ValueError(f"Unsupported encoding '{encoding}'.")

        row["success"] = True
        row["status"] = "success"

    except SimulatorLimitError as exc:
        row["success"] = False
        row["status"] = "simulator_limit"
        row["error_message"] = str(exc)
    except Exception as exc:
        row["success"] = False
        row["status"] = "error"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    finally:
        total_elapsed = time.perf_counter() - run_start
        row["total_runtime"] = rounded_seconds(total_elapsed)
        if row["success"]:
            row["completed_total_runtime"] = rounded_seconds(total_elapsed)
        row["finished_at_utc"] = utc_timestamp()

    return row


def _run_one_configuration_worker(
    output_queue,
    dataset_path: Path,
    dataset_category: str,
    encoding: str,
    repetition: int,
    args: argparse.Namespace,
    dataset_size: int | None,
) -> None:
    row = run_one_configuration(
        dataset_path=dataset_path,
        dataset_category=dataset_category,
        encoding=encoding,
        repetition=repetition,
        args=args,
        dataset_size=dataset_size,
    )
    output_queue.put(row)


def timeout_row(
    dataset_path: Path,
    dataset_category: str,
    encoding: str,
    repetition: int,
    args: argparse.Namespace,
    dataset_size: int | None,
    elapsed: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    row = empty_row(
        dataset_path=dataset_path,
        dataset_category=dataset_category,
        encoding=encoding,
        repetition=repetition,
        shots=args.shots,
        dataset_size=dataset_size,
    )
    apply_runtime_settings(row, args, timeout_seconds=timeout_seconds)
    row.update(
        {
            "status": "timeout",
            "error_message": (
                f"Run exceeded timeout of {timeout_seconds} seconds and was terminated."
            ),
            "total_runtime": rounded_seconds(elapsed),
            "finished_at_utc": utc_timestamp(),
        }
    )
    return row


def process_failure_row(
    dataset_path: Path,
    dataset_category: str,
    encoding: str,
    repetition: int,
    args: argparse.Namespace,
    dataset_size: int | None,
    elapsed: float,
    message: str,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    row = empty_row(
        dataset_path=dataset_path,
        dataset_category=dataset_category,
        encoding=encoding,
        repetition=repetition,
        shots=args.shots,
        dataset_size=dataset_size,
    )
    apply_runtime_settings(row, args, timeout_seconds=timeout_seconds)
    row.update(
        {
            "status": "error",
            "error_message": message,
            "total_runtime": rounded_seconds(elapsed),
            "finished_at_utc": utc_timestamp(),
        }
    )
    return row


def run_one_configuration_with_timeout(
    dataset_path: Path,
    dataset_category: str,
    encoding: str,
    repetition: int,
    args: argparse.Namespace,
    dataset_size: int | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if timeout_seconds is None or timeout_seconds <= 0:
        row = run_one_configuration(
            dataset_path=dataset_path,
            dataset_category=dataset_category,
            encoding=encoding,
            repetition=repetition,
            args=args,
            dataset_size=dataset_size,
        )
        apply_runtime_settings(row, args, timeout_seconds=timeout_seconds)
        return row

    context = mp.get_context("spawn")
    output_queue = context.Queue()
    start_time = time.perf_counter()
    process = context.Process(
        target=_run_one_configuration_worker,
        args=(
            output_queue,
            dataset_path,
            dataset_category,
            encoding,
            repetition,
            args,
            dataset_size,
        ),
    )
    process.start()
    process.join(timeout_seconds)
    elapsed = time.perf_counter() - start_time

    if process.is_alive():
        process.terminate()
        process.join()
        return timeout_row(
            dataset_path=dataset_path,
            dataset_category=dataset_category,
            encoding=encoding,
            repetition=repetition,
            args=args,
            dataset_size=dataset_size,
            elapsed=elapsed,
            timeout_seconds=timeout_seconds,
        )

    try:
        row = output_queue.get(timeout=5)
        apply_runtime_settings(row, args, timeout_seconds=timeout_seconds)
        return row
    except queue.Empty:
        return process_failure_row(
            dataset_path=dataset_path,
            dataset_category=dataset_category,
            encoding=encoding,
            repetition=repetition,
            args=args,
            dataset_size=dataset_size,
            elapsed=elapsed,
            message=f"Worker exited with code {process.exitcode} without returning a result.",
            timeout_seconds=timeout_seconds,
        )


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def load_raw_results(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        return []

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {field: row.get(field, "") for field in RAW_FIELDNAMES}
            for row in reader
        ]


def write_raw_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in RAW_FIELDNAMES})


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def key_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def keyed_setting(
    row: dict[str, Any],
    field: str,
    legacy_default: str,
) -> str:
    value = key_value(row.get(field))
    return value if value else legacy_default


def phase_mode_key_for_category(category: str, phase_predicate: Any, phase_mode: Any) -> str:
    mode = key_value(phase_mode)
    if mode:
        return mode
    if key_value(phase_predicate):
        return "custom"
    return "synthetic-default" if category == "synthetic" else "most-common-predicate"


def raw_row_key(row: dict[str, Any]) -> tuple[str, ...]:
    dataset_category = key_value(row.get("dataset_category"))
    return (
        key_value(row.get("dataset_name")),
        dataset_category,
        key_value(row.get("encoding")),
        key_value(row.get("repetition")),
        key_value(row.get("shots")),
        keyed_setting(row, "max_basis_simulation_qubits", "22"),
        keyed_setting(row, "max_phase_diagonal_qubits", "10"),
        keyed_setting(row, "max_metric_qubits", "14"),
        keyed_setting(row, "weight_strategy", "uniform"),
        phase_mode_key_for_category(
            dataset_category,
            row.get("phase_mark_predicate"),
            row.get("phase_marker_mode"),
        ),
        key_value(row.get("phase_mark_predicate")),
        keyed_setting(row, "phase_angle", str(math.pi)),
        keyed_setting(row, "decompose_reps", "1"),
        keyed_setting(row, "compute_decomposed_metrics", "True"),
        keyed_setting(row, "compute_transpiled_metrics", "True"),
        key_value(row.get("rdf_format")),
        key_value(row.get("timeout_seconds")),
        key_value(row.get("max_real_triples")),
    )


def configuration_key(
    dataset_spec: DatasetSpec,
    encoding: str,
    repetition: int,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    return (
        dataset_spec.path.stem,
        dataset_spec.category,
        encoding,
        str(repetition),
        key_value(getattr(args, "shots", "")),
        key_value(getattr(args, "max_basis_simulation_qubits", "")),
        key_value(getattr(args, "max_phase_diagonal_qubits", "")),
        key_value(getattr(args, "max_metric_qubits", "")),
        key_value(getattr(args, "weight_strategy", "")),
        phase_mode_key_for_category(
            dataset_spec.category,
            getattr(args, "phase_mark_predicate", ""),
            getattr(args, "phase_marker_mode", ""),
        ),
        key_value(getattr(args, "phase_mark_predicate", "")),
        key_value(getattr(args, "phase_angle", "")),
        key_value(getattr(args, "decompose_reps", "")),
        key_value(getattr(args, "compute_decomposed_metrics", "")),
        key_value(getattr(args, "compute_transpiled_metrics", "")),
        key_value(getattr(args, "rdf_format", "")),
        key_value(getattr(args, "timeout_seconds", "")),
        key_value(getattr(args, "max_real_triples", "")),
    )


def merge_rows_without_duplicates(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for row in [*existing_rows, *new_rows]:
        row_key = raw_row_key(row)
        if row_key in seen_keys:
            continue
        seen_keys.add(row_key)
        merged_rows.append(row)
    return merged_rows


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_blank(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 9)


def std_or_blank(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return round(statistics.stdev(values), 9)


def values_for(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = to_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def values_for_any(rows: list[dict[str, Any]], *fields: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        for field in fields:
            value = to_float(row.get(field))
            if value is not None:
                values.append(value)
                break
    return values


def first_nonempty(rows: list[dict[str, Any]], field: str) -> Any:
    for row in rows:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return ""


def sorted_statuses(rows: list[dict[str, Any]]) -> str:
    return ",".join(sorted({str(row.get("status")) for row in rows if row.get("status")}))


def summary_sort_key(row: dict[str, Any]) -> tuple[str, float, str, str]:
    num_triples = to_float(row.get("num_triples")) or to_float(row.get("dataset_size")) or 0
    return (
        str(row.get("dataset_category") or ""),
        num_triples,
        str(row.get("dataset_name") or ""),
        str(row.get("encoding") or ""),
    )


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("dataset_category") or ""),
                str(row.get("dataset_name") or ""),
                str(row.get("encoding") or ""),
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (_category, _dataset_name, _encoding), group_rows in grouped.items():
        completed_totals = [
            value
            for row in group_rows
            if truthy(row.get("success"))
            for value in [to_float(row.get("completed_total_runtime"))]
            if value is not None
        ]
        summary_rows.append(
            {
                "dataset_category": first_nonempty(group_rows, "dataset_category"),
                "dataset_name": first_nonempty(group_rows, "dataset_name"),
                "dataset_path": first_nonempty(group_rows, "dataset_path"),
                "dataset_size": first_nonempty(group_rows, "dataset_size"),
                "num_triples": first_nonempty(group_rows, "num_triples"),
                "encoding": first_nonempty(group_rows, "encoding"),
                "run_count": len(group_rows),
                "success_count": sum(1 for row in group_rows if truthy(row.get("success"))),
                "failure_count": sum(
                    1 for row in group_rows if not truthy(row.get("success"))
                ),
                "statuses": sorted_statuses(group_rows),
                "mean_total_runtime": mean_or_blank(values_for(group_rows, "total_runtime")),
                "std_total_runtime": std_or_blank(values_for(group_rows, "total_runtime")),
                "mean_completed_total_runtime": mean_or_blank(completed_totals),
                "std_completed_total_runtime": std_or_blank(completed_totals),
                "mean_rdf_parse_time": mean_or_blank(
                    values_for(group_rows, "rdf_parse_time")
                ),
                "mean_id_mapping_time": mean_or_blank(
                    values_for(group_rows, "id_mapping_time")
                ),
                "mean_state_preparation_time": mean_or_blank(
                    values_for(group_rows, "state_preparation_time")
                ),
                "mean_circuit_construction_time": mean_or_blank(
                    values_for(group_rows, "circuit_construction_time")
                ),
                "mean_simulation_time": mean_or_blank(
                    values_for(group_rows, "simulation_time")
                ),
                "mean_measurement_time": mean_or_blank(
                    values_for(group_rows, "measurement_time")
                ),
                "mean_qubits": mean_or_blank(values_for(group_rows, "qubits")),
                "mean_circuit_depth": mean_or_blank(
                    values_for_any(group_rows, "logical_circuit_depth", "circuit_depth")
                ),
                "mean_gate_count": mean_or_blank(
                    values_for_any(group_rows, "logical_gate_count", "gate_count")
                ),
                "mean_logical_circuit_depth": mean_or_blank(
                    values_for_any(group_rows, "logical_circuit_depth", "circuit_depth")
                ),
                "mean_logical_gate_count": mean_or_blank(
                    values_for_any(group_rows, "logical_gate_count", "gate_count")
                ),
                "mean_decomposed_circuit_depth": mean_or_blank(
                    values_for(group_rows, "decomposed_circuit_depth")
                ),
                "mean_decomposed_gate_count": mean_or_blank(
                    values_for(group_rows, "decomposed_gate_count")
                ),
                "mean_transpiled_circuit_depth": mean_or_blank(
                    values_for(group_rows, "transpiled_circuit_depth")
                ),
                "mean_transpiled_gate_count": mean_or_blank(
                    values_for(group_rows, "transpiled_gate_count")
                ),
                "mean_phase_marked_triples": mean_or_blank(
                    values_for(group_rows, "phase_marked_triples")
                ),
                "mean_phase_marked_fraction": mean_or_blank(
                    values_for(group_rows, "phase_marked_fraction")
                ),
                "shots": first_nonempty(group_rows, "shots"),
                "backend": first_nonempty(group_rows, "backend"),
            }
        )

    return sorted(summary_rows, key=summary_sort_key)


def write_summary(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {field: csv_value(row.get(field)) for field in SUMMARY_FIELDNAMES}
            )


def numeric_x_value(row: dict[str, Any]) -> float | None:
    return to_float(row.get("num_triples")) or to_float(row.get("dataset_size"))


def pretty_label(value: str) -> str:
    label = value.replace("_", " ").replace("-", " ").title()
    return (
        label.replace(" Kg ", " KG ")
        .replace("Kg ", "KG ")
        .replace(" Rdf ", " RDF ")
        .replace(" Id ", " ID ")
    )


def compact_number(value: float, _position: int | None = None) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2g}"
    return f"{value:.2g}"


def metric_title(group: str, filename_stem: str) -> str:
    metric_name = pretty_label(filename_stem)
    if group == "real":
        return f"Real KG {metric_name} By Dataset"
    if group == "combined":
        return f"Combined {metric_name} By Triple Count"
    return f"Synthetic {metric_name} By Triple Count"


def metric_stem_from_plot_path(output_path: Path, group: str) -> str:
    stem = output_path.stem
    prefix = f"{group}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    for suffix in ("_vs_triples", "_by_dataset"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def log_spaced_ticks(values: list[float], min_log_distance: float = 0.18) -> list[float]:
    spaced_ticks: list[float] = []
    for value in sorted(values):
        if value <= 0:
            continue
        if not spaced_ticks:
            spaced_ticks.append(value)
            continue
        if math.log10(value) - math.log10(spaced_ticks[-1]) >= min_log_distance:
            spaced_ticks.append(value)
    return spaced_ticks


def use_log_y_scale(values: list[float], metric_field: str) -> bool:
    if metric_field == "mean_qubits":
        return False
    positive_values = [value for value in values if value > 0]
    if len(positive_values) < 2:
        return False
    return max(positive_values) / min(positive_values) >= 25


def apply_plot_style(
    fig,
    ax,
    *,
    title: str,
    ylabel: str,
    y_values: list[float],
    metric_field: str,
) -> None:
    fig.patch.set_facecolor(PLOT_BACKGROUND)
    ax.set_facecolor(AXIS_BACKGROUND)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)

    if use_log_y_scale(y_values, metric_field):
        ax.set_yscale("log")
        ax.set_ylabel(f"{ylabel} (log scale)")
    else:
        ax.set_ylabel(ylabel)
        lower, _upper = ax.get_ylim()
        if lower >= 0:
            ax.set_ylim(bottom=0)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(compact_number))
    ax.grid(True, which="major", axis="both", color=GRID_COLOR, alpha=0.72, linewidth=0.8)
    ax.grid(True, which="minor", axis="y", color=GRID_COLOR, alpha=0.25, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#1f2937", labelsize=10)
    ax.margins(y=0.14)


def plot_vs_triples(
    summary_rows: list[dict[str, Any]],
    metric_field: str,
    ylabel: str,
    output_path: Path,
    *,
    combined: bool = False,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    plotted_any = False
    categories = DATASET_CATEGORIES if combined else ("synthetic",)
    plotted_values: list[float] = []

    for category in categories:
        for encoding in ENCODINGS:
            selected_rows = [
                row
                for row in summary_rows
                if row.get("encoding") == encoding
                and (combined or row.get("dataset_category") == category)
                and (not combined or row.get("dataset_category") == category)
            ]
            points: list[tuple[float, float]] = []
            for row in selected_rows:
                x_value = numeric_x_value(row)
                y_value = to_float(row.get(metric_field))
                if x_value is not None and y_value is not None:
                    points.append((x_value, y_value))
            points.sort(key=lambda item: item[0])
            if not points:
                continue

            plotted_any = True
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            plotted_values.extend(y_values)
            label = (
                f"{pretty_label(encoding)} ({pretty_label(category)})"
                if combined
                else pretty_label(encoding)
            )
            ax.plot(
                x_values,
                y_values,
                marker=CATEGORY_MARKERS.get(category, ENCODING_MARKERS.get(encoding, "o"))
                if combined
                else ENCODING_MARKERS.get(encoding, "o"),
                markersize=7.8,
                markeredgecolor="white",
                markeredgewidth=1.2,
                color=ENCODING_COLORS.get(encoding),
                linestyle=CATEGORY_LINESTYLES.get(category, "-"),
                linewidth=2.4,
                label=label,
            )

    if not plotted_any:
        plt.close(fig)
        return False

    x_ticks = sorted(
        {
            numeric_x_value(row)
            for row in summary_rows
            if numeric_x_value(row) is not None
        }
    )
    if combined and len(x_ticks) > 4:
        x_ticks = log_spaced_ticks(x_ticks)
    ax.set_xscale("log", base=10)
    ax.set_xticks(x_ticks)
    if len(x_ticks) == 1:
        only_tick = x_ticks[0]
        ax.set_xlim(only_tick / 1.6, only_tick * 1.6)
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(compact_number))
    ax.set_xlabel("Number of triples")
    apply_plot_style(
        fig,
        ax,
        title=metric_title(
            "combined" if combined else "synthetic",
            metric_stem_from_plot_path(
                output_path,
                "combined" if combined else "synthetic",
            ),
        ),
        ylabel=ylabel,
        y_values=plotted_values,
        metric_field=metric_field,
    )
    ax.legend(
        title="Encoding / Dataset" if combined else "Encoding",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def plot_real_by_dataset(
    summary_rows: list[dict[str, Any]],
    metric_field: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    real_rows = [row for row in summary_rows if row.get("dataset_category") == "real"]
    dataset_names = sorted(
        {str(row.get("dataset_name")) for row in real_rows if row.get("dataset_name")},
        key=lambda name: (
            min(
                [
                    numeric_x_value(row) or 0
                    for row in real_rows
                    if row.get("dataset_name") == name
                ]
                or [0]
            ),
            name,
        ),
    )
    if not dataset_names:
        return False

    fig_width = max(9.4, 1.45 * len(dataset_names) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    plotted_any = False
    x_positions = np.arange(len(dataset_names), dtype=float)
    bar_width = min(0.24, 0.78 / max(len(ENCODINGS), 1))
    plotted_values: list[float] = []

    for encoding_index, encoding in enumerate(ENCODINGS):
        y_values: list[float | None] = []
        for dataset_name in dataset_names:
            matching_rows = [
                row
                for row in real_rows
                if row.get("dataset_name") == dataset_name
                and row.get("encoding") == encoding
            ]
            y_values.append(
                to_float(matching_rows[0].get(metric_field)) if matching_rows else None
            )

        offset = (encoding_index - ((len(ENCODINGS) - 1) / 2)) * bar_width
        present_bars = [
            (float(x_position + offset), y_value)
            for x_position, y_value in zip(x_positions, y_values)
            if y_value is not None
        ]
        if not present_bars:
            continue

        plotted_any = True
        plotted_values.extend([bar[1] for bar in present_bars])
        ax.bar(
            [bar[0] for bar in present_bars],
            [bar[1] for bar in present_bars],
            width=bar_width,
            color=ENCODING_COLORS.get(encoding),
            edgecolor="white",
            linewidth=1.0,
            label=pretty_label(encoding),
            zorder=3,
        )

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xticks(x_positions)
    ax.set_xticklabels(dataset_names, rotation=22, ha="right")
    ax.set_xlabel("Real KG dataset")
    apply_plot_style(
        fig,
        ax,
        title=metric_title("real", metric_stem_from_plot_path(output_path, "real")),
        ylabel=ylabel,
        y_values=plotted_values,
        metric_field=metric_field,
    )
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.72, linewidth=0.8)
    ax.legend(
        title="Encoding",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        borderaxespad=0,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def generate_group_plots(
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
    group: str,
) -> list[Path]:
    plot_dir = output_dir / "plots"
    generated: list[Path] = []
    if plot_dir.exists():
        for stale_plot in plot_dir.glob(f"{group}_*.png"):
            stale_plot.unlink()

    for metric_field, ylabel, filename_stem in PLOT_METRICS:
        if group == "real":
            filename = f"real_{filename_stem}_by_dataset.png"
            output_path = plot_dir / filename
            created = plot_real_by_dataset(
                summary_rows=summary_rows,
                metric_field=metric_field,
                ylabel=ylabel,
                output_path=output_path,
            )
        else:
            filename = f"{group}_{filename_stem}_vs_triples.png"
            output_path = plot_dir / filename
            created = plot_vs_triples(
                summary_rows=summary_rows,
                metric_field=metric_field,
                ylabel=ylabel,
                output_path=output_path,
                combined=group == "combined",
            )
        if created:
            generated.append(output_path)

    return generated


def write_experiment_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    group: str,
) -> tuple[Path, Path, list[Path]]:
    raw_path = output_dir / "scaling_raw_results.csv"
    write_raw_results(rows, raw_path)
    summary_path, plots = write_summary_and_plots(rows, output_dir, group)
    return raw_path, summary_path, plots


def write_summary_and_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    group: str,
) -> tuple[Path, list[Path]]:
    summary_path = output_dir / "scaling_summary.csv"
    summary_rows = build_summary(rows)
    write_summary(summary_rows, summary_path)
    plots = generate_group_plots(summary_rows, output_dir, group)
    return summary_path, plots


def update_experiment_outputs(
    dataset_specs: list[DatasetSpec],
    encodings: list[str],
    repetitions: int,
    args: argparse.Namespace,
    output_dir: Path,
    group: str,
    *,
    timeout_seconds: float | None = None,
    append: bool = False,
    regenerate_plots_only: bool = False,
) -> tuple[list[dict[str, Any]], Path, Path, list[Path]]:
    raw_path = output_dir / "scaling_raw_results.csv"
    existing_rows = load_raw_results(raw_path)

    if regenerate_plots_only:
        if not existing_rows:
            raise FileNotFoundError(
                f"Cannot regenerate plots because {raw_path} does not exist or is empty."
            )
        summary_path, plots = write_summary_and_plots(existing_rows, output_dir, group)
        return existing_rows, raw_path, summary_path, plots

    new_rows = run_experiment_rows(
        dataset_specs=dataset_specs,
        encodings=encodings,
        repetitions=repetitions,
        args=args,
        timeout_seconds=timeout_seconds,
        existing_rows=existing_rows,
        skip_existing=append,
    )
    all_rows = (
        merge_rows_without_duplicates(existing_rows, new_rows)
        if append
        else new_rows
    )
    raw_path, summary_path, plots = write_experiment_outputs(
        rows=all_rows,
        output_dir=output_dir,
        group=group,
    )
    return all_rows, raw_path, summary_path, plots


def ensure_synthetic_dataset_file(
    data_dir: Path,
    dataset_size: int,
    generate_missing: bool,
) -> Path:
    dataset_path = data_dir / f"synthetic_{dataset_size}.ttl"
    if dataset_path.exists():
        return dataset_path

    if not generate_missing:
        raise FileNotFoundError(
            f"Missing {dataset_path}. Run scripts/generate_scaling_datasets.py first."
        )

    if dataset_size not in DATASET_CONFIGS:
        raise ValueError(f"No synthetic dataset configuration exists for {dataset_size}.")

    triple_count, entity_count, predicate_count = DATASET_CONFIGS[dataset_size]
    return generate_dataset(
        output_path=dataset_path,
        triple_count=triple_count,
        entity_count=entity_count,
        predicate_count=predicate_count,
    )


def synthetic_dataset_specs(
    sizes: list[int],
    data_dir: Path = Path("data/scaling"),
    generate_missing: bool = True,
) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            category="synthetic",
            path=ensure_synthetic_dataset_file(
                data_dir=data_dir,
                dataset_size=size,
                generate_missing=generate_missing,
            ),
            size=size,
        )
        for size in sizes
    ]


def real_dataset_specs(
    real_dir: Path = DEFAULT_REAL_KG_DIR,
    filenames: list[str] | None = None,
) -> list[DatasetSpec]:
    selected_filenames = filenames if filenames is not None else list(REAL_KG_FILES)
    return [
        DatasetSpec(category="real", path=real_dir / filename, size=None)
        for filename in selected_filenames
    ]


def run_experiment_rows(
    dataset_specs: list[DatasetSpec],
    encodings: list[str],
    repetitions: int,
    args: argparse.Namespace,
    timeout_seconds: float | None = None,
    existing_rows: list[dict[str, Any]] | None = None,
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1.")

    rows: list[dict[str, Any]] = []
    existing_keys = {
        raw_row_key(row)
        for row in (existing_rows or [])
    }
    for dataset_spec in dataset_specs:
        for encoding in encodings:
            for repetition in range(1, repetitions + 1):
                expected_key = configuration_key(
                    dataset_spec=dataset_spec,
                    encoding=encoding,
                    repetition=repetition,
                    args=args,
                )
                if skip_existing and expected_key in existing_keys:
                    print(
                        f"Skipping existing {dataset_spec.category} | "
                        f"{dataset_spec.path.name} | {encoding} | "
                        f"repetition {repetition}/{repetitions}"
                    )
                    continue
                print(
                    f"Running {dataset_spec.category} | {dataset_spec.path.name} | "
                    f"{encoding} | repetition {repetition}/{repetitions}"
                )
                row = run_one_configuration_with_timeout(
                    dataset_path=dataset_spec.path,
                    dataset_category=dataset_spec.category,
                    encoding=encoding,
                    repetition=repetition,
                    args=args,
                    dataset_size=dataset_spec.size,
                    timeout_seconds=timeout_seconds,
                )
                print(f"  {row['status']}: {row.get('error_message') or 'ok'}")
                rows.append(row)
                existing_keys.add(raw_row_key(row))

    return rows


def build_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    phase_mark_predicate = getattr(args, "phase_mark_predicate", None)
    if phase_mark_predicate == "":
        phase_mark_predicate = None
    phase_marker_mode = getattr(args, "phase_marker_mode", None)
    if phase_marker_mode == "":
        phase_marker_mode = None
    return argparse.Namespace(
        shots=args.shots,
        rdf_format=args.rdf_format,
        weight_strategy=args.weight_strategy,
        phase_marker_mode=phase_marker_mode,
        phase_mark_predicate=phase_mark_predicate,
        phase_angle=args.phase_angle,
        max_basis_simulation_qubits=args.max_basis_simulation_qubits,
        max_phase_diagonal_qubits=args.max_phase_diagonal_qubits,
        max_metric_qubits=args.max_metric_qubits,
        decompose_reps=args.decompose_reps,
        compute_decomposed_metrics=args.compute_decomposed_metrics,
        compute_transpiled_metrics=args.compute_transpiled_metrics,
        timeout_seconds=getattr(args, "timeout_seconds", None),
        max_real_triples=getattr(args, "max_real_triples", None),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled synthetic KG quantum-encoding scaling experiments."
    )
    parser.add_argument(
        "--data-dir",
        default="data/scaling",
        help="Directory containing synthetic_<size>.ttl files.",
    )
    parser.add_argument(
        "--output-dir",
        "--results-dir",
        dest="output_dir",
        default="results/scaling/synthetic",
        help="Directory where synthetic scaling CSVs and plots are written.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 1000, 5000],
        help="Synthetic dataset sizes to run.",
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        choices=ENCODINGS,
        default=list(ENCODINGS),
        help="Encoding families to run.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions per dataset/encoding configuration.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=2048,
        help="Number of measurement shots.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional per-run timeout. Timed-out runs are recorded and the batch continues.",
    )
    parser.add_argument(
        "--max-real-triples",
        type=int,
        default=None,
        help=(
            "Deterministically truncate real KG files to this many triples after "
            "parsing. Synthetic datasets are unaffected."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new rows to an existing raw CSV, skipping already-recorded configurations.",
    )
    parser.add_argument(
        "--regenerate-plots-only",
        action="store_true",
        help="Do not run experiments; rebuild summary and plots from the existing raw CSV.",
    )
    parser.add_argument(
        "--rdf-format",
        default=None,
        help="Optional rdflib parser format override.",
    )
    parser.add_argument(
        "--weight-strategy",
        choices=("uniform", "linear"),
        default="uniform",
        help="Amplitude weight strategy used by the scaling experiment.",
    )
    parser.add_argument(
        "--phase-marker-mode",
        choices=PHASE_MARKER_MODES,
        default=None,
        help=(
            "Phase predicate selection mode. If omitted, synthetic datasets use "
            "synthetic-default and real datasets use most-common-predicate."
        ),
    )
    parser.add_argument(
        "--phase-mark-predicate",
        default=None,
        help=(
            "Custom predicate URI marked in phase encoding. Supplying this value "
            "uses custom phase selection unless --phase-marker-mode overrides it."
        ),
    )
    parser.add_argument(
        "--phase-angle",
        type=float,
        default=math.pi,
        help="Phase angle used for marked triples.",
    )
    parser.add_argument(
        "--max-basis-simulation-qubits",
        type=int,
        default=22,
        help="Guardrail for dense basis statevector simulation.",
    )
    parser.add_argument(
        "--max-phase-diagonal-qubits",
        type=int,
        default=10,
        help="Guardrail for the dense phase-oracle matrix.",
    )
    parser.add_argument(
        "--max-metric-qubits",
        type=int,
        default=14,
        help=(
            "Guardrail for decomposed/transpiled circuit metrics. Logical metrics "
            "are still recorded above this limit."
        ),
    )
    parser.add_argument(
        "--decompose-reps",
        type=int,
        default=1,
        help="Number of Qiskit decompose() repetitions for decomposed metrics.",
    )
    parser.add_argument(
        "--compute-decomposed-metrics",
        dest="compute_decomposed_metrics",
        action="store_true",
        default=True,
        help="Compute guarded decomposed circuit metrics.",
    )
    parser.add_argument(
        "--no-compute-decomposed-metrics",
        dest="compute_decomposed_metrics",
        action="store_false",
        help="Skip decomposed circuit metrics.",
    )
    parser.add_argument(
        "--compute-transpiled-metrics",
        dest="compute_transpiled_metrics",
        action="store_true",
        default=True,
        help="Compute guarded transpiled circuit metrics.",
    )
    parser.add_argument(
        "--no-compute-transpiled-metrics",
        dest="compute_transpiled_metrics",
        action="store_false",
        help="Skip transpiled circuit metrics.",
    )
    parser.add_argument(
        "--no-generate-missing",
        action="store_true",
        help="Do not generate missing synthetic datasets automatically.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    runtime_args = build_runtime_args(args)
    dataset_specs = []
    if not args.regenerate_plots_only:
        dataset_specs = synthetic_dataset_specs(
            sizes=args.sizes,
            data_dir=Path(args.data_dir),
            generate_missing=not args.no_generate_missing,
        )

    _rows, raw_path, summary_path, plots = update_experiment_outputs(
        dataset_specs=dataset_specs,
        encodings=list(args.encodings),
        repetitions=args.repetitions,
        args=runtime_args,
        timeout_seconds=args.timeout_seconds,
        output_dir=Path(args.output_dir),
        group="synthetic",
        append=args.append,
        regenerate_plots_only=args.regenerate_plots_only,
    )

    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    for path in plots:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
