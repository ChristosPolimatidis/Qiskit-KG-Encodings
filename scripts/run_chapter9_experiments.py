from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.amplitude_encoding import (
    build_amplitude_encoding_artifacts,
    build_sparse_index_amplitude_encoding,
    prepare_amplitude_state_from_normalized_vector,
)
from src.combined_encoding import combined_amplitude_phase_encoding
from src.paper_phase_encoding import predicate_phase, presence_phase
from src.running_example import (
    PAPER_INDEX_DIMENSION,
    SEQUENTIAL_INDEX_DIMENSION,
    get_running_example_indices,
    get_running_example_triples,
)
from src.tasks.amplitude_similarity import run_amplitude_similarity_task
from src.tasks.basis_lookup import run_basis_lookup_task
from src.tasks.combined_demo import run_combined_demo_task
from src.tasks.link_prediction_distance import run_link_prediction_distance_task
from src.tasks.multihop_phase_kickback import run_multihop_phase_kickback_task
from src.tasks.schema_matching_qft import run_schema_matching_qft_task


ENCODING_TABLE_FIELDS = [
    "Encoding",
    "Variant",
    "Index Mode",
    "Qubits",
    "Dimension",
    "Time to Create (ms)",
    "Circuit Depth",
    "Gate Count",
    "Transpiled Depth",
    "Transpiled Gate Count",
    "Notes",
]

ENCODING_LATEX_FIELDS = [
    "Encoding",
    "Variant",
    "Index Mode",
    "Qubits",
    "Time to Create (ms)",
    "Circuit Depth",
    "Notes",
]

ENCODING_LABELS = {
    "basis": ("Basis", "Index support"),
    "amplitude_baseline": ("Amplitude", "Compact baseline"),
    "amplitude_paper_aligned": ("Amplitude", "Paper-aligned index"),
    "phase_presence": ("Phase", "Presence phase"),
    "phase_predicate": ("Phase", "Predicate phase"),
    "combined_amplitude_phase": ("Combined", "Amplitude + phase"),
}

USAGE_TABLE_FIELDS = [
    "KG Task",
    "Encoding",
    "Quantum Method",
    "Main Result",
    "Time",
]

USAGE_LATEX_FIELDS = USAGE_TABLE_FIELDS

TABLE6_CIRCUIT_FIELDS = [
    "Task",
    "Encoding",
    "Method",
    "Qubits",
    "Circuit Depth",
    "Gate Count",
    "Transpiled Depth",
    "Transpiled Gate Count",
    "Shots",
    "Repetitions",
]

TABLE1_TASK_ORDER = [
    "search_grover_lookup",
    "entity_matching_swap_test",
    "link_prediction_distance_estimation",
    "multihop_phase_kickback",
    "schema_matching_qft",
]

TABLE1_TASK_LABELS = {
    "search_grover_lookup": ("Search", "Basis", "Grover lookup"),
    "entity_matching_swap_test": ("Entity Matching", "Amplitude", "Swap Test"),
    "link_prediction_distance_estimation": (
        "Link Prediction",
        "Amplitude",
        "Distance Estimation",
    ),
    "multihop_phase_kickback": (
        "Multi-hop Reasoning",
        "Phase",
        "Phase Kickback",
    ),
    "schema_matching_qft": ("Schema Matching", "Phase", "QFT"),
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    commit_hash = completed.stdout.strip()
    return commit_hash or None


def cpu_info() -> dict[str, Any]:
    processor = (
        platform.processor()
        or platform.uname().processor
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.machine()
        or None
    )
    return {
        "processor": processor,
        "machine": platform.machine() or None,
        "architecture": platform.architecture()[0],
        "logical_cpus": os.cpu_count(),
    }


def total_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        pass

    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = MemoryStatus()
            memory_status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
                return int(memory_status.ullTotalPhys)
        except Exception:
            return None

    if sys.platform.startswith("linux"):
        try:
            with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        return int(parts[1]) * 1024
        except Exception:
            return None

    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                return int(completed.stdout.strip())
        except Exception:
            return None

    return None


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 2)


def format_ram(value: int | None) -> str:
    gib = bytes_to_gib(value)
    return f"{gib} GiB" if gib is not None else "unknown RAM"


def operation_count(circuit: QuantumCircuit | None) -> int | None:
    if circuit is None:
        return None
    return sum(int(count) for count in circuit.count_ops().values())


def circuit_depth(circuit: QuantumCircuit | None) -> int | None:
    return circuit.depth() if circuit is not None else None


def transpiled_metrics(
    circuit: QuantumCircuit | None,
    simulator: AerSimulator,
) -> tuple[int | None, int | None]:
    if circuit is None:
        return None, None
    try:
        transpiled_circuit = transpile(circuit, simulator, optimization_level=0)
    except Exception:
        return None, None
    return transpiled_circuit.depth(), operation_count(transpiled_circuit)


def state_norm(statevector: np.ndarray) -> float:
    return float(np.linalg.norm(statevector))


def dimension_for_index_mode(index_mode: str) -> int:
    if index_mode == "sequential":
        return SEQUENTIAL_INDEX_DIMENSION
    if index_mode == "paper":
        return PAPER_INDEX_DIMENSION
    raise ValueError("Unsupported index mode.")


def initialize_sparse_index_circuit(
    indices: list[int],
    dimension: int,
    name: str,
) -> tuple[np.ndarray, QuantumCircuit, float]:
    start = time.perf_counter()
    statevector = np.zeros(dimension, dtype=complex)
    amplitude = 1 / math.sqrt(len(indices))
    for index in indices:
        statevector[index] = amplitude
    num_qubits = int(math.log2(dimension))
    circuit = QuantumCircuit(num_qubits, name=name)
    circuit.initialize(statevector, circuit.qubits)
    preparation_time = time.perf_counter() - start
    return statevector, circuit, preparation_time


def process_row(
    *,
    experiment: str,
    repetition: int,
    index_mode: str,
    backend: str,
    status: str,
    task_time_seconds: float,
    preparation_time_seconds: float | None,
    num_qubits: int | None,
    dimension: int | None,
    norm: float | None,
    circuit: QuantumCircuit | None,
    simulator: AerSimulator,
    notes: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transpiled_depth, transpiled_gate_count = transpiled_metrics(circuit, simulator)
    row = {
        "group": "encoding_process",
        "experiment": experiment,
        "repetition": repetition,
        "index_mode": index_mode,
        "backend": backend,
        "status": status,
        "task_time_seconds": task_time_seconds,
        "preparation_time_seconds": preparation_time_seconds,
        "num_qubits": num_qubits,
        "dimension": dimension,
        "state_norm": norm,
        "circuit_depth": circuit_depth(circuit),
        "gate_count": operation_count(circuit),
        "transpiled_depth": transpiled_depth,
        "transpiled_gate_count": transpiled_gate_count,
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def run_basis_process(
    *,
    repetition: int,
    index_mode: str,
    backend: str,
    simulator: AerSimulator,
) -> dict[str, Any]:
    start = time.perf_counter()
    triple_to_index = get_running_example_indices(mode=index_mode)
    indices = sorted(triple_to_index.values())
    dimension = dimension_for_index_mode(index_mode)
    statevector, circuit, preparation_time = initialize_sparse_index_circuit(
        indices=indices,
        dimension=dimension,
        name="Chapter9BasisIndexEncoding",
    )
    return process_row(
        experiment="basis",
        repetition=repetition,
        index_mode=index_mode,
        backend=backend,
        status="success",
        task_time_seconds=time.perf_counter() - start,
        preparation_time_seconds=preparation_time,
        num_qubits=circuit.num_qubits,
        dimension=dimension,
        norm=state_norm(statevector),
        circuit=circuit,
        simulator=simulator,
        notes="Basis/index support state over the six running-example triples.",
        extra={"nonzero_indices": indices},
    )


def run_amplitude_baseline_process(
    *,
    repetition: int,
    backend: str,
    simulator: AerSimulator,
) -> dict[str, Any]:
    start = time.perf_counter()
    triples = get_running_example_triples()
    prep_start = time.perf_counter()
    artifacts = build_amplitude_encoding_artifacts(triples=triples, strategy="uniform")
    circuit = prepare_amplitude_state_from_normalized_vector(
        artifacts["normalized_vector"],
        name="Chapter9AmplitudeBaseline",
    )
    preparation_time = time.perf_counter() - prep_start
    statevector = artifacts["normalized_vector"]
    return process_row(
        experiment="amplitude_baseline",
        repetition=repetition,
        index_mode="compact",
        backend=backend,
        status="success",
        task_time_seconds=time.perf_counter() - start,
        preparation_time_seconds=preparation_time,
        num_qubits=artifacts["num_qubits"],
        dimension=len(statevector),
        norm=state_norm(statevector),
        circuit=circuit,
        simulator=simulator,
        notes="Compact amplitude baseline over contiguous triple positions 0..5.",
    )


def run_amplitude_paper_aligned_process(
    *,
    repetition: int,
    index_mode: str,
    backend: str,
    simulator: AerSimulator,
) -> dict[str, Any]:
    start = time.perf_counter()
    sparse_result = build_sparse_index_amplitude_encoding(
        triples=get_running_example_triples(),
        index_mode=index_mode,
    )
    circuit = prepare_amplitude_state_from_normalized_vector(
        sparse_result.statevector,
        name="Chapter9AmplitudePaperAligned",
    )
    return process_row(
        experiment="amplitude_paper_aligned",
        repetition=repetition,
        index_mode=index_mode,
        backend=backend,
        status="success",
        task_time_seconds=time.perf_counter() - start,
        preparation_time_seconds=sparse_result.preparation_time_seconds,
        num_qubits=sparse_result.num_qubits,
        dimension=sparse_result.dimension,
        norm=state_norm(sparse_result.statevector),
        circuit=circuit,
        simulator=simulator,
        notes="Amplitude support state using the selected Chapter 9 index mapping.",
        extra={"nonzero_indices": sparse_result.nonzero_indices},
    )


def run_phase_presence_process(
    *,
    repetition: int,
    index_mode: str,
    backend: str,
    simulator: AerSimulator,
    shots: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = presence_phase(index_mode=index_mode, shots=shots)
    return process_row(
        experiment="phase_presence",
        repetition=repetition,
        index_mode=index_mode,
        backend=backend,
        status="success",
        task_time_seconds=time.perf_counter() - start,
        preparation_time_seconds=result.preparation_time_seconds,
        num_qubits=result.num_qubits,
        dimension=len(result.statevector),
        norm=state_norm(result.statevector),
        circuit=result.circuit,
        simulator=simulator,
        notes="Presence phase assigns pi to existing triple indices and 0 elsewhere.",
        extra={"marked_indices": result.marked_indices},
    )


def run_phase_predicate_process(
    *,
    repetition: int,
    index_mode: str,
    backend: str,
    simulator: AerSimulator,
    shots: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    support_mode = "triples" if index_mode == "sequential" else "full"
    result = predicate_phase(index_mode=index_mode, support_mode=support_mode, shots=shots)
    return process_row(
        experiment="phase_predicate",
        repetition=repetition,
        index_mode=index_mode,
        backend=backend,
        status="success",
        task_time_seconds=time.perf_counter() - start,
        preparation_time_seconds=result.preparation_time_seconds,
        num_qubits=result.num_qubits,
        dimension=len(result.statevector),
        norm=state_norm(result.statevector),
        circuit=result.circuit,
        simulator=simulator,
        notes=f"Predicate phase over {support_mode} support.",
        extra={"marked_indices": result.marked_indices},
    )


def run_combined_process(
    *,
    repetition: int,
    index_mode: str,
    backend: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = combined_amplitude_phase_encoding(index_mode=index_mode)
    return process_row(
        experiment="combined_amplitude_phase",
        repetition=repetition,
        index_mode=index_mode,
        backend=backend,
        status="success",
        task_time_seconds=time.perf_counter() - start,
        preparation_time_seconds=result.preparation_time_seconds,
        num_qubits=result.num_qubits,
        dimension=result.dimension,
        norm=state_norm(result.statevector),
        circuit=None,
        simulator=AerSimulator(),
        notes="Combined alpha_k exp(i theta_k) state over existing triple indices.",
        extra={"nonzero_indices": result.nonzero_indices},
    )


def run_encoding_processes(args: argparse.Namespace) -> list[dict[str, Any]]:
    simulator = AerSimulator(seed_simulator=args.seed)
    rows: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        rows.extend(
            [
                run_basis_process(
                    repetition=repetition,
                    index_mode=args.index_mode,
                    backend=args.backend,
                    simulator=simulator,
                ),
                run_amplitude_baseline_process(
                    repetition=repetition,
                    backend=args.backend,
                    simulator=simulator,
                ),
                run_amplitude_paper_aligned_process(
                    repetition=repetition,
                    index_mode=args.index_mode,
                    backend=args.backend,
                    simulator=simulator,
                ),
                run_phase_presence_process(
                    repetition=repetition,
                    index_mode=args.index_mode,
                    backend=args.backend,
                    simulator=simulator,
                    shots=args.shots,
                ),
                run_phase_predicate_process(
                    repetition=repetition,
                    index_mode=args.index_mode,
                    backend=args.backend,
                    simulator=simulator,
                    shots=args.shots,
                ),
            ]
        )
        if args.include_combined:
            rows.append(
                run_combined_process(
                    repetition=repetition,
                    index_mode=args.index_mode,
                    backend=args.backend,
                )
            )
    return rows


def usage_row(
    *,
    task: str,
    repetition: int,
    index_mode: str,
    backend: str,
    shots: int,
    status: str,
    result: Any,
    primary_metric: str,
    primary_value: float | int | None,
    notes: str,
) -> dict[str, Any]:
    row = {
        "group": "usage_task",
        "task": task,
        "repetition": repetition,
        "index_mode": index_mode,
        "backend": backend,
        "shots": shots,
        "status": status,
        "task_time_seconds": getattr(result, "task_time_seconds", None),
        "num_qubits": getattr(result, "num_qubits", None),
        "circuit_depth": getattr(result, "circuit_depth", None),
        "gate_count": getattr(result, "gate_count", None),
        "transpiled_depth": getattr(result, "transpiled_depth", None),
        "transpiled_gate_count": getattr(result, "transpiled_gate_count", None),
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "notes": notes,
        "result": result,
    }
    return row


def run_usage_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        seed = args.seed + repetition - 1

        basis_result = run_basis_lookup_task(shots=args.shots, seed_simulator=seed)
        rows.append(
            usage_row(
                task="search_grover_lookup",
                repetition=repetition,
                index_mode="sequential",
                backend=args.backend,
                shots=args.shots,
                status="success",
                result=basis_result,
                primary_metric="success",
                primary_value=basis_result.success_probability,
                notes=(
                    "sequential-only validation task: compact 3-qubit/8-state "
                    "Grover-style lookup; paper mode would require an "
                    "unnecessarily large 8-qubit/256-state search circuit."
                ),
            )
        )

        amplitude_result = run_amplitude_similarity_task(
            shots=args.shots,
            seed_simulator=seed + 10_000,
        )
        rows.append(
            usage_row(
                task="entity_matching_swap_test",
                repetition=repetition,
                index_mode="feature_vector",
                backend=args.backend,
                shots=args.shots,
                status="success",
                result=amplitude_result,
                primary_metric="similarity",
                primary_value=amplitude_result.estimated_similarity,
                notes=(
                    "Index-mode independent feature-vector task; swap-test "
                    "estimate compared with classical squared dot-product similarity."
                ),
            )
        )

        link_result = run_link_prediction_distance_task(
            shots=args.shots,
            seed_simulator=seed + 20_000,
        )
        rows.append(
            usage_row(
                task="link_prediction_distance_estimation",
                repetition=repetition,
                index_mode="feature_vector",
                backend=args.backend,
                shots=args.shots,
                status="success",
                result=link_result,
                primary_metric="distance",
                primary_value=link_result.classical_distance,
                notes=(
                    "TransE-style e_h + r versus e_t distance-estimation "
                    "validation using amplitude encoding and a swap test; not HHL."
                ),
            )
        )

        multihop_result = run_multihop_phase_kickback_task()
        rows.append(
            usage_row(
                task="multihop_phase_kickback",
                repetition=repetition,
                index_mode="path_phase",
                backend="statevector",
                shots=0,
                status="success",
                result=multihop_result,
                primary_metric="phase error",
                primary_value=multihop_result.phase_error,
                notes=(
                    "Two-hop phase accumulation with controlled phase operations; "
                    "not a full RDFS reasoner."
                ),
            )
        )

        schema_result = run_schema_matching_qft_task()
        rows.append(
            usage_row(
                task="schema_matching_qft",
                repetition=repetition,
                index_mode="phase_pattern",
                backend="statevector",
                shots=0,
                status="success",
                result=schema_result,
                primary_metric="Fourier similarity",
                primary_value=schema_result.fourier_pattern_similarity,
                notes=(
                    "Toy schema-pattern validation using QFT magnitude signatures; "
                    "not full schema matching."
                ),
            )
        )
    return rows


def run_additional_validations(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not args.include_combined:
        return rows

    for repetition in range(1, args.repetitions + 1):
        combined_result = run_combined_demo_task(
            index_mode=args.index_mode,
            output_path=None,
        )
        rows.append(
            usage_row(
                task="combined_amplitude_phase_demo",
                repetition=repetition,
                index_mode=args.index_mode,
                backend="statevector",
                shots=0,
                status="success",
                result=combined_result,
                primary_metric="state_norm",
                primary_value=combined_result.state_norm,
                notes=(
                    "Additional validation after Table 4: combined alpha_k "
                    f"exp(i theta_k) state construction using {args.index_mode} index mode."
                ),
            )
        )
    return rows


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def mean_or_blank(rows: list[dict[str, Any]], field: str) -> float | str:
    values = numeric_values(rows, field)
    return round(statistics.mean(values), 9) if values else ""


def std_or_blank(rows: list[dict[str, Any]], field: str) -> float | str:
    values = numeric_values(rows, field)
    if not values:
        return ""
    if len(values) == 1:
        return 0.0
    return round(statistics.stdev(values), 9)


def first_value(rows: list[dict[str, Any]], field: str) -> Any:
    for row in rows:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return ""


def summarize_encoding_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    experiments = sorted({str(row["experiment"]) for row in rows})
    summary: list[dict[str, Any]] = []
    for experiment in experiments:
        group = [row for row in rows if row["experiment"] == experiment]
        summary.append(
            {
                "experiment": experiment,
                "run_count": len(group),
                "success_count": sum(1 for row in group if row.get("status") == "success"),
                "index_mode": first_value(group, "index_mode"),
                "backend": first_value(group, "backend"),
                "mean_task_time_seconds": mean_or_blank(group, "task_time_seconds"),
                "std_task_time_seconds": std_or_blank(group, "task_time_seconds"),
                "mean_preparation_time_seconds": mean_or_blank(group, "preparation_time_seconds"),
                "mean_num_qubits": mean_or_blank(group, "num_qubits"),
                "mean_dimension": mean_or_blank(group, "dimension"),
                "mean_state_norm": mean_or_blank(group, "state_norm"),
                "mean_circuit_depth": mean_or_blank(group, "circuit_depth"),
                "mean_gate_count": mean_or_blank(group, "gate_count"),
                "mean_transpiled_depth": mean_or_blank(group, "transpiled_depth"),
                "mean_transpiled_gate_count": mean_or_blank(group, "transpiled_gate_count"),
                "notes": first_value(group, "notes"),
            }
        )
    return summary


def summarize_usage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_order = {
        task: position
        for position, task in enumerate(TABLE1_TASK_ORDER)
    }
    tasks = sorted(
        {str(row["task"]) for row in rows},
        key=lambda task: (task_order.get(task, len(task_order)), task),
    )
    summary: list[dict[str, Any]] = []
    for task in tasks:
        group = [row for row in rows if row["task"] == task]
        summary.append(
            {
                "task": task,
                "run_count": len(group),
                "success_count": sum(1 for row in group if row.get("status") == "success"),
                "index_mode": first_value(group, "index_mode"),
                "backend": first_value(group, "backend"),
                "shots": first_value(group, "shots"),
                "mean_task_time_seconds": mean_or_blank(group, "task_time_seconds"),
                "std_task_time_seconds": std_or_blank(group, "task_time_seconds"),
                "mean_num_qubits": mean_or_blank(group, "num_qubits"),
                "mean_circuit_depth": mean_or_blank(group, "circuit_depth"),
                "mean_gate_count": mean_or_blank(group, "gate_count"),
                "mean_transpiled_depth": mean_or_blank(group, "transpiled_depth"),
                "mean_transpiled_gate_count": mean_or_blank(group, "transpiled_gate_count"),
                "primary_metric": first_value(group, "primary_metric"),
                "mean_primary_value": mean_or_blank(group, "primary_value"),
                "notes": first_value(group, "notes"),
            }
        )
    return summary


def number_or_blank(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.{digits}f}".rstrip("0").rstrip(".")


def seconds_to_ms(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return number_or_blank(float(value) * 1000, digits=2)
    except (TypeError, ValueError):
        return ""


def seconds_to_ms_label(value: Any) -> str:
    milliseconds = seconds_to_ms(value)
    return f"{milliseconds} ms" if milliseconds else ""


def number_or_dash(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "--"
    formatted = number_or_blank(value, digits=digits)
    return formatted if formatted else "--"


def shots_or_zero(value: Any) -> str:
    if value in (None, ""):
        return "0"
    formatted = number_or_blank(value)
    return formatted if formatted else "0"


def compact_note(note: Any, max_length: int = 72) -> str:
    text = str(note or "")
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3].rstrip()}..."


def paper_encoding_table_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in summary_rows:
        encoding, variant = ENCODING_LABELS.get(
            str(row.get("experiment")),
            (str(row.get("experiment") or ""), ""),
        )
        rows.append(
            {
                "Encoding": encoding,
                "Variant": variant,
                "Index Mode": str(row.get("index_mode") or ""),
                "Qubits": number_or_blank(row.get("mean_num_qubits")),
                "Dimension": number_or_blank(row.get("mean_dimension")),
                "Time to Create (ms)": seconds_to_ms(
                    row.get("mean_preparation_time_seconds")
                    or row.get("mean_task_time_seconds")
                ),
                "Circuit Depth": number_or_blank(row.get("mean_circuit_depth")),
                "Gate Count": number_or_blank(row.get("mean_gate_count")),
                "Transpiled Depth": number_or_blank(row.get("mean_transpiled_depth")),
                "Transpiled Gate Count": number_or_blank(
                    row.get("mean_transpiled_gate_count")
                ),
                "Notes": compact_note(row.get("notes")),
            }
        )
    return rows


def format_metric(metric_name: Any, value: Any) -> str:
    metric = str(metric_name or "")
    formatted_value = number_or_blank(value, digits=3)
    if not metric:
        return formatted_value
    label = metric.replace("_", " ")
    return f"{label}: {formatted_value}" if formatted_value else label


def paper_usage_table_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    summary_by_task = {
        str(row.get("task")): row
        for row in summary_rows
    }
    rows: list[dict[str, str]] = []
    for task_id in TABLE1_TASK_ORDER:
        if task_id not in summary_by_task:
            continue
        row = summary_by_task[task_id]
        kg_task, encoding, quantum_method = TABLE1_TASK_LABELS[task_id]
        main_result = format_metric(
            row.get("primary_metric"),
            row.get("mean_primary_value"),
        )
        if task_id == "search_grover_lookup":
            main_result = f"{main_result}; sequential-only validation task"
        rows.append(
            {
                "KG Task": kg_task,
                "Encoding": encoding,
                "Quantum Method": quantum_method,
                "Main Result": main_result,
                "Time": seconds_to_ms_label(row.get("mean_task_time_seconds")),
            }
        )
    return rows


def circuit_statistics_table_rows(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build Table 6 from task statistics already returned by the task modules."""

    summary_by_task = {
        str(row.get("task")): row
        for row in summary_rows
    }
    rows: list[dict[str, str]] = []
    for task_id in TABLE1_TASK_ORDER:
        if task_id not in summary_by_task:
            continue
        row = summary_by_task[task_id]
        task, encoding, method = TABLE1_TASK_LABELS[task_id]
        rows.append(
            {
                "Task": task,
                "Encoding": encoding,
                "Method": method.replace(" lookup", ""),
                "Qubits": number_or_dash(row.get("mean_num_qubits")),
                "Circuit Depth": number_or_dash(row.get("mean_circuit_depth")),
                "Gate Count": number_or_dash(row.get("mean_gate_count")),
                "Transpiled Depth": number_or_dash(row.get("mean_transpiled_depth")),
                "Transpiled Gate Count": number_or_dash(
                    row.get("mean_transpiled_gate_count")
                ),
                "Shots": shots_or_zero(row.get("shots")),
                "Repetitions": number_or_dash(row.get("run_count")),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def latex_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex_table(
    rows: list[dict[str, Any]],
    output_path: Path,
    fieldnames: list[str],
    *,
    caption: str,
    label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    column_spec = "@{}" + ("p{0.085\\linewidth}" * len(fieldnames)) + "@{}"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(latex_escape(field) for field in fieldnames) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(latex_escape(row.get(field, "")) for field in fieldnames)
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def environment_payload(args: argparse.Namespace) -> dict[str, Any]:
    qiskit_version = package_version("qiskit")
    qiskit_aer_version = package_version("qiskit-aer")
    numpy_version = package_version("numpy")
    rdflib_version = package_version("rdflib")
    ram_bytes = total_ram_bytes()
    return {
        "timestamp_utc": utc_timestamp(),
        "created_at_utc": utc_timestamp(),
        "python_version": platform.python_version(),
        "python": sys.version,
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "platform": platform.platform(),
        "hostname": socket.gethostname() or None,
        "cpu": cpu_info(),
        "total_ram_bytes": ram_bytes,
        "total_ram_gib": bytes_to_gib(ram_bytes),
        "qiskit_version": qiskit_version,
        "qiskit_aer_version": qiskit_aer_version,
        "numpy_version": numpy_version,
        "rdflib_version": rdflib_version,
        "packages": {
            "qiskit": qiskit_version,
            "qiskit-aer": qiskit_aer_version,
            "numpy": numpy_version,
            "rdflib": rdflib_version,
            "matplotlib": package_version("matplotlib"),
        },
        "script": str(Path(__file__).resolve()),
        "repository_root": str(REPO_ROOT),
        "command_line_arguments": vars(args),
        "arguments": vars(args),
        "random_seed": args.seed,
        "git_commit_hash": git_commit_hash(),
        "note": (
            "This runner only executes Chapter 9 six-triple running-example "
            "experiments. It does not call the old scalability runners."
        ),
    }


def machine_summary_line(payload: dict[str, Any]) -> str:
    os_payload = payload.get("operating_system") or {}
    cpu_payload = payload.get("cpu") or {}
    os_name = f"{os_payload.get('system', '')} {os_payload.get('release', '')}".strip()
    cpu_name = cpu_payload.get("processor") or cpu_payload.get("machine") or "unknown CPU"
    logical_cpus = cpu_payload.get("logical_cpus")
    cpu_suffix = f", {logical_cpus} logical CPUs" if logical_cpus else ""
    return (
        f"{payload.get('hostname') or 'unknown host'} | "
        f"{os_name or payload.get('platform') or 'unknown OS'} | "
        f"{cpu_name}{cpu_suffix} | "
        f"{format_ram(payload.get('total_ram_bytes'))} | "
        f"Qiskit {payload.get('qiskit_version')} / Aer {payload.get('qiskit_aer_version')}"
    )


def plot_numeric_value(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value).strip().split()
        if not text:
            return 0.0
        try:
            return float(text[0])
        except (TypeError, ValueError):
            return 0.0


def encoding_plot_label(row: dict[str, Any]) -> str:
    encoding = str(row.get("Encoding") or "")
    variant = str(row.get("Variant") or "")
    return f"{encoding}\n{variant}" if variant else encoding


def usage_plot_label(row: dict[str, Any]) -> str:
    kg_task = str(row.get("KG Task") or "")
    method = str(row.get("Quantum Method") or "")
    return f"{kg_task}\n{method}" if method else kg_task


def save_bar_figure(
    plt: Any,
    *,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str = "#35618f",
) -> Path | None:
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(max(7.2, 1.15 * len(labels)), 4.4))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def maybe_write_plots(
    encoding_summary: list[dict[str, Any]],
    usage_summary: list[dict[str, Any]],
    table3_rows: list[dict[str, Any]],
    table4_rows: list[dict[str, Any]],
    figures_dir: Path,
    index_mode: str,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for rows, label_field, value_field, title, filename, ylabel in [
        (
            encoding_summary,
            "experiment",
            "mean_task_time_seconds",
            "Chapter 9 Encoding Process Runtime",
            "table3_encoding_process_runtime.png",
            "Mean runtime (seconds)",
        ),
        (
            usage_summary,
            "task",
            "mean_task_time_seconds",
            "Chapter 9 Usage Task Runtime",
            "table4_usage_task_runtime.png",
            "Mean runtime (seconds)",
        ),
    ]:
        labels = [str(row[label_field]) for row in rows]
        values = [plot_numeric_value(row.get(value_field)) for row in rows]
        path = save_bar_figure(
            plt,
            labels=labels,
            values=values,
            title=title,
            ylabel=ylabel,
            output_path=figures_dir / filename,
        )
        if path is not None:
            created.append(path)

    table3_time = save_bar_figure(
        plt,
        labels=[encoding_plot_label(row) for row in table3_rows],
        values=[plot_numeric_value(row.get("Time to Create (ms)")) for row in table3_rows],
        title="Table 3 Encoding Creation Time",
        ylabel="Time to Create (ms)",
        output_path=figures_dir / "table3_encoding_time_bar.png",
        color="#2f6f6d",
    )
    if table3_time is not None:
        created.append(table3_time)

    table3_qubits = save_bar_figure(
        plt,
        labels=[encoding_plot_label(row) for row in table3_rows],
        values=[plot_numeric_value(row.get("Qubits")) for row in table3_rows],
        title="Table 3 Qubit Requirements",
        ylabel="Qubits",
        output_path=figures_dir / "table3_qubits_bar.png",
        color="#8a5a44",
    )
    if table3_qubits is not None:
        created.append(table3_qubits)

    table4_time = save_bar_figure(
        plt,
        labels=[usage_plot_label(row) for row in table4_rows],
        values=[plot_numeric_value(row.get("Time")) for row in table4_rows],
        title="Table 4 Usage Task Runtime",
        ylabel="Time (ms)",
        output_path=figures_dir / "table4_task_time_bar.png",
        color="#6b5b95",
    )
    if table4_time is not None:
        created.append(table4_time)

    amplitude_result = build_sparse_index_amplitude_encoding(
        triples=get_running_example_triples(),
        index_mode=index_mode,
    )
    amplitude_indices = amplitude_result.nonzero_indices
    amplitude_probabilities = [
        float(abs(amplitude_result.statevector[index]) ** 2)
        for index in amplitude_indices
    ]
    amplitude_path = save_bar_figure(
        plt,
        labels=[str(index) for index in amplitude_indices],
        values=amplitude_probabilities,
        title="Amplitude Encoding Measurement Probabilities",
        ylabel="Probability",
        output_path=figures_dir / "amplitude_probabilities.png",
        color="#4f7cac",
    )
    if amplitude_path is not None:
        created.append(amplitude_path)

    combined_result = combined_amplitude_phase_encoding(index_mode=index_mode)
    combined_indices = combined_result.nonzero_indices
    if combined_indices:
        labels = [str(index) for index in combined_indices]
        magnitudes = [
            float(combined_result.amplitude_map[index])
            for index in combined_indices
        ]
        phases = [
            float(combined_result.index_phase_map[index])
            for index in combined_indices
        ]
        fig, magnitude_axis = plt.subplots(
            figsize=(max(7.2, 1.15 * len(labels)), 4.4)
        )
        magnitude_axis.bar(labels, magnitudes, color="#2f6f6d", label="Magnitude")
        magnitude_axis.set_title("Combined Encoding Magnitude and Phase")
        magnitude_axis.set_xlabel("Triple index")
        magnitude_axis.set_ylabel("Amplitude magnitude")
        magnitude_axis.tick_params(axis="x", rotation=25, labelsize=8)
        magnitude_axis.grid(axis="y", alpha=0.25)

        phase_axis = magnitude_axis.twinx()
        phase_axis.plot(labels, phases, color="#9c3d54", marker="o", label="Phase")
        phase_axis.set_ylabel("Phase angle (radians)")
        phase_axis.set_ylim(0, max(phases + [math.pi]) * 1.15)

        handles_1, labels_1 = magnitude_axis.get_legend_handles_labels()
        handles_2, labels_2 = phase_axis.get_legend_handles_labels()
        magnitude_axis.legend(
            handles_1 + handles_2,
            labels_1 + labels_2,
            loc="upper right",
            fontsize=8,
        )
        fig.tight_layout()
        combined_path = figures_dir / "combined_magnitude_phase.png"
        fig.savefig(combined_path, dpi=200)
        plt.close(fig)
        created.append(combined_path)

    return created


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Chapter 9 six-triple running-example experiments only."
    )
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--index-mode",
        choices=("sequential", "paper"),
        default="sequential",
    )
    parser.add_argument("--output-dir", default="results/chapter9")
    parser.add_argument(
        "--backend",
        choices=("aer_simulator",),
        default="aer_simulator",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--include-combined",
        action="store_true",
        default=True,
        help="Include combined amplitude-phase rows. Enabled by default.",
    )
    parser.add_argument(
        "--exclude-combined",
        dest="include_combined",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save JSON outputs. Enabled by default.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        default=True,
        help="Save CSV table outputs. Enabled by default.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.shots < 1:
        raise ValueError("--shots must be at least 1.")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    validate_args(args)

    output_dir = Path(args.output_dir)
    run_started_at = utc_timestamp()
    start = time.perf_counter()

    encoding_rows = run_encoding_processes(args)
    usage_rows = run_usage_tasks(args)
    additional_validation_rows = run_additional_validations(args)
    encoding_summary = summarize_encoding_rows(encoding_rows)
    usage_summary = summarize_usage_rows(usage_rows)
    additional_validation_summary = summarize_usage_rows(additional_validation_rows)
    table3_rows = paper_encoding_table_rows(encoding_summary)
    table4_rows = paper_usage_table_rows(usage_summary)
    table6_rows = circuit_statistics_table_rows(usage_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    table3_path = output_dir / "table3_encoding_process.csv"
    table4_path = output_dir / "table4_usage_tasks.csv"
    table6_path = output_dir / "table6_circuit_statistics.csv"
    table3_tex_path = output_dir / "table3_encoding_process.tex"
    table4_tex_path = output_dir / "table4_usage_tasks.tex"
    table6_tex_path = output_dir / "table6_circuit_statistics.tex"
    raw_path = output_dir / "chapter9_raw_results.json"
    env_path = output_dir / "environment.json"

    if args.save_csv:
        write_csv(table3_rows, table3_path, ENCODING_TABLE_FIELDS)
        write_csv(table4_rows, table4_path, USAGE_TABLE_FIELDS)
        write_csv(table6_rows, table6_path, TABLE6_CIRCUIT_FIELDS)
        write_latex_table(
            table3_rows,
            table3_tex_path,
            ENCODING_LATEX_FIELDS,
            caption="Chapter 9 encoding process benchmark on the six-triple running example.",
            label="tab:chapter9-encoding-process",
        )
        write_latex_table(
            table4_rows,
            table4_tex_path,
            USAGE_LATEX_FIELDS,
            caption="Chapter 9 usage task benchmark on the six-triple running example.",
            label="tab:chapter9-usage-tasks",
        )
        write_latex_table(
            table6_rows,
            table6_tex_path,
            TABLE6_CIRCUIT_FIELDS,
            caption="Chapter 9 circuit statistics for running-example tasks.",
            label="tab:chapter9-circuit-statistics",
        )

    figures = maybe_write_plots(
        encoding_summary=encoding_summary,
        usage_summary=usage_summary,
        table3_rows=table3_rows,
        table4_rows=table4_rows,
        figures_dir=output_dir / "figures",
        index_mode=args.index_mode,
    )

    total_runtime = time.perf_counter() - start
    raw_payload = {
        "run_started_at_utc": run_started_at,
        "run_finished_at_utc": utc_timestamp(),
        "total_runtime_seconds": total_runtime,
        "settings": vars(args),
        "running_example_triple_count": len(get_running_example_triples()),
        "encoding_process_rows": encoding_rows,
        "usage_task_rows": usage_rows,
        "additional_validation_rows": additional_validation_rows,
        "table3_encoding_process_internal": encoding_summary,
        "table4_usage_tasks_internal": usage_summary,
        "additional_validation_internal": additional_validation_summary,
        "table3_encoding_process": table3_rows,
        "table4_usage_tasks": table4_rows,
        "table6_circuit_statistics": table6_rows,
        "output_files": {
            "table3_encoding_process": table3_path,
            "table4_usage_tasks": table4_path,
            "table6_circuit_statistics": table6_path,
            "table3_encoding_process_tex": table3_tex_path,
            "table4_usage_tasks_tex": table4_tex_path,
            "table6_circuit_statistics_tex": table6_tex_path,
            "raw_results": raw_path,
            "environment": env_path,
            "figures": figures,
        },
    }
    if args.save_json:
        write_json(raw_payload, raw_path)
    environment = environment_payload(args)
    write_json(environment, env_path)

    print()
    print("Chapter 9 experiments complete")
    print(f"  Machine: {machine_summary_line(environment)}")
    print(f"  Running example triples: {len(get_running_example_triples())}")
    print(f"  Repetitions: {args.repetitions}")
    print(f"  Shots: {args.shots}")
    print(f"  Index mode: {args.index_mode}")
    print(f"  Encoding process rows: {len(encoding_rows)}")
    print(f"  Usage task rows: {len(usage_rows)}")
    print(f"  Additional validation rows: {len(additional_validation_rows)}")
    print(f"  Output directory: {output_dir}")
    if args.save_csv:
        print(f"  Table 3 CSV: {table3_path}")
        print(f"  Table 4 CSV: {table4_path}")
        print(f"  Table 6 CSV: {table6_path}")
        print(f"  Table 3 LaTeX: {table3_tex_path}")
        print(f"  Table 4 LaTeX: {table4_tex_path}")
        print(f"  Table 6 LaTeX: {table6_tex_path}")
    if args.save_json:
        print(f"  Raw JSON: {raw_path}")
    print(f"  Environment JSON: {env_path}")
    if figures:
        print(f"  Figures: {output_dir / 'figures'}")
    print("  Note: old scalability experiments were not run.")

    return raw_payload


if __name__ == "__main__":
    main()
