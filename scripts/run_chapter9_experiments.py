from __future__ import annotations

import argparse
import csv
from collections import Counter
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
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
from src.tasks.keyword_search import run_keyword_search_task
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

VALIDATION_TABLE_FIELDS = [
    "KG Task",
    "Validation Metric",
    "Value",
    "Notes",
]

VALIDATION_LATEX_FIELDS = VALIDATION_TABLE_FIELDS

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
    "Measurement Mode",
    "Grover Iterations",
    "Claim Scope",
]

PER_REPETITION_FIELDS = [
    "group",
    "task",
    "experiment",
    "repetition",
    "index_mode",
    "backend",
    "measurement_mode",
    "status",
    "shots",
    "num_qubits",
    "circuit_depth",
    "gate_count",
    "transpiled_depth",
    "transpiled_gate_count",
    "task_time_seconds",
    "state_preparation_time_seconds",
    "simulation_time_seconds",
    "measurement_time_seconds",
    "readout_decoding_time_seconds",
    "primary_metric",
    "primary_value",
    "expected_value",
    "estimated_value",
    "absolute_error",
    "relative_error",
    "pass_threshold",
    "pass_fail",
    "threshold_reason",
    "recommended_grover_iterations",
    "grover_iterations",
    "search_space_size",
    "marked_count",
    "claim_scope",
    "notes",
]

SCORE_DERIVATION_FIELDS = [
    "task",
    "repetition",
    "score_name",
    "expected_value",
    "estimated_value",
    "exact_or_classical_value",
    "formula",
    "counts_source",
    "shots",
    "absolute_error",
    "relative_error",
    "pass_threshold",
    "pass_fail",
    "threshold_reason",
]

THRESHOLD_FIELDS = [
    "task",
    "repetition",
    "expected_value",
    "estimated_value",
    "absolute_error",
    "relative_error",
    "pass_threshold",
    "pass_fail",
    "threshold_reason",
]

SUITABILITY_FIELDS = [
    "task",
    "encoding",
    "measurement_mode",
    "qubits",
    "state_preparation_time_seconds",
    "circuit_construction_time_seconds",
    "simulation_time_seconds",
    "measurement_time_seconds",
    "readout_decoding_time_seconds",
    "circuit_depth",
    "gate_count",
    "transpiled_depth",
    "transpiled_gate_count",
    "shots",
    "error_against_exact_or_classical",
    "pass_fail",
    "notes",
    "claim_scope",
]

SYNTHETIC_ENCODINGS = ("basis", "amplitude", "phase", "combined")
DEFAULT_SYNTHETIC_SIZES = (6, 10, 25, 50, 100, 250, 500)

SYNTHETIC_TABLE_FIELDS = [
    "triple_count",
    "entity_count",
    "predicate_count",
    "encoding",
    "index_mode",
    "qubits",
    "dimension",
    "preprocessing_time_ms",
    "encoding_time_ms",
    "circuit_construction_time_ms",
    "transpilation_time_ms",
    "simulation_time_ms",
    "total_time_ms",
    "circuit_depth",
    "gate_count",
    "transpiled_depth",
    "transpiled_gate_count",
    "status",
    "notes",
]

SYNTHETIC_LATEX_FIELDS = [
    "Triples",
    "Encoding",
    "Qubits",
    "Enc. Time",
    "Depth",
    "Sim. Time",
    "Status",
]

REAL_TABLE_FIELDS = [
    "dataset_name",
    "triple_count",
    "entity_count",
    "predicate_count",
    "encoding",
    "qubits",
    "dimension",
    "preprocessing_time_ms",
    "encoding_time_ms",
    "circuit_construction_time_ms",
    "transpilation_time_ms",
    "simulation_time_ms",
    "total_time_ms",
    "circuit_depth",
    "gate_count",
    "transpiled_depth",
    "transpiled_gate_count",
    "status",
    "notes",
]

REAL_LATEX_FIELDS = [
    "Dataset",
    "Triples",
    "Entities",
    "Predicates",
    "Encoding",
    "Qubits",
    "Enc. Time",
    "Status",
]

TABLE1_TASK_ORDER = [
    "search_grover_lookup",
    "entity_matching_swap_test",
    "keyword_search_swap_test",
    "link_prediction_distance_estimation",
    "multihop_phase_kickback",
    "schema_matching_qft",
]

TABLE1_TASK_LABELS = {
    "search_grover_lookup": ("Search", "Basis", "Grover lookup"),
    "entity_matching_swap_test": ("Entity Matching", "Amplitude", "Swap Test"),
    "keyword_search_swap_test": ("Keyword Search", "Amplitude", "Swap Test"),
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


def timed_call(function, *args, **kwargs) -> tuple[Any, float]:
    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start_time


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
    expected_value = getattr(result, "expected_value", None)
    if expected_value is None:
        expected_value = getattr(result, "classical_similarity", None)
    if expected_value is None:
        expected_value = getattr(result, "exact_statevector_quantity", None)
    if expected_value is None:
        expected_value = getattr(result, "exact_distribution_similarity", None)

    estimated_value = getattr(result, "estimated_value", None)
    if estimated_value is None:
        estimated_value = getattr(result, "estimated_similarity", None)
    if estimated_value is None:
        estimated_value = getattr(result, "shot_based_estimate", None)
    if estimated_value is None:
        estimated_value = getattr(result, "measured_distribution_similarity", None)

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
        "measurement_mode": getattr(result, "measurement_mode", ""),
        "claim_scope": getattr(result, "claim_scope", ""),
        "expected_value": expected_value,
        "estimated_value": estimated_value,
        "absolute_error": getattr(result, "absolute_error", None),
        "relative_error": getattr(result, "relative_error", None),
        "pass_threshold": getattr(result, "pass_threshold", None),
        "pass_fail": getattr(result, "pass_fail", ""),
        "threshold_reason": getattr(result, "threshold_reason", ""),
        "state_preparation_time_seconds": getattr(
            result,
            "state_preparation_time_seconds",
            None,
        ),
        "simulation_time_seconds": getattr(result, "simulation_time_seconds", None),
        "measurement_time_seconds": getattr(result, "measurement_time_seconds", None),
        "readout_decoding_time_seconds": getattr(
            result,
            "readout_decoding_time_seconds",
            None,
        ),
        "recommended_grover_iterations": getattr(
            result,
            "recommended_grover_iterations",
            None,
        ),
        "grover_iterations": getattr(result, "grover_iterations", None),
        "grover_iteration_formula": getattr(result, "grover_iteration_formula", ""),
        "search_space_size": getattr(result, "search_space_size", None),
        "marked_count": getattr(result, "marked_count", None),
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

        keyword_result = run_keyword_search_task(
            shots=args.shots,
            seed_simulator=seed + 15_000,
            repetitions=args.repetitions,
        )
        rows.append(
            usage_row(
                task="keyword_search_swap_test",
                repetition=repetition,
                index_mode="feature_vector",
                backend=args.backend,
                shots=args.shots,
                status="success",
                result=keyword_result,
                primary_metric="top_score",
                primary_value=keyword_result.top_score,
                notes=(
                    f"top result = {keyword_result.top_result}; small "
                    "keyword-search validation over the running example using "
                    "amplitude encoding and a swap test; not a full RDF keyword "
                    "search engine."
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

        multihop_result = run_multihop_phase_kickback_task(
            shots=args.shots,
            seed_simulator=seed + 30_000,
        )
        rows.append(
            usage_row(
                task="multihop_phase_kickback",
                repetition=repetition,
                index_mode="path_phase",
                backend=args.backend,
                shots=args.shots,
                status="success",
                result=multihop_result,
                primary_metric="shot phase error",
                primary_value=multihop_result.shot_phase_error,
                notes=(
                    "Two-hop phase accumulation with exact statevector and "
                    "shot-based X/Y Hadamard-test validation; not a full RDFS reasoner."
                ),
            )
        )

        schema_result = run_schema_matching_qft_task(
            shots=args.shots,
            seed_simulator=seed + 40_000,
        )
        rows.append(
            usage_row(
                task="schema_matching_qft",
                repetition=repetition,
                index_mode="phase_pattern",
                backend=args.backend,
                shots=args.shots,
                status="success",
                result=schema_result,
                primary_metric="measured Fourier similarity",
                primary_value=schema_result.measured_distribution_similarity,
                notes=(
                    "Toy schema-pattern validation using measured QFT distributions "
                    "with exact-URI, synonym-dictionary, and negative-control assumptions; "
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
                "measurement_mode": first_value(group, "measurement_mode"),
                "claim_scope": first_value(group, "claim_scope"),
                "pass_fail": first_value(group, "pass_fail"),
                "mean_absolute_error": mean_or_blank(group, "absolute_error"),
                "mean_pass_threshold": mean_or_blank(group, "pass_threshold"),
                "recommended_grover_iterations": first_value(
                    group,
                    "recommended_grover_iterations",
                ),
                "grover_iterations": first_value(group, "grover_iterations"),
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


def seconds_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def seconds_to_ms_value(value: Any) -> str:
    seconds = seconds_value(value)
    if seconds is None:
        return "--"
    return number_or_blank(seconds * 1000, digits=3)


def sum_seconds_to_ms(*values: Any) -> str:
    numeric_values = [seconds_value(value) for value in values]
    present_values = [value for value in numeric_values if value is not None]
    if not present_values:
        return "--"
    return number_or_blank(sum(present_values) * 1000, digits=3)


def power_of_two_dimension(qubits: Any) -> str:
    if qubits in (None, ""):
        return "--"
    try:
        qubit_count = int(float(qubits))
    except (TypeError, ValueError):
        return "--"
    return str(2**qubit_count)


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


def validation_metric_table_rows(
    usage_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task_id in TABLE1_TASK_ORDER:
        group = [row for row in usage_rows if row.get("task") == task_id]
        if not group:
            continue

        values = numeric_values(group, "primary_value")
        mean_value = statistics.mean(values) if values else ""
        task, _, _ = TABLE1_TASK_LABELS[task_id]
        first_result = group[0].get("result")
        notes = compact_note(group[0].get("notes"), max_length=96)

        if task_id == "keyword_search_swap_test" and first_result is not None:
            notes = compact_note(
                f"top result = {getattr(first_result, 'top_result', '')}; "
                "small running-example keyword-search validation",
                max_length=96,
            )

        rows.append(
            {
                "KG Task": task,
                "Validation Metric": str(group[0].get("primary_metric") or ""),
                "Value": number_or_blank(mean_value),
                "Notes": notes,
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
                "Measurement Mode": compact_note(row.get("measurement_mode"), 44),
                "Grover Iterations": number_or_dash(row.get("grover_iterations")),
                "Claim Scope": compact_note(row.get("claim_scope"), 44),
            }
        )
    return rows


def synthetic_dataset_config(size: int) -> tuple[int, int, int]:
    """Use the existing scaling config when present, else the same generator shape."""

    from generate_scaling_datasets import DATASET_CONFIGS

    if size in DATASET_CONFIGS:
        return DATASET_CONFIGS[size]
    entity_count = max(2, size // 2)
    predicate_count = max(1, min(5, size))
    return size, entity_count, predicate_count


def scaling_args_for_chapter9(
    args: argparse.Namespace,
    *,
    dataset_category: str = "synthetic",
) -> argparse.Namespace:
    is_synthetic = dataset_category == "synthetic"
    return argparse.Namespace(
        shots=args.shots,
        rdf_format="turtle" if is_synthetic else None,
        weight_strategy="uniform",
        phase_marker_mode="synthetic-default" if is_synthetic else "most-common-predicate",
        phase_mark_predicate=None,
        phase_angle=math.pi,
        max_basis_simulation_qubits=22,
        max_phase_diagonal_qubits=10,
        max_metric_qubits=14,
        decompose_reps=1,
        compute_decomposed_metrics=False,
        compute_transpiled_metrics=True,
    )


def synthetic_index_mode_for_encoding(encoding: str) -> str:
    return {
        "basis": "entity-predicate-object",
        "amplitude": "compact",
        "phase": "sequential-index",
        "combined": "unsupported",
    }.get(encoding, "--")


def synthetic_notes_from_scaling_row(row: dict[str, Any]) -> str:
    notes = ["software-level observation; no quantum-advantage claim"]
    for field in ("error_message", "phase_warning", "metric_error"):
        value = row.get(field)
        if value not in (None, ""):
            notes.append(str(value))
    return "; ".join(notes)


def real_notes_from_scaling_row(
    row: dict[str, Any],
    *,
    original_triple_count: int,
    max_real_triples: int,
) -> str:
    notes = ["software-level observation; no quantum-advantage claim"]
    if original_triple_count > max_real_triples:
        notes.append(
            f"deterministically truncated from {original_triple_count} triples "
            f"to {max_real_triples}"
        )
    for field in ("error_message", "phase_warning", "metric_error"):
        value = row.get(field)
        if value not in (None, ""):
            notes.append(str(value))
    return "; ".join(notes)


def synthetic_table_row_from_scaling_row(
    row: dict[str, Any],
    *,
    encoding: str,
    triple_count: int,
    entity_count: int,
    predicate_count: int,
) -> dict[str, Any]:
    qubits = number_or_dash(row.get("qubits"))
    status = str(row.get("status") or "unknown")
    return {
        "triple_count": triple_count,
        "entity_count": entity_count,
        "predicate_count": predicate_count,
        "encoding": encoding,
        "index_mode": synthetic_index_mode_for_encoding(encoding),
        "qubits": qubits,
        "dimension": power_of_two_dimension(row.get("qubits")),
        "preprocessing_time_ms": sum_seconds_to_ms(
            row.get("rdf_parse_time"),
            row.get("id_mapping_time"),
        ),
        "encoding_time_ms": seconds_to_ms_value(row.get("state_preparation_time")),
        "circuit_construction_time_ms": seconds_to_ms_value(
            row.get("circuit_construction_time")
        ),
        "transpilation_time_ms": "--",
        "simulation_time_ms": seconds_to_ms_value(row.get("simulation_time")),
        "total_time_ms": seconds_to_ms_value(row.get("total_runtime")),
        "circuit_depth": number_or_dash(row.get("circuit_depth")),
        "gate_count": number_or_dash(row.get("gate_count")),
        "transpiled_depth": number_or_dash(row.get("transpiled_circuit_depth")),
        "transpiled_gate_count": number_or_dash(row.get("transpiled_gate_count")),
        "status": status,
        "notes": synthetic_notes_from_scaling_row(row),
    }


def synthetic_skip_row(
    *,
    encoding: str,
    triple_count: int,
    entity_count: int,
    predicate_count: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "triple_count": triple_count,
        "entity_count": entity_count,
        "predicate_count": predicate_count,
        "encoding": encoding,
        "index_mode": synthetic_index_mode_for_encoding(encoding),
        "qubits": "--",
        "dimension": "--",
        "preprocessing_time_ms": "--",
        "encoding_time_ms": "--",
        "circuit_construction_time_ms": "--",
        "transpilation_time_ms": "--",
        "simulation_time_ms": "--",
        "total_time_ms": "--",
        "circuit_depth": "--",
        "gate_count": "--",
        "transpiled_depth": "--",
        "transpiled_gate_count": "--",
        "status": "skipped",
        "notes": notes,
    }


def run_chapter9_synthetic_experiments(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    from generate_scaling_datasets import generate_dataset
    from run_scaling_experiments import run_one_configuration

    scaling_args = scaling_args_for_chapter9(args)
    dataset_dir = output_dir / "synthetic_datasets"
    table_rows: list[dict[str, Any]] = []
    raw_scaling_rows: list[dict[str, Any]] = []

    for size in args.synthetic_sizes:
        triple_count, entity_count, predicate_count = synthetic_dataset_config(size)
        dataset_path = dataset_dir / f"synthetic_{size}.ttl"
        generate_dataset(
            output_path=dataset_path,
            triple_count=triple_count,
            entity_count=entity_count,
            predicate_count=predicate_count,
        )

        for repetition in range(1, args.synthetic_repetitions + 1):
            for encoding in SYNTHETIC_ENCODINGS:
                if encoding == "combined":
                    table_rows.append(
                        synthetic_skip_row(
                            encoding=encoding,
                            triple_count=triple_count,
                            entity_count=entity_count,
                            predicate_count=predicate_count,
                            notes=(
                                "Combined synthetic scaling is not supported by the "
                                "existing scalability runner; skipped. "
                                "software-level observation; no quantum-advantage claim"
                            ),
                        )
                    )
                    continue

                scaling_row = run_one_configuration(
                    dataset_path=dataset_path,
                    dataset_category="synthetic",
                    encoding=encoding,
                    repetition=repetition,
                    args=scaling_args,
                    dataset_size=triple_count,
                )
                raw_scaling_rows.append(scaling_row)
                table_rows.append(
                    synthetic_table_row_from_scaling_row(
                        scaling_row,
                        encoding=encoding,
                        triple_count=triple_count,
                        entity_count=entity_count,
                        predicate_count=predicate_count,
                    )
                )

    return {
        "settings": {
            "sizes": list(args.synthetic_sizes),
            "repetitions": args.synthetic_repetitions,
            "encodings": list(SYNTHETIC_ENCODINGS),
            "label": "software-level observations; no quantum-advantage claim",
        },
        "rows": table_rows,
        "raw_scaling_rows": raw_scaling_rows,
    }


def resolve_real_kg_paths(file_args: list[str] | None) -> list[Path]:
    from run_scaling_experiments import DEFAULT_REAL_KG_DIR, REAL_KG_FILES

    if not file_args:
        return [DEFAULT_REAL_KG_DIR / filename for filename in REAL_KG_FILES]

    paths: list[Path] = []
    for value in file_args:
        path = Path(value)
        candidates = [path]
        if not path.is_absolute():
            candidates.extend(
                [
                    REPO_ROOT / path,
                    DEFAULT_REAL_KG_DIR / path,
                ]
            )
        selected = next((candidate for candidate in candidates if candidate.exists()), path)
        paths.append(selected)
    return paths


def real_table_row_from_scaling_row(
    row: dict[str, Any],
    *,
    dataset_name: str,
    triple_count: int | str,
    entity_count: int | str,
    predicate_count: int | str,
    encoding: str,
    original_triple_count: int,
    max_real_triples: int,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "triple_count": triple_count,
        "entity_count": entity_count,
        "predicate_count": predicate_count,
        "encoding": encoding,
        "qubits": number_or_dash(row.get("qubits")),
        "dimension": power_of_two_dimension(row.get("qubits")),
        "preprocessing_time_ms": sum_seconds_to_ms(
            row.get("rdf_parse_time"),
            row.get("id_mapping_time"),
        ),
        "encoding_time_ms": seconds_to_ms_value(row.get("state_preparation_time")),
        "circuit_construction_time_ms": seconds_to_ms_value(
            row.get("circuit_construction_time")
        ),
        "transpilation_time_ms": "--",
        "simulation_time_ms": seconds_to_ms_value(row.get("simulation_time")),
        "total_time_ms": seconds_to_ms_value(row.get("total_runtime")),
        "circuit_depth": number_or_dash(row.get("circuit_depth")),
        "gate_count": number_or_dash(row.get("gate_count")),
        "transpiled_depth": number_or_dash(row.get("transpiled_circuit_depth")),
        "transpiled_gate_count": number_or_dash(row.get("transpiled_gate_count")),
        "status": str(row.get("status") or "unknown"),
        "notes": real_notes_from_scaling_row(
            row,
            original_triple_count=original_triple_count,
            max_real_triples=max_real_triples,
        ),
    }


def real_skip_row(
    *,
    dataset_name: str,
    triple_count: int | str,
    entity_count: int | str,
    predicate_count: int | str,
    encoding: str,
    status: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "triple_count": triple_count,
        "entity_count": entity_count,
        "predicate_count": predicate_count,
        "encoding": encoding,
        "qubits": "--",
        "dimension": "--",
        "preprocessing_time_ms": "--",
        "encoding_time_ms": "--",
        "circuit_construction_time_ms": "--",
        "transpilation_time_ms": "--",
        "simulation_time_ms": "--",
        "total_time_ms": "--",
        "circuit_depth": "--",
        "gate_count": "--",
        "transpiled_depth": "--",
        "transpiled_gate_count": "--",
        "status": status,
        "notes": notes,
    }


def run_real_encoding_observation(
    *,
    dataset_path: Path,
    dataset_name: str,
    encoding: str,
    context: Any,
    parse_elapsed: float,
    mapping_elapsed: float,
    original_triple_count: int,
    args: argparse.Namespace,
    scaling_args: argparse.Namespace,
) -> dict[str, Any]:
    from run_scaling_experiments import (
        SimulatorLimitError,
        apply_runtime_settings,
        empty_row,
        rounded_seconds,
        run_amplitude_scaling,
        run_basis_scaling,
        run_phase_scaling,
        update_context_metrics,
        utc_timestamp as scaling_utc_timestamp,
    )

    row = empty_row(
        dataset_path=dataset_path,
        dataset_category="real",
        encoding=encoding,
        repetition=1,
        shots=args.shots,
        dataset_size=context.triple_count,
    )
    apply_runtime_settings(row, scaling_args)
    row["rdf_parse_time"] = rounded_seconds(parse_elapsed)
    row["id_mapping_time"] = rounded_seconds(mapping_elapsed)
    row["original_triple_count"] = original_triple_count
    update_context_metrics(row, context)
    run_start = time.perf_counter()

    try:
        if encoding == "basis":
            run_basis_scaling(row, context, scaling_args)
        elif encoding == "amplitude":
            run_amplitude_scaling(row, context, scaling_args)
        elif encoding == "phase":
            run_phase_scaling(row, context, scaling_args)
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
        preprocessing_elapsed = parse_elapsed + mapping_elapsed
        row["total_runtime"] = rounded_seconds(preprocessing_elapsed + total_elapsed)
        if row["success"]:
            row["completed_total_runtime"] = row["total_runtime"]
        row["finished_at_utc"] = scaling_utc_timestamp()
        row["dataset_name"] = dataset_name

    return row


def run_chapter9_real_kg_experiments(
    args: argparse.Namespace,
) -> dict[str, Any]:
    from src.id_mapper import build_encoding_context
    from src.kg_parser import load_triples

    scaling_args = scaling_args_for_chapter9(args, dataset_category="real")
    table_rows: list[dict[str, Any]] = []
    raw_scaling_rows: list[dict[str, Any]] = []
    real_paths = resolve_real_kg_paths(args.real_kg_files)

    for dataset_path in real_paths:
        dataset_name = dataset_path.stem
        if not dataset_path.exists():
            note = (
                f"Real KG file not found: {dataset_path}; "
                "software-level observation; no quantum-advantage claim"
            )
            for encoding in SYNTHETIC_ENCODINGS:
                table_rows.append(
                    real_skip_row(
                        dataset_name=dataset_name,
                        triple_count="--",
                        entity_count="--",
                        predicate_count="--",
                        encoding=encoding,
                        status="skipped",
                        notes=note,
                    )
                )
            continue

        try:
            triples, parse_elapsed = timed_call(load_triples, dataset_path, None)
            original_triple_count = len(triples)
            truncated_triples = triples[: args.max_real_triples]
            context, mapping_elapsed = timed_call(
                build_encoding_context,
                triples=truncated_triples,
                fixed_thesis_mapping=False,
                dataset_path=dataset_path,
            )
        except Exception as exc:
            note = (
                f"Could not parse/build context for {dataset_path}: "
                f"{type(exc).__name__}: {exc}; software-level observation; "
                "no quantum-advantage claim"
            )
            for encoding in SYNTHETIC_ENCODINGS:
                table_rows.append(
                    real_skip_row(
                        dataset_name=dataset_name,
                        triple_count="--",
                        entity_count="--",
                        predicate_count="--",
                        encoding=encoding,
                        status="parse_error",
                        notes=note,
                    )
                )
            continue

        for encoding in SYNTHETIC_ENCODINGS:
            if encoding == "combined":
                table_rows.append(
                    real_skip_row(
                        dataset_name=dataset_name,
                        triple_count=context.triple_count,
                        entity_count=context.entity_count,
                        predicate_count=context.predicate_count,
                        encoding=encoding,
                        status="skipped",
                        notes=(
                            "Combined real-KG scaling is not supported by the "
                            "existing scalability runner; skipped. "
                            "software-level observation; no quantum-advantage claim"
                        ),
                    )
                )
                continue

            scaling_row = run_real_encoding_observation(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                encoding=encoding,
                context=context,
                parse_elapsed=parse_elapsed,
                mapping_elapsed=mapping_elapsed,
                original_triple_count=original_triple_count,
                args=args,
                scaling_args=scaling_args,
            )
            raw_scaling_rows.append(scaling_row)
            table_rows.append(
                real_table_row_from_scaling_row(
                    scaling_row,
                    dataset_name=dataset_name,
                    triple_count=context.triple_count,
                    entity_count=context.entity_count,
                    predicate_count=context.predicate_count,
                    encoding=encoding,
                    original_triple_count=original_triple_count,
                    max_real_triples=args.max_real_triples,
                )
            )

    return {
        "settings": {
            "files": [str(path) for path in real_paths],
            "max_real_triples": args.max_real_triples,
            "encodings": list(SYNTHETIC_ENCODINGS),
            "label": "software-level observations; no quantum-advantage claim",
        },
        "rows": table_rows,
        "raw_scaling_rows": raw_scaling_rows,
    }


def mean_numeric_string(rows: list[dict[str, Any]], field: str) -> str:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value in (None, "", "--"):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return "--"
    return number_or_blank(statistics.mean(values), digits=3)


def status_summary(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("status") or "unknown") for row in rows)
    if len(counts) == 1:
        return next(iter(counts))
    return "; ".join(f"{status}:{count}" for status, count in sorted(counts.items()))


def synthetic_latex_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["triple_count"]), str(row["encoding"]))
        groups.setdefault(key, []).append(row)

    compact_rows: list[dict[str, str]] = []
    for (triple_count, encoding), group in sorted(groups.items()):
        compact_rows.append(
            {
                "Triples": str(triple_count),
                "Encoding": encoding,
                "Qubits": mean_numeric_string(group, "qubits"),
                "Enc. Time": mean_numeric_string(group, "encoding_time_ms"),
                "Depth": mean_numeric_string(group, "circuit_depth"),
                "Sim. Time": mean_numeric_string(group, "simulation_time_ms"),
                "Status": status_summary(group),
            }
        )
    return compact_rows


def real_latex_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact_rows: list[dict[str, str]] = []
    for row in rows:
        compact_rows.append(
            {
                "Dataset": str(row.get("dataset_name") or ""),
                "Triples": str(row.get("triple_count") or "--"),
                "Entities": str(row.get("entity_count") or "--"),
                "Predicates": str(row.get("predicate_count") or "--"),
                "Encoding": str(row.get("encoding") or ""),
                "Qubits": str(row.get("qubits") or "--"),
                "Enc. Time": str(row.get("encoding_time_ms") or "--"),
                "Status": str(row.get("status") or "--"),
            }
        )
    return compact_rows


def numeric_table_value(value: Any) -> float | None:
    if value in (None, "", "--"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def synthetic_plot_points(
    rows: list[dict[str, Any]],
    metric_field: str,
) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        x_value = numeric_table_value(row.get("triple_count"))
        y_value = numeric_table_value(row.get(metric_field))
        encoding = str(row.get("encoding") or "")
        if x_value is None or y_value is None or not encoding:
            continue
        grouped.setdefault((encoding, int(x_value)), []).append(y_value)

    points_by_encoding: dict[str, list[tuple[float, float]]] = {}
    for (encoding, triple_count), values in grouped.items():
        points_by_encoding.setdefault(encoding, []).append(
            (float(triple_count), statistics.mean(values))
        )
    return {
        encoding: sorted(points, key=lambda item: item[0])
        for encoding, points in sorted(points_by_encoding.items())
    }


def write_synthetic_observation_plots(
    rows: list[dict[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    if not rows:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from run_scaling_experiments import (
            AXIS_BACKGROUND,
            ENCODING_COLORS,
            ENCODING_MARKERS,
            GRID_COLOR,
            PLOT_BACKGROUND,
            compact_number,
            pretty_label,
        )
    except Exception:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    plot_specs = [
        (
            "encoding_time_ms",
            "Encoding time (ms)",
            "Section 9.2 Synthetic Encoding Time",
            "synthetic_encoding_time.png",
        ),
        (
            "qubits",
            "Qubits",
            "Section 9.2 Synthetic Qubits",
            "synthetic_qubits.png",
        ),
        (
            "circuit_depth",
            "Circuit depth",
            "Section 9.2 Synthetic Circuit Depth",
            "synthetic_depth.png",
        ),
        (
            "total_time_ms",
            "Total time (ms)",
            "Section 9.2 Synthetic Total Time",
            "synthetic_total_time.png",
        ),
    ]

    for metric_field, ylabel, title, filename in plot_specs:
        points_by_encoding = synthetic_plot_points(rows, metric_field)
        if not points_by_encoding:
            continue

        fig, ax = plt.subplots(figsize=(9.4, 5.4))
        fig.patch.set_facecolor(PLOT_BACKGROUND)
        ax.set_facecolor(AXIS_BACKGROUND)
        plotted_values: list[float] = []

        for encoding, points in points_by_encoding.items():
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            plotted_values.extend(y_values)
            ax.plot(
                x_values,
                y_values,
                marker=ENCODING_MARKERS.get(encoding, "o"),
                markersize=7.8,
                markeredgecolor="white",
                markeredgewidth=1.2,
                color=ENCODING_COLORS.get(encoding),
                linewidth=2.4,
                label=pretty_label(encoding),
            )

        x_ticks = sorted(
            {
                numeric_table_value(row.get("triple_count"))
                for row in rows
                if numeric_table_value(row.get("triple_count")) is not None
            }
        )
        ax.set_xscale("log", base=10)
        ax.set_xticks(x_ticks)
        if len(x_ticks) == 1:
            only_tick = x_ticks[0]
            ax.set_xlim(only_tick / 1.6, only_tick * 1.6)
        ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(compact_number))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(compact_number))
        ax.set_xlabel("Number of triples")
        ax.set_ylabel(ylabel)
        if plotted_values and min(plotted_values) >= 0:
            ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
        ax.grid(
            True,
            which="major",
            axis="both",
            color=GRID_COLOR,
            alpha=0.72,
            linewidth=0.8,
        )
        ax.grid(
            True,
            which="minor",
            axis="y",
            color=GRID_COLOR,
            alpha=0.25,
            linewidth=0.45,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9ca3af")
        ax.spines["bottom"].set_color("#9ca3af")
        ax.tick_params(colors="#1f2937", labelsize=10)
        ax.legend(
            title="Encoding",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            borderaxespad=0,
        )
        fig.tight_layout()
        output_path = figures_dir / filename
        fig.savefig(
            output_path,
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        created.append(output_path)

    return created


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


def probability_rows(counts: dict[str, int], shots: int) -> list[dict[str, Any]]:
    if shots < 1:
        return []
    return [
        {
            "outcome": outcome,
            "count": count,
            "probability": count / shots,
        }
        for outcome, count in sorted(counts.items())
    ]


def slugify(value: Any) -> str:
    text = str(value or "item").lower()
    cleaned = [
        char if char.isalnum() else "_"
        for char in text
    ]
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "item"


def histogram_filename(task: str, name: str) -> str:
    if task == "search_grover_lookup":
        return "hist_basis_lookup.png"
    if task == "entity_matching_swap_test":
        return "hist_entity_matching_swap_test.png"
    if task == "keyword_search_swap_test":
        return f"hist_keyword_search_{slugify(name)}.png"
    if task == "link_prediction_distance_estimation":
        return "hist_link_prediction_distance.png"
    if task == "multihop_phase_kickback":
        return (
            "hist_multihop_phase.png"
            if name == "x_quadrature"
            else f"hist_multihop_phase_{slugify(name)}.png"
        )
    if task == "schema_matching_qft":
        if name == "pattern_a":
            return "hist_schema_qft_pattern_a.png"
        if name == "pattern_b":
            return "hist_schema_qft_pattern_b.png"
        return f"hist_schema_qft_{slugify(name)}.png"
    return f"hist_{slugify(task)}_{slugify(name)}.png"


def result_count_sets(task: str, result: Any) -> list[tuple[str, dict[str, int]]]:
    if result is None:
        return []
    count_sets: list[tuple[str, dict[str, int]]] = []
    counts_by_entity = getattr(result, "counts_by_entity", None)
    if isinstance(counts_by_entity, dict):
        for entity, counts in counts_by_entity.items():
            count_sets.append((str(entity), dict(counts)))
        return count_sets
    if hasattr(result, "counts_x") and hasattr(result, "counts_y"):
        count_sets.append(("x_quadrature", dict(getattr(result, "counts_x"))))
        count_sets.append(("y_quadrature", dict(getattr(result, "counts_y"))))
        return count_sets
    if hasattr(result, "counts_pattern_a") and hasattr(result, "counts_pattern_b"):
        count_sets.append(("pattern_a", dict(getattr(result, "counts_pattern_a"))))
        count_sets.append(("pattern_b", dict(getattr(result, "counts_pattern_b"))))
        return count_sets
    counts = getattr(result, "counts", None)
    if isinstance(counts, dict):
        count_sets.append(("main", dict(counts)))
    return count_sets


def save_histogram_figure(
    counts: dict[str, int],
    *,
    title: str,
    output_path: Path,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not counts:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(sorted(counts))
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(5.8, 0.72 * len(labels)), 3.8))
    ax.bar(labels, values, color="#2f6f6d")
    ax.set_title(title)
    ax.set_xlabel("Measured bitstring")
    ax.set_ylabel("Counts")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def build_per_repetition_rows(
    encoding_rows: list[dict[str, Any]],
    usage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in encoding_rows:
        rows.append(
            {
                "group": row.get("group"),
                "experiment": row.get("experiment"),
                "repetition": row.get("repetition"),
                "index_mode": row.get("index_mode"),
                "backend": row.get("backend"),
                "status": row.get("status"),
                "num_qubits": row.get("num_qubits"),
                "circuit_depth": row.get("circuit_depth"),
                "gate_count": row.get("gate_count"),
                "transpiled_depth": row.get("transpiled_depth"),
                "transpiled_gate_count": row.get("transpiled_gate_count"),
                "task_time_seconds": row.get("task_time_seconds"),
                "state_preparation_time_seconds": row.get("preparation_time_seconds"),
                "notes": row.get("notes"),
            }
        )
    for row in usage_rows:
        rows.append(
            {
                field: row.get(field)
                for field in PER_REPETITION_FIELDS
            }
        )
    return rows


def score_derivation_rows(usage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in usage_rows:
        task = str(row.get("task") or "")
        result = row.get("result")
        repetition = row.get("repetition")
        shots = row.get("shots")
        if task == "search_grover_lookup":
            rows.append(
                {
                    "task": task,
                    "repetition": repetition,
                    "score_name": "target_success_probability",
                    "expected_value": 1.0,
                    "estimated_value": getattr(result, "success_probability", None),
                    "exact_or_classical_value": 1.0,
                    "formula": "target_bitstring_count / shots",
                    "counts_source": "counts",
                    "shots": shots,
                    "absolute_error": getattr(result, "absolute_error", None),
                    "relative_error": getattr(result, "relative_error", None),
                    "pass_threshold": getattr(result, "pass_threshold", None),
                    "pass_fail": getattr(result, "pass_fail", ""),
                    "threshold_reason": getattr(result, "threshold_reason", ""),
                }
            )
        elif task in {
            "entity_matching_swap_test",
            "link_prediction_distance_estimation",
        }:
            rows.append(
                {
                    "task": task,
                    "repetition": repetition,
                    "score_name": "swap_test_squared_similarity",
                    "expected_value": getattr(result, "classical_similarity", None),
                    "estimated_value": getattr(result, "estimated_similarity", None),
                    "exact_or_classical_value": getattr(
                        result,
                        "classical_similarity",
                        None,
                    ),
                    "formula": "max(0, min(1, 2 * Pr(ancilla=0) - 1))",
                    "counts_source": "counts",
                    "shots": shots,
                    "absolute_error": getattr(result, "absolute_error", None),
                    "relative_error": getattr(result, "relative_error", None),
                    "pass_threshold": getattr(result, "pass_threshold", None),
                    "pass_fail": getattr(result, "pass_fail", ""),
                    "threshold_reason": getattr(result, "threshold_reason", ""),
                }
            )
        elif task == "keyword_search_swap_test":
            estimated_scores = getattr(result, "estimated_scores", {})
            classical_scores = getattr(result, "classical_scores", {})
            absolute_errors = getattr(result, "absolute_errors", {})
            for entity, estimated_score in estimated_scores.items():
                rows.append(
                    {
                        "task": task,
                        "repetition": repetition,
                        "score_name": f"keyword_score:{entity}",
                        "expected_value": classical_scores.get(entity),
                        "estimated_value": estimated_score,
                        "exact_or_classical_value": classical_scores.get(entity),
                        "formula": "max(0, min(1, 2 * Pr(ancilla=0) - 1))",
                        "counts_source": f"counts_by_entity:{entity}",
                        "shots": shots,
                        "absolute_error": absolute_errors.get(entity),
                        "relative_error": "",
                        "pass_threshold": getattr(result, "pass_threshold", None),
                        "pass_fail": getattr(result, "pass_fail", ""),
                        "threshold_reason": getattr(result, "threshold_reason", ""),
                    }
                )
        elif task == "multihop_phase_kickback":
            rows.append(
                {
                    "task": task,
                    "repetition": repetition,
                    "score_name": "wrapped_phase_estimate",
                    "expected_value": getattr(result, "expected_composed_phase", None),
                    "estimated_value": getattr(result, "estimated_phase", None),
                    "exact_or_classical_value": getattr(
                        result,
                        "observed_composed_phase",
                        None,
                    ),
                    "formula": "atan2(2*Pr_y(0)-1, 2*Pr_x(0)-1) modulo 2*pi",
                    "counts_source": "counts_x/counts_y",
                    "shots": shots,
                    "absolute_error": getattr(result, "shot_phase_error", None),
                    "relative_error": getattr(result, "relative_error", None),
                    "pass_threshold": getattr(result, "pass_threshold", None),
                    "pass_fail": getattr(result, "pass_fail", ""),
                    "threshold_reason": getattr(result, "threshold_reason", ""),
                }
            )
        elif task == "schema_matching_qft":
            rows.append(
                {
                    "task": task,
                    "repetition": repetition,
                    "score_name": "measured_qft_distribution_similarity",
                    "expected_value": getattr(
                        result,
                        "exact_distribution_similarity",
                        None,
                    ),
                    "estimated_value": getattr(
                        result,
                        "measured_distribution_similarity",
                        None,
                    ),
                    "exact_or_classical_value": getattr(
                        result,
                        "fourier_pattern_similarity",
                        None,
                    ),
                    "formula": "cosine_similarity(measured_probabilities_a, measured_probabilities_b)",
                    "counts_source": "counts_pattern_a/counts_pattern_b",
                    "shots": shots,
                    "absolute_error": getattr(result, "absolute_error", None),
                    "relative_error": getattr(result, "relative_error", None),
                    "pass_threshold": getattr(result, "pass_threshold", None),
                    "pass_fail": getattr(result, "pass_fail", ""),
                    "threshold_reason": getattr(result, "threshold_reason", ""),
                }
            )
    return rows


def validation_threshold_rows(usage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task": row.get("task"),
            "repetition": row.get("repetition"),
            "expected_value": row.get("expected_value"),
            "estimated_value": row.get("estimated_value"),
            "absolute_error": row.get("absolute_error"),
            "relative_error": row.get("relative_error"),
            "pass_threshold": row.get("pass_threshold"),
            "pass_fail": row.get("pass_fail"),
            "threshold_reason": row.get("threshold_reason"),
        }
        for row in usage_rows
    ]


def suitability_metric_rows(usage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in usage_rows:
        task_label, encoding, _method = TABLE1_TASK_LABELS.get(
            str(row.get("task")),
            (str(row.get("task") or ""), "", ""),
        )
        rows.append(
            {
                "task": task_label,
                "encoding": encoding,
                "measurement_mode": row.get("measurement_mode"),
                "qubits": row.get("num_qubits"),
                "state_preparation_time_seconds": row.get(
                    "state_preparation_time_seconds"
                ),
                "circuit_construction_time_seconds": "",
                "simulation_time_seconds": row.get("simulation_time_seconds"),
                "measurement_time_seconds": row.get("measurement_time_seconds"),
                "readout_decoding_time_seconds": row.get(
                    "readout_decoding_time_seconds"
                ),
                "circuit_depth": row.get("circuit_depth"),
                "gate_count": row.get("gate_count"),
                "transpiled_depth": row.get("transpiled_depth"),
                "transpiled_gate_count": row.get("transpiled_gate_count"),
                "shots": row.get("shots"),
                "error_against_exact_or_classical": row.get("absolute_error"),
                "pass_fail": row.get("pass_fail"),
                "notes": row.get("notes"),
                "claim_scope": row.get("claim_scope"),
            }
        )
    return rows


def write_simple_markdown_table(
    rows: list[dict[str, Any]],
    fields: list[str],
    output_path: Path,
    *,
    title: str,
    intro: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        intro,
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(field, "")) for field in fields)
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_schema_matching_artifacts(
    usage_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path | None]:
    schema_rows = [
        row
        for row in usage_rows
        if row.get("task") == "schema_matching_qft"
    ]
    schema_dir = output_dir / "schema_matching"
    schema_dir.mkdir(parents=True, exist_ok=True)
    if not schema_rows:
        return {
            "schema_phase_assignments": None,
            "schema_matching_raw": None,
            "schema_matching_negative_control": None,
        }

    assignment_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    raw_payload: list[dict[str, Any]] = []
    for row in schema_rows:
        result = row.get("result")
        result_payload = json_safe(result)
        raw_payload.append(
            {
                "repetition": row.get("repetition"),
                "result": result_payload,
            }
        )
        strategy_results = getattr(result, "strategy_results", {})
        for strategy, strategy_payload in strategy_results.items():
            for assignment in strategy_payload.get("phase_assignments", []):
                assignment_row = dict(assignment)
                assignment_row["repetition"] = row.get("repetition")
                assignment_rows.append(assignment_row)
        negative_control = getattr(result, "negative_control", {})
        negative_rows.append(
            {
                "repetition": row.get("repetition"),
                "strategy": "negative_control",
                "measured_distribution_similarity": negative_control.get(
                    "measured_distribution_similarity"
                ),
                "exact_distribution_similarity": negative_control.get(
                    "exact_distribution_similarity"
                ),
                "absolute_error": negative_control.get("absolute_error"),
                "assumption": "no semantic prior is available",
            }
        )

    assignments_path = schema_dir / "schema_phase_assignments.csv"
    negative_path = schema_dir / "schema_matching_negative_control.csv"
    raw_path = schema_dir / "schema_matching_raw.json"
    write_csv(
        assignment_rows,
        assignments_path,
        [
            "repetition",
            "strategy",
            "relation",
            "canonical_relation",
            "phase",
            "assumption",
        ],
    )
    write_csv(
        negative_rows,
        negative_path,
        [
            "repetition",
            "strategy",
            "measured_distribution_similarity",
            "exact_distribution_similarity",
            "absolute_error",
            "assumption",
        ],
    )
    write_json({"schema_matching_results": raw_payload}, raw_path)
    return {
        "schema_phase_assignments": assignments_path,
        "schema_matching_raw": raw_path,
        "schema_matching_negative_control": negative_path,
    }


def write_chapter9_validation_report(
    *,
    output_path: Path,
    args: argparse.Namespace,
    usage_rows: list[dict[str, Any]],
    artifact_paths: dict[str, Any],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_rows = validation_threshold_rows(usage_rows)
    passed = sum(1 for row in threshold_rows if row.get("pass_fail") == "pass")
    failed = sum(1 for row in threshold_rows if row.get("pass_fail") == "fail")
    lines = [
        "# Chapter 9 Validation Summary",
        "",
        (
            "This report records implementation-level validation evidence. "
            "It does not claim quantum advantage."
        ),
        "",
        "## Settings",
        "",
        f"- Shots: `{args.shots}`",
        f"- Repetitions: `{args.repetitions}`",
        f"- Index mode: `{args.index_mode}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Threshold Summary",
        "",
        f"- Passed task rows: `{passed}`",
        f"- Failed task rows: `{failed}`",
        "",
        "## Task Evidence",
        "",
    ]
    for row in usage_rows:
        result = row.get("result")
        lines.extend(
            [
                f"### {row.get('task')}",
                "",
                f"- Repetition: `{row.get('repetition')}`",
                f"- Measurement mode: `{row.get('measurement_mode')}`",
                f"- Shots: `{row.get('shots')}`",
                f"- Expected value: `{row.get('expected_value')}`",
                f"- Estimated value: `{row.get('estimated_value')}`",
                f"- Absolute error: `{row.get('absolute_error')}`",
                f"- Threshold: `{row.get('pass_threshold')}`",
                f"- Pass/fail: `{row.get('pass_fail')}`",
                f"- Claim scope: `{row.get('claim_scope')}`",
                f"- Raw result fields: `{', '.join(sorted(json_safe(result).keys())) if result else ''}`",
                "",
            ]
        )
    lines.extend(["## Artifact Paths", ""])
    for key, value in sorted(artifact_paths.items()):
        if isinstance(value, list):
            for item in value:
                lines.append(f"- `{key}`: `{item}`")
        elif value not in (None, ""):
            lines.append(f"- `{key}`: `{value}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_validation_artifacts(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    encoding_rows: list[dict[str, Any]],
    usage_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[Path]]:
    raw_counts_dir = output_dir / "raw_counts"
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    raw_counts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Path] = []
    count_artifacts: list[Path] = []
    for row in usage_rows:
        task = str(row.get("task") or "")
        repetition = int(row.get("repetition") or 0)
        shots = int(row.get("shots") or 0)
        result = row.get("result")
        for name, counts in result_count_sets(task, result):
            stem = f"{slugify(task)}_rep{repetition}_{slugify(name)}"
            counts_path = raw_counts_dir / f"{stem}_counts.json"
            probabilities_path = raw_counts_dir / f"{stem}_probabilities.csv"
            write_json(
                {
                    "task": task,
                    "repetition": repetition,
                    "count_set": name,
                    "shots": shots,
                    "counts": counts,
                    "probabilities": {
                        item["outcome"]: item["probability"]
                        for item in probability_rows(counts, shots)
                    },
                },
                counts_path,
            )
            write_csv(
                probability_rows(counts, shots),
                probabilities_path,
                ["outcome", "count", "probability"],
            )
            count_artifacts.extend([counts_path, probabilities_path])
            figure_path = figures_dir / histogram_filename(task, name)
            created = save_histogram_figure(
                counts,
                title=f"{task} ({name}) measurement counts",
                output_path=figure_path,
            )
            if created is not None and created not in figures:
                figures.append(created)

    per_repetition_path = output_dir / "chapter9_per_repetition.csv"
    derivation_path = output_dir / "chapter9_score_derivations.csv"
    thresholds_path = output_dir / "chapter9_validation_thresholds.csv"
    suitability_csv_path = reports_dir / "encoding_suitability_metrics.csv"
    suitability_md_path = reports_dir / "encoding_suitability_metrics.md"
    validation_md_path = reports_dir / "chapter9_validation.md"
    validation_json_path = reports_dir / "chapter9_validation.json"

    per_repetition = build_per_repetition_rows(encoding_rows, usage_rows)
    derivations = score_derivation_rows(usage_rows)
    thresholds = validation_threshold_rows(usage_rows)
    suitability_rows = suitability_metric_rows(usage_rows)
    schema_paths = write_schema_matching_artifacts(usage_rows, output_dir)

    write_csv(per_repetition, per_repetition_path, PER_REPETITION_FIELDS)
    write_csv(derivations, derivation_path, SCORE_DERIVATION_FIELDS)
    write_csv(thresholds, thresholds_path, THRESHOLD_FIELDS)
    write_csv(suitability_rows, suitability_csv_path, SUITABILITY_FIELDS)
    write_simple_markdown_table(
        suitability_rows,
        [
            "task",
            "encoding",
            "measurement_mode",
            "qubits",
            "shots",
            "error_against_exact_or_classical",
            "pass_fail",
            "claim_scope",
        ],
        suitability_md_path,
        title="Encoding Suitability Metrics",
        intro=(
            "Relative implementation-level metrics for the implemented tasks. "
            "These rows compare encodings within this codebase and do not "
            "claim absolute superiority or quantum advantage."
        ),
    )

    artifact_paths: dict[str, Any] = {
        "per_repetition": per_repetition_path,
        "score_derivations": derivation_path,
        "validation_thresholds": thresholds_path,
        "raw_counts_dir": raw_counts_dir,
        "raw_count_artifacts": count_artifacts,
        "encoding_suitability_metrics_csv": suitability_csv_path,
        "encoding_suitability_metrics_md": suitability_md_path,
        **schema_paths,
    }
    write_chapter9_validation_report(
        output_path=validation_md_path,
        args=args,
        usage_rows=usage_rows,
        artifact_paths=artifact_paths,
    )
    write_json(
        {
            "settings": vars(args),
            "thresholds": thresholds,
            "score_derivations": derivations,
            "suitability_metrics": suitability_rows,
            "artifact_paths": artifact_paths,
        },
        validation_json_path,
    )
    artifact_paths["chapter9_validation_md"] = validation_md_path
    artifact_paths["chapter9_validation_json"] = validation_json_path
    return artifact_paths, figures


def collect_generated_tables(output_files: dict[str, Any]) -> list[str]:
    table_keys = [
        "table3_encoding_process",
        "table3_encoding_process_tex",
        "table4_usage_tasks",
        "table4_usage_tasks_tex",
        "table5_validation_metrics",
        "table5_validation_metrics_tex",
        "table6_circuit_statistics",
        "table6_circuit_statistics_tex",
        "table7_synthetic_results",
        "table7_synthetic_results_tex",
        "table8_real_kg_results",
        "table8_real_kg_results_tex",
        "chapter9_per_repetition",
        "chapter9_score_derivations",
        "chapter9_validation_thresholds",
        "encoding_suitability_metrics_csv",
        "schema_phase_assignments",
        "schema_matching_negative_control",
    ]
    return [
        str(output_files[key])
        for key in table_keys
        if output_files.get(key) not in (None, "")
    ]


def collect_generated_plots(output_files: dict[str, Any]) -> list[str]:
    return [
        str(path)
        for path in output_files.get("figures", [])
        if path not in (None, "")
    ]


def collect_generated_data_files(output_files: dict[str, Any]) -> list[str]:
    data_keys = [
        "raw_results",
        "synthetic_raw_results",
        "real_kg_raw_results",
        "environment",
        "run_summary",
        "raw_counts_dir",
        "encoding_suitability_metrics_md",
        "chapter9_validation_md",
        "chapter9_validation_json",
        "schema_matching_raw",
    ]
    return [
        str(output_files[key])
        for key in data_keys
        if output_files.get(key) not in (None, "")
    ]


def collect_skipped_or_failed_experiments(
    raw_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    skipped_or_failed: list[dict[str, Any]] = []

    for row in raw_payload.get("encoding_process_rows", []):
        if row.get("status") != "success":
            skipped_or_failed.append(
                {
                    "section": "encoding_process",
                    "name": row.get("experiment"),
                    "encoding": row.get("experiment"),
                    "status": row.get("status"),
                    "reason": row.get("notes") or row.get("error_message") or "",
                }
            )

    for row in raw_payload.get("usage_task_rows", []):
        if row.get("status") != "success":
            skipped_or_failed.append(
                {
                    "section": "usage_task",
                    "name": row.get("task"),
                    "encoding": row.get("task"),
                    "status": row.get("status"),
                    "reason": row.get("notes") or row.get("error_message") or "",
                }
            )

    for section, rows in (
        ("synthetic_observations", raw_payload.get("table7_synthetic_results", [])),
        ("real_kg_observations", raw_payload.get("table8_real_kg_results", [])),
    ):
        for row in rows:
            if row.get("status") != "success":
                skipped_or_failed.append(
                    {
                        "section": section,
                        "dataset": row.get("dataset_name") or row.get("triple_count"),
                        "encoding": row.get("encoding"),
                        "status": row.get("status"),
                        "reason": row.get("notes") or "",
                    }
                )

    return skipped_or_failed


def environment_payload(
    args: argparse.Namespace,
    *,
    raw_payload: dict[str, Any] | None = None,
    run_summary_path: Path | None = None,
    command_line: list[str] | None = None,
) -> dict[str, Any]:
    qiskit_version = package_version("qiskit")
    qiskit_aer_version = package_version("qiskit-aer")
    numpy_version = package_version("numpy")
    rdflib_version = package_version("rdflib")
    ram_bytes = total_ram_bytes()
    output_files = raw_payload.get("output_files", {}) if raw_payload else {}
    synthetic_payload = raw_payload.get("synthetic_observations") if raw_payload else None
    real_payload = raw_payload.get("real_kg_observations") if raw_payload else None
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
        "exact_command_line": command_line or sys.argv,
        "command_line_arguments": vars(args),
        "arguments": vars(args),
        "random_seed": args.seed,
        "git_commit_hash": git_commit_hash(),
        "generated_tables": collect_generated_tables(output_files),
        "generated_plots": collect_generated_plots(output_files),
        "generated_data_files": collect_generated_data_files(output_files),
        "synthetic_sizes_used": (
            synthetic_payload.get("settings", {}).get("sizes", [])
            if synthetic_payload
            else []
        ),
        "real_kg_files_used": (
            real_payload.get("settings", {}).get("files", [])
            if real_payload
            else []
        ),
        "skipped_or_failed_experiments": (
            collect_skipped_or_failed_experiments(raw_payload)
            if raw_payload
            else []
        ),
        "run_summary": str(run_summary_path) if run_summary_path else None,
        "software_level_observation_note": (
            "Results are simulator-based software-level observations and do "
            "not show quantum advantage."
        ),
        "note": (
            "By default this runner executes the Chapter 9 six-triple "
            "running-example experiments. Optional --include-synthetic and "
            "--include-real runs add Chapter 9 software-level observations "
            "using the existing scalability helpers."
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


def markdown_list(items: list[Any], empty_text: str = "None") -> list[str]:
    if not items:
        return [f"- {empty_text}"]
    return [f"- {item}" for item in items]


def skipped_failed_markdown(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None"]
    grouped: dict[tuple[str, str | None, str, str], int] = {}
    lines: list[str] = []
    for item in items:
        label = (
            item.get("dataset")
            or item.get("name")
            or item.get("section")
            or "experiment"
        )
        encoding = item.get("encoding")
        status = item.get("status") or "unknown"
        reason = item.get("reason") or "No reason recorded."
        key = (
            str(label),
            str(encoding) if encoding else None,
            str(status),
            str(reason),
        )
        grouped[key] = grouped.get(key, 0) + 1
    for (label, encoding, status, reason), count in grouped.items():
        suffix = f" ({encoding})" if encoding else ""
        count_suffix = f" ({count} rows)" if count > 1 else ""
        lines.append(f"- {label}{suffix}: {status}{count_suffix}; {reason}")
    return lines


def write_run_summary(
    *,
    output_path: Path,
    args: argparse.Namespace,
    raw_payload: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_files = raw_payload.get("output_files", {})
    generated_tables = collect_generated_tables(output_files)
    generated_plots = collect_generated_plots(output_files)
    generated_data_files = collect_generated_data_files(output_files)
    synthetic_payload = raw_payload.get("synthetic_observations")
    real_payload = raw_payload.get("real_kg_observations")
    skipped_or_failed = collect_skipped_or_failed_experiments(raw_payload)
    os_payload = environment.get("operating_system") or {}
    cpu_payload = environment.get("cpu") or {}

    lines = [
        "# Chapter 9 Run Summary",
        "",
        (
            "Results are simulator-based software-level observations and do "
            "not show quantum advantage."
        ),
        "",
        "## Command",
        "",
        "```text",
        " ".join(str(part) for part in environment.get("exact_command_line", [])),
        "```",
        "",
        "## Command-Line Arguments",
        "",
    ]
    for key, value in sorted(vars(args).items()):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Generated Tables",
            "",
            *markdown_list(generated_tables),
            "",
            "## Generated Plots",
            "",
            *markdown_list(generated_plots),
            "",
            "## Generated Data Files",
            "",
            *markdown_list(generated_data_files),
            "",
            "## Section 9.2 Synthetic Sizes",
            "",
        ]
    )
    if synthetic_payload:
        synthetic_sizes = synthetic_payload.get("settings", {}).get("sizes", [])
        lines.extend(markdown_list([str(size) for size in synthetic_sizes]))
    else:
        lines.extend(markdown_list([], empty_text="Not run"))

    lines.extend(["", "## Section 9.3 Real KG Files", ""])
    if real_payload:
        real_files = real_payload.get("settings", {}).get("files", [])
        lines.extend(markdown_list(real_files))
    else:
        lines.extend(markdown_list([], empty_text="Not run"))

    lines.extend(
        [
            "",
            "## Skipped Or Failed Experiments",
            "",
            *skipped_failed_markdown(skipped_or_failed),
            "",
            "## Environment",
            "",
            f"- Timestamp UTC: `{environment.get('timestamp_utc')}`",
            f"- Hostname: `{environment.get('hostname')}`",
            f"- Python: `{environment.get('python_version')}`",
            f"- Qiskit: `{environment.get('qiskit_version')}`",
            f"- Qiskit Aer: `{environment.get('qiskit_aer_version')}`",
            f"- NumPy: `{environment.get('numpy_version')}`",
            f"- rdflib: `{environment.get('rdflib_version')}`",
            f"- OS: `{os_payload.get('platform') or environment.get('platform')}`",
            f"- CPU: `{cpu_payload.get('processor') or cpu_payload.get('machine')}`",
            f"- RAM: `{format_ram(environment.get('total_ram_bytes'))}`",
            f"- Git commit: `{environment.get('git_commit_hash')}`",
            f"- Random seed: `{environment.get('random_seed')}`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


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
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help=(
            "Also run Section 9.2 synthetic KG software-level observations "
            "using the existing scalability infrastructure."
        ),
    )
    parser.add_argument(
        "--synthetic-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SYNTHETIC_SIZES),
        help="Synthetic triple counts for Section 9.2 observations.",
    )
    parser.add_argument(
        "--synthetic-repetitions",
        type=int,
        default=3,
        help="Repetitions for each synthetic size and encoding.",
    )
    parser.add_argument(
        "--include-real",
        action="store_true",
        help=(
            "Also run real KG software-level observations using existing real "
            "KG examples and parsing helpers."
        ),
    )
    parser.add_argument(
        "--real-kg-files",
        nargs="+",
        default=None,
        help=(
            "Real KG files to observe. Relative names are resolved against the "
            "repository root and data/real_kgs. Defaults to existing examples."
        ),
    )
    parser.add_argument(
        "--max-real-triples",
        type=int,
        default=500,
        help="Deterministically truncate each real KG to this many triples.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.shots < 1:
        raise ValueError("--shots must be at least 1.")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1.")
    if args.synthetic_repetitions < 1:
        raise ValueError("--synthetic-repetitions must be at least 1.")
    if any(size < 1 for size in args.synthetic_sizes):
        raise ValueError("--synthetic-sizes values must be at least 1.")
    if args.max_real_triples < 1:
        raise ValueError("--max-real-triples must be at least 1.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    validate_args(args)
    command_line = (
        [sys.executable, *sys.argv]
        if argv is None
        else [sys.executable, str(Path(__file__).resolve()), *argv]
    )

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
    table5_rows = validation_metric_table_rows(usage_rows)
    table6_rows = circuit_statistics_table_rows(usage_summary)
    synthetic_payload = (
        run_chapter9_synthetic_experiments(args, output_dir)
        if args.include_synthetic
        else None
    )
    real_payload = run_chapter9_real_kg_experiments(args) if args.include_real else None
    table7_rows = synthetic_payload["rows"] if synthetic_payload else []
    table7_latex_rows = synthetic_latex_table_rows(table7_rows) if table7_rows else []
    table8_rows = real_payload["rows"] if real_payload else []
    table8_latex_rows = real_latex_table_rows(table8_rows) if table8_rows else []

    output_dir.mkdir(parents=True, exist_ok=True)
    table3_path = output_dir / "table3_encoding_process.csv"
    table4_path = output_dir / "table4_usage_tasks.csv"
    table5_path = output_dir / "table5_validation_metrics.csv"
    table6_path = output_dir / "table6_circuit_statistics.csv"
    table7_path = output_dir / "table7_synthetic_results.csv"
    table8_path = output_dir / "table8_real_kg_results.csv"
    table3_tex_path = output_dir / "table3_encoding_process.tex"
    table4_tex_path = output_dir / "table4_usage_tasks.tex"
    table5_tex_path = output_dir / "table5_validation_metrics.tex"
    table6_tex_path = output_dir / "table6_circuit_statistics.tex"
    table7_tex_path = output_dir / "table7_synthetic_results.tex"
    table8_tex_path = output_dir / "table8_real_kg_results.tex"
    raw_path = output_dir / "chapter9_raw_results.json"
    synthetic_raw_path = output_dir / "synthetic_raw_results.json"
    real_raw_path = output_dir / "real_kg_raw_results.json"
    env_path = output_dir / "environment.json"
    run_summary_path = output_dir / "RUN_SUMMARY.md"

    if args.save_csv:
        write_csv(table3_rows, table3_path, ENCODING_TABLE_FIELDS)
        write_csv(table4_rows, table4_path, USAGE_TABLE_FIELDS)
        write_csv(table5_rows, table5_path, VALIDATION_TABLE_FIELDS)
        write_csv(table6_rows, table6_path, TABLE6_CIRCUIT_FIELDS)
        if synthetic_payload:
            write_csv(table7_rows, table7_path, SYNTHETIC_TABLE_FIELDS)
        if real_payload:
            write_csv(table8_rows, table8_path, REAL_TABLE_FIELDS)
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
            table5_rows,
            table5_tex_path,
            VALIDATION_LATEX_FIELDS,
            caption="Chapter 9 validation metrics for running-example tasks.",
            label="tab:chapter9-validation-metrics",
        )
        write_latex_table(
            table6_rows,
            table6_tex_path,
            TABLE6_CIRCUIT_FIELDS,
            caption="Chapter 9 circuit statistics for running-example tasks.",
            label="tab:chapter9-circuit-statistics",
        )
        if synthetic_payload:
            write_latex_table(
                table7_latex_rows,
                table7_tex_path,
                SYNTHETIC_LATEX_FIELDS,
                caption=(
                    "Section 9.2 synthetic KG software-level observations; "
                    "no quantum-advantage claim."
                ),
                label="tab:chapter9-synthetic-observations",
            )
        if real_payload:
            write_latex_table(
                table8_latex_rows,
                table8_tex_path,
                REAL_LATEX_FIELDS,
                caption=(
                    "Real KG software-level observations; no quantum-advantage "
                    "claim."
                ),
                label="tab:chapter9-real-kg-observations",
            )

    figures = maybe_write_plots(
        encoding_summary=encoding_summary,
        usage_summary=usage_summary,
        table3_rows=table3_rows,
        table4_rows=table4_rows,
        figures_dir=output_dir / "figures",
        index_mode=args.index_mode,
    )
    validation_artifact_paths, validation_figures = write_validation_artifacts(
        output_dir=output_dir,
        args=args,
        encoding_rows=encoding_rows,
        usage_rows=usage_rows,
    )
    figures.extend(validation_figures)
    if synthetic_payload:
        figures.extend(
            write_synthetic_observation_plots(
                table7_rows,
                output_dir / "figures",
            )
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
        "table5_validation_metrics": table5_rows,
        "table6_circuit_statistics": table6_rows,
        "synthetic_observations": synthetic_payload,
        "table7_synthetic_results": table7_rows,
        "real_kg_observations": real_payload,
        "table8_real_kg_results": table8_rows,
        "output_files": {
            "table3_encoding_process": table3_path,
            "table4_usage_tasks": table4_path,
            "table5_validation_metrics": table5_path,
            "table6_circuit_statistics": table6_path,
            "table7_synthetic_results": table7_path if synthetic_payload else None,
            "table8_real_kg_results": table8_path if real_payload else None,
            "table3_encoding_process_tex": table3_tex_path,
            "table4_usage_tasks_tex": table4_tex_path,
            "table5_validation_metrics_tex": table5_tex_path,
            "table6_circuit_statistics_tex": table6_tex_path,
            "table7_synthetic_results_tex": (
                table7_tex_path if synthetic_payload else None
            ),
            "table8_real_kg_results_tex": table8_tex_path if real_payload else None,
            "raw_results": raw_path,
            "synthetic_raw_results": synthetic_raw_path if synthetic_payload else None,
            "real_kg_raw_results": real_raw_path if real_payload else None,
            "environment": env_path,
            "run_summary": run_summary_path,
            "figures": figures,
            "chapter9_per_repetition": validation_artifact_paths.get(
                "per_repetition"
            ),
            "chapter9_score_derivations": validation_artifact_paths.get(
                "score_derivations"
            ),
            "chapter9_validation_thresholds": validation_artifact_paths.get(
                "validation_thresholds"
            ),
            "raw_counts_dir": validation_artifact_paths.get("raw_counts_dir"),
            "encoding_suitability_metrics_csv": validation_artifact_paths.get(
                "encoding_suitability_metrics_csv"
            ),
            "encoding_suitability_metrics_md": validation_artifact_paths.get(
                "encoding_suitability_metrics_md"
            ),
            "chapter9_validation_md": validation_artifact_paths.get(
                "chapter9_validation_md"
            ),
            "chapter9_validation_json": validation_artifact_paths.get(
                "chapter9_validation_json"
            ),
            "schema_phase_assignments": validation_artifact_paths.get(
                "schema_phase_assignments"
            ),
            "schema_matching_raw": validation_artifact_paths.get(
                "schema_matching_raw"
            ),
            "schema_matching_negative_control": validation_artifact_paths.get(
                "schema_matching_negative_control"
            ),
        },
    }
    environment = environment_payload(
        args,
        raw_payload=raw_payload,
        run_summary_path=run_summary_path,
        command_line=command_line,
    )
    if args.save_json:
        write_json(raw_payload, raw_path)
        if synthetic_payload:
            write_json(synthetic_payload, synthetic_raw_path)
        if real_payload:
            write_json(real_payload, real_raw_path)
    write_json(environment, env_path)
    write_run_summary(
        output_path=run_summary_path,
        args=args,
        raw_payload=raw_payload,
        environment=environment,
    )

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
    if synthetic_payload:
        print(
            "  Synthetic software-level observation rows: "
            f"{len(table7_rows)}"
        )
    if real_payload:
        print(f"  Real KG software-level observation rows: {len(table8_rows)}")
    print(f"  Output directory: {output_dir}")
    if args.save_csv:
        print(f"  Table 3 CSV: {table3_path}")
        print(f"  Table 4 CSV: {table4_path}")
        print(f"  Table 5 CSV: {table5_path}")
        print(f"  Table 6 CSV: {table6_path}")
        if synthetic_payload:
            print(f"  Table 7 CSV: {table7_path}")
        if real_payload:
            print(f"  Table 8 CSV: {table8_path}")
        print(f"  Table 3 LaTeX: {table3_tex_path}")
        print(f"  Table 4 LaTeX: {table4_tex_path}")
        print(f"  Table 5 LaTeX: {table5_tex_path}")
        print(f"  Table 6 LaTeX: {table6_tex_path}")
        if synthetic_payload:
            print(f"  Table 7 LaTeX: {table7_tex_path}")
        if real_payload:
            print(f"  Table 8 LaTeX: {table8_tex_path}")
    if args.save_json:
        print(f"  Raw JSON: {raw_path}")
        if synthetic_payload:
            print(f"  Synthetic raw JSON: {synthetic_raw_path}")
        if real_payload:
            print(f"  Real KG raw JSON: {real_raw_path}")
    print(f"  Environment JSON: {env_path}")
    print(f"  Run summary: {run_summary_path}")
    if figures:
        print(f"  Figures: {output_dir / 'figures'}")
    if args.include_synthetic or args.include_real:
        print(
            "  Note: Chapter 9 software-level observations reused existing "
            "scalability helpers; old standalone scalability sweeps were not run."
        )
    else:
        print("  Note: old scalability experiments were not run.")

    return raw_payload


if __name__ == "__main__":
    main()
