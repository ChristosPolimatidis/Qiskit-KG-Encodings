from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from qiskit import QuantumCircuit

from src.models import TripleRecord
from src.running_example import (
    PAPER_INDEX_DIMENSION,
    SEQUENTIAL_INDEX_DIMENSION,
    get_predicate_phase_map,
    get_running_example_indices,
)


PHASE_MODES = ("presence_phase", "predicate_phase")
INDEX_MODES = ("sequential", "paper")
PREDICATE_SUPPORT_MODES = ("triples", "full")
MIXING_MODES = (None, "hadamard", "qft")


@dataclass(frozen=True, slots=True)
class PaperPhaseEncodingResult:
    """Result bundle for the Chapter 9 phase-encoding experiments."""

    statevector: np.ndarray
    circuit: QuantumCircuit | None
    num_qubits: int
    phase_map: dict[str, float]
    marked_indices: list[int]
    preparation_time_seconds: float
    measurement_counts: dict[str, int]
    initial_statevector: np.ndarray
    measurement_statevector: np.ndarray
    measurement_probabilities: dict[str, float]
    index_phase_map: dict[int, float]
    index_mode: str
    phase_mode: str
    support_mode: str
    mixing: str | None


def index_space_dimension(index_mode: str) -> int:
    """Return the basis-state dimension required by an index mode."""

    if index_mode == "sequential":
        return SEQUENTIAL_INDEX_DIMENSION
    if index_mode == "paper":
        return PAPER_INDEX_DIMENSION
    raise ValueError("Unsupported index mode. Use 'sequential' or 'paper'.")


def index_space_qubits(index_mode: str) -> int:
    """Return the number of qubits required by an index mode."""

    return int(math.log2(index_space_dimension(index_mode)))


def _running_example_indices(index_mode: str) -> dict[TripleRecord, int]:
    if index_mode not in INDEX_MODES:
        raise ValueError("Unsupported index mode. Use 'sequential' or 'paper'.")
    return get_running_example_indices(mode=index_mode)


def _uniform_full_statevector(dimension: int) -> np.ndarray:
    return np.full(dimension, 1 / math.sqrt(dimension), dtype=complex)


def _uniform_triple_statevector(
    dimension: int,
    marked_indices: list[int],
) -> np.ndarray:
    statevector = np.zeros(dimension, dtype=complex)
    amplitude = 1 / math.sqrt(len(marked_indices))
    for index in marked_indices:
        statevector[index] = amplitude
    return statevector


def _phase_angles_vector(
    dimension: int,
    index_phase_map: dict[int, float],
) -> np.ndarray:
    phase_angles = np.zeros(dimension, dtype=float)
    for index, phase in index_phase_map.items():
        phase_angles[index] = phase
    return phase_angles


def _build_circuit(
    initial_statevector: np.ndarray,
    phase_angles: np.ndarray,
    mixing: str | None,
) -> QuantumCircuit | None:
    try:
        num_qubits = int(math.log2(len(initial_statevector)))
        circuit = QuantumCircuit(num_qubits, name="PaperPhaseEncoding")
        circuit.initialize(initial_statevector, circuit.qubits)
        circuit.unitary(
            np.diag(np.exp(1j * phase_angles)),
            circuit.qubits,
            label="PhaseMap",
        )
        if mixing == "hadamard":
            circuit.h(range(num_qubits))
        elif mixing == "qft":
            from qiskit.circuit.library import QFT

            circuit.compose(QFT(num_qubits, do_swaps=False), inplace=True)
        return circuit
    except Exception:
        return None


def _hadamard_matrix(num_qubits: int) -> np.ndarray:
    matrix = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    result = np.array([[1]], dtype=complex)
    for _ in range(num_qubits):
        result = np.kron(result, matrix)
    return result


def _qft_matrix(dimension: int) -> np.ndarray:
    indices = np.arange(dimension)
    omega = np.exp(2j * math.pi / dimension)
    return omega ** np.outer(indices, indices) / math.sqrt(dimension)


def _apply_mixing(
    statevector: np.ndarray,
    mixing: str | None,
    num_qubits: int,
) -> np.ndarray:
    if mixing is None:
        return statevector.copy()
    if mixing == "hadamard":
        return _hadamard_matrix(num_qubits) @ statevector
    if mixing == "qft":
        return _qft_matrix(len(statevector)) @ statevector
    raise ValueError("Unsupported mixing mode. Use None, 'hadamard', or 'qft'.")


def _probabilities(statevector: np.ndarray) -> np.ndarray:
    return np.abs(statevector) ** 2


def _probabilities_dict(statevector: np.ndarray, num_qubits: int) -> dict[str, float]:
    probabilities = _probabilities(statevector)
    return {
        format(index, f"0{num_qubits}b"): float(probability)
        for index, probability in enumerate(probabilities)
        if probability > 1e-15
    }


def _deterministic_counts(
    statevector: np.ndarray,
    num_qubits: int,
    shots: int,
) -> dict[str, int]:
    if shots < 1:
        raise ValueError("shots must be at least 1.")

    probabilities = _probabilities(statevector)
    scaled_counts = probabilities * shots
    counts = np.floor(scaled_counts).astype(int)
    remainder = int(shots - counts.sum())

    if remainder > 0:
        fractional_parts = scaled_counts - counts
        for index in np.argsort(-fractional_parts)[:remainder]:
            counts[index] += 1

    return {
        format(index, f"0{num_qubits}b"): int(count)
        for index, count in enumerate(counts)
        if count > 0
    }


def _build_result(
    *,
    phase_mode: str,
    index_mode: str,
    support_mode: str,
    phase_map: dict[str, float],
    index_phase_map: dict[int, float],
    initial_statevector: np.ndarray,
    mixing: str | None,
    shots: int,
    start_time: float,
) -> PaperPhaseEncodingResult:
    dimension = len(initial_statevector)
    num_qubits = int(math.log2(dimension))
    phase_angles = _phase_angles_vector(
        dimension=dimension,
        index_phase_map=index_phase_map,
    )
    statevector = initial_statevector * np.exp(1j * phase_angles)
    circuit = _build_circuit(
        initial_statevector=initial_statevector,
        phase_angles=phase_angles,
        mixing=mixing,
    )
    preparation_time_seconds = time.perf_counter() - start_time

    measurement_statevector = _apply_mixing(
        statevector=statevector,
        mixing=mixing,
        num_qubits=num_qubits,
    )
    return PaperPhaseEncodingResult(
        statevector=statevector,
        circuit=circuit,
        num_qubits=num_qubits,
        phase_map=phase_map,
        marked_indices=sorted(index_phase_map),
        preparation_time_seconds=preparation_time_seconds,
        measurement_counts=_deterministic_counts(
            statevector=measurement_statevector,
            num_qubits=num_qubits,
            shots=shots,
        ),
        initial_statevector=initial_statevector,
        measurement_statevector=measurement_statevector,
        measurement_probabilities=_probabilities_dict(
            statevector=measurement_statevector,
            num_qubits=num_qubits,
        ),
        index_phase_map=dict(sorted(index_phase_map.items())),
        index_mode=index_mode,
        phase_mode=phase_mode,
        support_mode=support_mode,
        mixing=mixing,
    )


def presence_phase(
    *,
    index_mode: str = "sequential",
    mixing: str | None = None,
    shots: int = 2048,
) -> PaperPhaseEncodingResult:
    """Apply a sign flip to every running-example triple index."""

    start_time = time.perf_counter()
    triple_indices = _running_example_indices(index_mode=index_mode)
    dimension = index_space_dimension(index_mode)
    initial_statevector = _uniform_full_statevector(dimension)
    index_phase_map = {
        index: math.pi
        for index in triple_indices.values()
    }
    return _build_result(
        phase_mode="presence_phase",
        index_mode=index_mode,
        support_mode="full",
        phase_map={"existing_triple": math.pi},
        index_phase_map=index_phase_map,
        initial_statevector=initial_statevector,
        mixing=mixing,
        shots=shots,
        start_time=start_time,
    )


def predicate_phase(
    *,
    index_mode: str = "sequential",
    support_mode: str = "triples",
    mixing: str | None = None,
    shots: int = 2048,
    phase_map: dict[str, float] | None = None,
) -> PaperPhaseEncodingResult:
    """Apply deterministic predicate-specific phases to running-example triples."""

    if support_mode not in PREDICATE_SUPPORT_MODES:
        raise ValueError("Unsupported support mode. Use 'triples' or 'full'.")

    start_time = time.perf_counter()
    triple_indices = _running_example_indices(index_mode=index_mode)
    dimension = index_space_dimension(index_mode)
    marked_indices = sorted(triple_indices.values())
    if support_mode == "full":
        initial_statevector = _uniform_full_statevector(dimension)
    else:
        initial_statevector = _uniform_triple_statevector(
            dimension=dimension,
            marked_indices=marked_indices,
        )

    selected_phase_map = (
        get_predicate_phase_map() if phase_map is None else dict(phase_map)
    )
    index_phase_map = {
        index: selected_phase_map[triple.predicate]
        for triple, index in triple_indices.items()
    }
    return _build_result(
        phase_mode="predicate_phase",
        index_mode=index_mode,
        support_mode=support_mode,
        phase_map=selected_phase_map,
        index_phase_map=index_phase_map,
        initial_statevector=initial_statevector,
        mixing=mixing,
        shots=shots,
        start_time=start_time,
    )


def run_phase_encoding(
    phase_mode: str,
    *,
    index_mode: str = "sequential",
    support_mode: str = "triples",
    mixing: str | None = None,
    shots: int = 2048,
) -> PaperPhaseEncodingResult:
    """Dispatch a Chapter 9 paper-aligned phase-encoding run."""

    if phase_mode == "presence_phase":
        return presence_phase(index_mode=index_mode, mixing=mixing, shots=shots)
    if phase_mode == "predicate_phase":
        return predicate_phase(
            index_mode=index_mode,
            support_mode=support_mode,
            mixing=mixing,
            shots=shots,
        )
    raise ValueError("Unsupported phase mode. Use 'presence_phase' or 'predicate_phase'.")
