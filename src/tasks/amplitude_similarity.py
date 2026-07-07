from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.amplitude_encoding import next_power_of_two_length
from src.running_example import (
    EX,
    get_predicate_phase_map,
    get_running_example_triples,
)


DEFAULT_ENTITY_A = f"{EX}Aristotle"
DEFAULT_ENTITY_B = f"{EX}Athens"
DEFAULT_SHOTS = 4096
DEFAULT_SEED_SIMULATOR = 24680
DEFAULT_SWAP_TEST_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class AmplitudeSimilarityResult:
    """Result bundle for the Chapter 9 amplitude similarity swap-test task."""

    task_name: str
    entity_a: str
    entity_b: str
    feature_labels: list[str]
    vector_a: list[float]
    vector_b: list[float]
    normalized_vector_a: list[float]
    normalized_vector_b: list[float]
    counts: dict[str, int]
    probabilities: dict[str, float]
    task_time_seconds: float
    normalization_time_seconds: float
    state_preparation_time_seconds: float
    simulation_time_seconds: float
    measurement_time_seconds: float
    readout_decoding_time_seconds: float
    num_qubits: int
    vector_dimension: int
    nonzero_entries_a: int
    nonzero_entries_b: int
    dense_memory_bytes: int
    sparse_memory_bytes: int
    circuit_depth: int
    gate_count: int
    transpiled_depth: int
    transpiled_gate_count: int
    shots: int
    p0: float
    estimated_similarity: float
    classical_similarity: float
    absolute_error: float
    relative_error: float | None
    pass_threshold: float
    pass_fail: str
    threshold_reason: str
    measurement_mode: str
    claim_scope: str
    backend: str
    claim_note: str


def _operation_count(circuit: QuantumCircuit) -> int:
    return sum(int(count) for count in circuit.count_ops().values())


def _predicate_feature_labels() -> list[str]:
    return list(get_predicate_phase_map())


def entity_context_vector(
    entity: str,
    *,
    feature_labels: list[str] | None = None,
) -> np.ndarray:
    """Count predicates incident to an entity in the running example."""

    labels = _predicate_feature_labels() if feature_labels is None else feature_labels
    label_to_position = {
        label: position
        for position, label in enumerate(labels)
    }
    vector = np.zeros(len(labels), dtype=float)
    for triple in get_running_example_triples():
        if triple.subject == entity or triple.object == entity:
            vector[label_to_position[triple.predicate]] += 1.0
    return vector


def normalize_and_pad(vector: np.ndarray) -> np.ndarray:
    """Normalize a real vector and pad it to a power-of-two dimension."""

    vector = np.asarray(vector, dtype=float)
    padded = np.zeros(next_power_of_two_length(len(vector)), dtype=complex)
    padded[: len(vector)] = vector
    norm = np.linalg.norm(padded)
    if np.isclose(norm, 0.0):
        raise ValueError("Cannot amplitude encode a zero vector.")
    return padded / norm


def build_swap_test_circuit(
    normalized_vector_a: np.ndarray,
    normalized_vector_b: np.ndarray,
) -> QuantumCircuit:
    """Build a swap test over two amplitude-encoded vectors."""

    vector_a = np.asarray(normalized_vector_a, dtype=complex)
    vector_b = np.asarray(normalized_vector_b, dtype=complex)
    if vector_a.shape != vector_b.shape:
        raise ValueError("Swap-test vectors must have the same dimension.")
    if not np.isclose(np.linalg.norm(vector_a), 1.0):
        raise ValueError("The first swap-test vector must be normalized.")
    if not np.isclose(np.linalg.norm(vector_b), 1.0):
        raise ValueError("The second swap-test vector must be normalized.")

    dimension = len(vector_a)
    if dimension & (dimension - 1):
        raise ValueError("Swap-test vector dimension must be a power of two.")

    register_qubits = int(math.log2(dimension))
    ancilla = 0
    register_a = list(range(1, 1 + register_qubits))
    register_b = list(range(1 + register_qubits, 1 + (2 * register_qubits)))

    circuit = QuantumCircuit(1 + (2 * register_qubits), 1, name="AmplitudeSimilaritySwapTest")
    circuit.h(ancilla)
    circuit.initialize(vector_a, register_a)
    circuit.initialize(vector_b, register_b)
    for qubit_a, qubit_b in zip(register_a, register_b):
        circuit.cswap(ancilla, qubit_a, qubit_b)
    circuit.h(ancilla)
    circuit.measure(ancilla, 0)
    return circuit


def classical_squared_similarity(
    normalized_vector_a: np.ndarray,
    normalized_vector_b: np.ndarray,
) -> float:
    """Return |<a|b>|^2 for two normalized state vectors."""

    overlap = np.vdot(normalized_vector_a, normalized_vector_b)
    return float(abs(overlap) ** 2)


def amplitude_memory_estimates(*vectors: np.ndarray) -> tuple[int, int]:
    """Return dense and sparse byte estimates for small complex vectors."""

    dense_bytes = sum(int(vector.size) * np.dtype(np.complex128).itemsize for vector in vectors)
    sparse_entries = sum(int(np.count_nonzero(vector)) for vector in vectors)
    index_bytes = np.dtype(np.int64).itemsize
    value_bytes = np.dtype(np.complex128).itemsize
    sparse_bytes = sparse_entries * (index_bytes + value_bytes)
    return dense_bytes, sparse_bytes


def run_amplitude_similarity_task(
    *,
    entity_a: str = DEFAULT_ENTITY_A,
    entity_b: str = DEFAULT_ENTITY_B,
    shots: int = DEFAULT_SHOTS,
    seed_simulator: int = DEFAULT_SEED_SIMULATOR,
) -> AmplitudeSimilarityResult:
    """Run the Chapter 9 amplitude-encoding similarity validation task."""

    task_start = time.perf_counter()
    feature_labels = _predicate_feature_labels()
    vector_a = entity_context_vector(entity_a, feature_labels=feature_labels)
    vector_b = entity_context_vector(entity_b, feature_labels=feature_labels)
    normalization_start = time.perf_counter()
    normalized_a = normalize_and_pad(vector_a)
    normalized_b = normalize_and_pad(vector_b)
    normalization_time = time.perf_counter() - normalization_start

    preparation_start = time.perf_counter()
    circuit = build_swap_test_circuit(
        normalized_vector_a=normalized_a,
        normalized_vector_b=normalized_b,
    )
    state_preparation_time = time.perf_counter() - preparation_start
    simulator = AerSimulator(seed_simulator=seed_simulator)
    simulation_start = time.perf_counter()
    transpiled_circuit = transpile(circuit, simulator, optimization_level=0)
    simulation_time = time.perf_counter() - simulation_start
    measurement_start = time.perf_counter()
    result = simulator.run(transpiled_circuit, shots=shots).result()
    measurement_time = time.perf_counter() - measurement_start
    readout_start = time.perf_counter()
    counts = dict(sorted(result.get_counts().items()))
    probabilities = {
        bitstring: count / shots
        for bitstring, count in counts.items()
    }

    p0 = counts.get("0", 0) / shots
    estimated_similarity = min(1.0, max(0.0, (2 * p0) - 1))
    classical_similarity = classical_squared_similarity(normalized_a, normalized_b)
    absolute_error = abs(estimated_similarity - classical_similarity)
    relative_error = (
        absolute_error / abs(classical_similarity)
        if not np.isclose(classical_similarity, 0.0)
        else None
    )
    readout_decoding_time = time.perf_counter() - readout_start
    tolerance = DEFAULT_SWAP_TEST_TOLERANCE if shots >= 1000 else 0.25
    pass_fail = "pass" if absolute_error <= tolerance else "fail"
    dense_memory_bytes, sparse_memory_bytes = amplitude_memory_estimates(
        normalized_a,
        normalized_b,
    )
    task_time_seconds = time.perf_counter() - task_start

    return AmplitudeSimilarityResult(
        task_name="amplitude_similarity_swap_test",
        entity_a=entity_a,
        entity_b=entity_b,
        feature_labels=feature_labels,
        vector_a=vector_a.tolist(),
        vector_b=vector_b.tolist(),
        normalized_vector_a=[float(value.real) for value in normalized_a],
        normalized_vector_b=[float(value.real) for value in normalized_b],
        counts=counts,
        probabilities=probabilities,
        task_time_seconds=task_time_seconds,
        normalization_time_seconds=normalization_time,
        state_preparation_time_seconds=state_preparation_time,
        simulation_time_seconds=simulation_time,
        measurement_time_seconds=measurement_time,
        readout_decoding_time_seconds=readout_decoding_time,
        num_qubits=circuit.num_qubits,
        vector_dimension=int(normalized_a.size),
        nonzero_entries_a=int(np.count_nonzero(normalized_a)),
        nonzero_entries_b=int(np.count_nonzero(normalized_b)),
        dense_memory_bytes=dense_memory_bytes,
        sparse_memory_bytes=sparse_memory_bytes,
        circuit_depth=circuit.depth(),
        gate_count=_operation_count(circuit),
        transpiled_depth=transpiled_circuit.depth(),
        transpiled_gate_count=_operation_count(transpiled_circuit),
        shots=shots,
        p0=p0,
        estimated_similarity=estimated_similarity,
        classical_similarity=classical_similarity,
        absolute_error=absolute_error,
        relative_error=relative_error,
        pass_threshold=tolerance,
        pass_fail=pass_fail,
        threshold_reason=(
            "Swap-test squared-similarity estimate must be within the configured "
            "absolute-error tolerance of the exact NumPy dot product."
        ),
        measurement_mode="shot-based swap-test ancilla readout",
        claim_scope="implementation-level validation; no quantum-advantage claim",
        backend="AerSimulator",
        claim_note=(
            "This is a small validation task for amplitude encoding and swap-test "
            "similarity estimation, not a quantum-advantage claim."
        ),
    )
