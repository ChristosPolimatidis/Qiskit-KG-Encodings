from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from src.running_example import EX
from src.tasks.amplitude_similarity import (
    DEFAULT_SWAP_TEST_TOLERANCE,
    amplitude_memory_estimates,
    build_swap_test_circuit,
    classical_squared_similarity,
    normalize_and_pad,
)


DEFAULT_HEAD_ENTITY = f"{EX}Aristotle"
DEFAULT_RELATION = f"{EX}livesAt"
DEFAULT_TAIL_ENTITY = f"{EX}Athens"
DEFAULT_SHOTS = 4096
DEFAULT_SEED_SIMULATOR = 11223

ENTITY_VECTORS: dict[str, tuple[float, float]] = {
    DEFAULT_HEAD_ENTITY: (0.20, 0.70),
    DEFAULT_TAIL_ENTITY: (0.68, 0.52),
}

RELATION_VECTORS: dict[str, tuple[float, float]] = {
    DEFAULT_RELATION: (0.50, -0.20),
}


@dataclass(frozen=True, slots=True)
class LinkPredictionDistanceResult:
    """Result bundle for the Chapter 9 link-prediction distance validation."""

    task_name: str
    head_entity: str
    relation: str
    tail_entity: str
    vector_labels: list[str]
    head_vector: list[float]
    relation_vector: list[float]
    tail_vector: list[float]
    source_vector: list[float]
    normalized_source_vector: list[float]
    normalized_tail_vector: list[float]
    classical_distance: float
    classical_similarity: float
    estimated_similarity: float
    absolute_error: float
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
    nonzero_entries_source: int
    nonzero_entries_tail: int
    dense_memory_bytes: int
    sparse_memory_bytes: int
    circuit_depth: int
    gate_count: int
    transpiled_depth: int
    transpiled_gate_count: int
    shots: int
    backend: str
    relative_error: float | None
    pass_threshold: float
    pass_fail: str
    threshold_reason: str
    measurement_mode: str
    claim_scope: str
    method_note: str
    claim_note: str


def _operation_count(circuit) -> int:
    return sum(int(count) for count in circuit.count_ops().values())


def _vector_for(mapping: dict[str, tuple[float, float]], key: str) -> np.ndarray:
    if key not in mapping:
        raise ValueError(f"No deterministic vector is defined for '{key}'.")
    return np.asarray(mapping[key], dtype=float)


def run_link_prediction_distance_task(
    *,
    head_entity: str = DEFAULT_HEAD_ENTITY,
    relation: str = DEFAULT_RELATION,
    tail_entity: str = DEFAULT_TAIL_ENTITY,
    shots: int = DEFAULT_SHOTS,
    seed_simulator: int = DEFAULT_SEED_SIMULATOR,
) -> LinkPredictionDistanceResult:
    """Run a small TransE-style distance-estimation validation.

    The task compares e_h + r against e_t for the running-example link
    (ex:Aristotle, ex:livesAt, ex:Athens). It uses a swap test as a compact
    distance/similarity validation; it does not implement HHL.
    """

    task_start = time.perf_counter()
    head_vector = _vector_for(ENTITY_VECTORS, head_entity)
    relation_vector = _vector_for(RELATION_VECTORS, relation)
    tail_vector = _vector_for(ENTITY_VECTORS, tail_entity)
    source_vector = head_vector + relation_vector

    normalization_start = time.perf_counter()
    normalized_source = normalize_and_pad(source_vector)
    normalized_tail = normalize_and_pad(tail_vector)
    normalization_time = time.perf_counter() - normalization_start
    preparation_start = time.perf_counter()
    circuit = build_swap_test_circuit(
        normalized_vector_a=normalized_source,
        normalized_vector_b=normalized_tail,
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
    classical_similarity = classical_squared_similarity(
        normalized_source,
        normalized_tail,
    )
    absolute_error = abs(estimated_similarity - classical_similarity)
    relative_error = (
        absolute_error / abs(classical_similarity)
        if not np.isclose(classical_similarity, 0.0)
        else None
    )
    classical_distance = float(np.linalg.norm(source_vector - tail_vector))
    readout_decoding_time = time.perf_counter() - readout_start
    tolerance = DEFAULT_SWAP_TEST_TOLERANCE if shots >= 1000 else 0.25
    pass_fail = "pass" if absolute_error <= tolerance else "fail"
    dense_memory_bytes, sparse_memory_bytes = amplitude_memory_estimates(
        normalized_source,
        normalized_tail,
    )
    task_time_seconds = time.perf_counter() - task_start

    return LinkPredictionDistanceResult(
        task_name="link_prediction_distance_estimation",
        head_entity=head_entity,
        relation=relation,
        tail_entity=tail_entity,
        vector_labels=["latent_dim_0", "latent_dim_1"],
        head_vector=head_vector.tolist(),
        relation_vector=relation_vector.tolist(),
        tail_vector=tail_vector.tolist(),
        source_vector=source_vector.tolist(),
        normalized_source_vector=[float(value.real) for value in normalized_source],
        normalized_tail_vector=[float(value.real) for value in normalized_tail],
        classical_distance=classical_distance,
        classical_similarity=classical_similarity,
        estimated_similarity=estimated_similarity,
        absolute_error=absolute_error,
        counts=counts,
        probabilities=probabilities,
        task_time_seconds=task_time_seconds,
        normalization_time_seconds=normalization_time,
        state_preparation_time_seconds=state_preparation_time,
        simulation_time_seconds=simulation_time,
        measurement_time_seconds=measurement_time,
        readout_decoding_time_seconds=readout_decoding_time,
        num_qubits=circuit.num_qubits,
        vector_dimension=int(normalized_source.size),
        nonzero_entries_source=int(np.count_nonzero(normalized_source)),
        nonzero_entries_tail=int(np.count_nonzero(normalized_tail)),
        dense_memory_bytes=dense_memory_bytes,
        sparse_memory_bytes=sparse_memory_bytes,
        circuit_depth=circuit.depth(),
        gate_count=_operation_count(circuit),
        transpiled_depth=transpiled_circuit.depth(),
        transpiled_gate_count=_operation_count(transpiled_circuit),
        shots=shots,
        backend="AerSimulator",
        relative_error=relative_error,
        pass_threshold=tolerance,
        pass_fail=pass_fail,
        threshold_reason=(
            "Swap-test similarity estimate for e_h + r versus e_t must be "
            "within the configured absolute-error tolerance of the exact "
            "NumPy similarity."
        ),
        measurement_mode="shot-based swap-test ancilla readout",
        claim_scope="implementation-level validation; no quantum-advantage claim",
        method_note=(
            "Distance-estimation validation using amplitude encoding and a swap "
            "test over e_h + r and e_t."
        ),
        claim_note=(
            "This validates the Table 1 Link Prediction -> Amplitude -> "
            "Distance Estimation mapping. It is not an HHL implementation."
        ),
    )
