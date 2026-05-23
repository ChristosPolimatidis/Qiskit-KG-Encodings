from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from src.running_example import EX
from src.tasks.amplitude_similarity import (
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
    task_time_seconds: float
    num_qubits: int
    circuit_depth: int
    gate_count: int
    transpiled_depth: int
    transpiled_gate_count: int
    shots: int
    backend: str
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

    normalized_source = normalize_and_pad(source_vector)
    normalized_tail = normalize_and_pad(tail_vector)
    circuit = build_swap_test_circuit(
        normalized_vector_a=normalized_source,
        normalized_vector_b=normalized_tail,
    )

    simulator = AerSimulator(seed_simulator=seed_simulator)
    transpiled_circuit = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = dict(sorted(result.get_counts().items()))

    p0 = counts.get("0", 0) / shots
    estimated_similarity = min(1.0, max(0.0, (2 * p0) - 1))
    classical_similarity = classical_squared_similarity(
        normalized_source,
        normalized_tail,
    )
    absolute_error = abs(estimated_similarity - classical_similarity)
    classical_distance = float(np.linalg.norm(source_vector - tail_vector))
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
        task_time_seconds=task_time_seconds,
        num_qubits=circuit.num_qubits,
        circuit_depth=circuit.depth(),
        gate_count=_operation_count(circuit),
        transpiled_depth=transpiled_circuit.depth(),
        transpiled_gate_count=_operation_count(transpiled_circuit),
        shots=shots,
        backend="AerSimulator",
        method_note=(
            "Distance-estimation validation using amplitude encoding and a swap "
            "test over e_h + r and e_t."
        ),
        claim_note=(
            "This validates the Table 1 Link Prediction -> Amplitude -> "
            "Distance Estimation mapping. It is not an HHL implementation."
        ),
    )
