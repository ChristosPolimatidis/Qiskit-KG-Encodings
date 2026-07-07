from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from src.tasks.amplitude_similarity import (
    build_swap_test_circuit,
    classical_squared_similarity,
    normalize_and_pad,
)
from src.running_example import EX, get_running_example_triples


FEATURE_SPACE = ["Aristotle", "Athens", "Person", "City", "Philosophy"]
QUERY_TERMS = ["Aristotle", "Athens"]
ENTITY_FEATURE_VECTORS = {
    "Aristotle": [1.0, 1.0, 1.0, 0.0, 1.0],
    "Athens": [0.0, 1.0, 0.0, 1.0, 0.0],
}
QUERY_VECTOR = [1.0, 1.0, 0.0, 0.0, 0.0]
DEFAULT_SHOTS = 4096
DEFAULT_SEED_SIMULATOR = 35791
DEFAULT_KEYWORD_SCORE_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class KeywordSearchResult:
    """Result bundle for the Chapter 9 keyword-search validation task."""

    task_name: str
    task: str
    encoding: str
    method: str
    feature_space: list[str]
    query_terms: list[str]
    query_vector: list[float]
    normalized_query_vector: list[float]
    candidate_entities: list[str]
    entity_vectors: dict[str, list[float]]
    normalized_entity_vectors: dict[str, list[float]]
    counts_by_entity: dict[str, dict[str, int]]
    probabilities_by_entity: dict[str, dict[str, float]]
    classical_scores: dict[str, float]
    estimated_scores: dict[str, float]
    ranking: list[str]
    top_result: str
    top_score: float
    absolute_errors: dict[str, float]
    max_absolute_error: float
    task_time_seconds: float
    num_qubits: int
    circuit_depth: int
    gate_count: int
    transpiled_depth: int
    transpiled_gate_count: int
    shots: int
    repetitions: int
    backend: str
    pass_threshold: float
    pass_fail: str
    threshold_reason: str
    measurement_mode: str
    claim_scope: str
    claim_note: str


def _operation_count(circuit) -> int:
    return sum(int(count) for count in circuit.count_ops().values())


def _local_name(uri: str) -> str:
    return uri.removeprefix(EX)


def running_example_local_terms() -> set[str]:
    """Return local names appearing in the canonical running-example triples."""

    terms: set[str] = set()
    for triple in get_running_example_triples():
        terms.add(_local_name(triple.subject))
        terms.add(_local_name(triple.object))
    return terms


def validate_running_example_terms() -> None:
    """Ensure the keyword-search toy vectors use only running-example terms."""

    available_terms = running_example_local_terms()
    required_terms = set(FEATURE_SPACE)
    required_terms.update(QUERY_TERMS)
    required_terms.update(ENTITY_FEATURE_VECTORS)
    missing_terms = sorted(required_terms - available_terms)
    if missing_terms:
        raise ValueError(
            "Keyword-search features must come from the running example. "
            f"Missing terms: {missing_terms}"
        )


def normalize_feature_vector(vector: list[float] | np.ndarray) -> np.ndarray:
    """Normalize and pad a keyword/entity feature vector for amplitude encoding."""

    return normalize_and_pad(np.asarray(vector, dtype=float))


def classical_keyword_scores(
    *,
    query_vector: list[float] | np.ndarray = QUERY_VECTOR,
    entity_vectors: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    """Compute exact squared similarities for the keyword-search toy task."""

    entities = ENTITY_FEATURE_VECTORS if entity_vectors is None else entity_vectors
    normalized_query = normalize_feature_vector(query_vector)
    scores: dict[str, float] = {}
    for entity, vector in entities.items():
        scores[entity] = classical_squared_similarity(
            normalized_query,
            normalize_feature_vector(vector),
        )
    return scores


def run_keyword_search_task(
    *,
    shots: int = DEFAULT_SHOTS,
    seed_simulator: int = DEFAULT_SEED_SIMULATOR,
    repetitions: int = 1,
) -> KeywordSearchResult:
    """Run a small keyword-search validation using amplitude encoding and swap tests.

    This validates ranking entities by query-vector similarity over the running
    example feature space. It is not a full RDF keyword search engine.
    """

    task_start = time.perf_counter()
    validate_running_example_terms()
    normalized_query = normalize_feature_vector(QUERY_VECTOR)
    normalized_entities = {
        entity: normalize_feature_vector(vector)
        for entity, vector in ENTITY_FEATURE_VECTORS.items()
    }

    simulator = AerSimulator(seed_simulator=seed_simulator)
    counts_by_entity: dict[str, dict[str, int]] = {}
    probabilities_by_entity: dict[str, dict[str, float]] = {}
    estimated_scores: dict[str, float] = {}
    classical_scores: dict[str, float] = {}
    absolute_errors: dict[str, float] = {}
    circuit_depths: list[int] = []
    gate_counts: list[int] = []
    transpiled_depths: list[int] = []
    transpiled_gate_counts: list[int] = []
    num_qubits = 0

    for position, (entity, normalized_entity) in enumerate(normalized_entities.items()):
        circuit = build_swap_test_circuit(normalized_query, normalized_entity)
        transpiled_circuit = transpile(circuit, simulator, optimization_level=0)
        result = simulator.run(
            transpiled_circuit,
            shots=shots,
            seed_simulator=seed_simulator + position,
        ).result()
        counts = dict(sorted(result.get_counts().items()))
        probabilities = {
            bitstring: count / shots
            for bitstring, count in counts.items()
        }
        p0 = counts.get("0", 0) / shots
        estimated_score = min(1.0, max(0.0, (2 * p0) - 1))
        classical_score = classical_squared_similarity(
            normalized_query,
            normalized_entity,
        )

        counts_by_entity[entity] = counts
        probabilities_by_entity[entity] = probabilities
        estimated_scores[entity] = estimated_score
        classical_scores[entity] = classical_score
        absolute_errors[entity] = abs(estimated_score - classical_score)
        num_qubits = max(num_qubits, circuit.num_qubits)
        circuit_depths.append(circuit.depth())
        gate_counts.append(_operation_count(circuit))
        transpiled_depths.append(transpiled_circuit.depth())
        transpiled_gate_counts.append(_operation_count(transpiled_circuit))

    ranking = sorted(
        estimated_scores,
        key=lambda entity: (
            -estimated_scores[entity],
            -classical_scores[entity],
            entity,
        ),
    )
    top_result = ranking[0]
    top_score = estimated_scores[top_result]
    max_absolute_error = max(absolute_errors.values()) if absolute_errors else 0.0
    tolerance = DEFAULT_KEYWORD_SCORE_TOLERANCE if shots >= 1000 else 0.25
    pass_fail = "pass" if max_absolute_error <= tolerance else "fail"
    task_time_seconds = time.perf_counter() - task_start

    return KeywordSearchResult(
        task_name="keyword_search_swap_test",
        task="Keyword Search",
        encoding="Amplitude",
        method="Swap Test",
        feature_space=list(FEATURE_SPACE),
        query_terms=list(QUERY_TERMS),
        query_vector=list(QUERY_VECTOR),
        normalized_query_vector=[float(value.real) for value in normalized_query],
        candidate_entities=list(ENTITY_FEATURE_VECTORS),
        entity_vectors={
            entity: list(vector)
            for entity, vector in ENTITY_FEATURE_VECTORS.items()
        },
        normalized_entity_vectors={
            entity: [float(value.real) for value in vector]
            for entity, vector in normalized_entities.items()
        },
        counts_by_entity=counts_by_entity,
        probabilities_by_entity=probabilities_by_entity,
        classical_scores=classical_scores,
        estimated_scores=estimated_scores,
        ranking=ranking,
        top_result=top_result,
        top_score=top_score,
        absolute_errors=absolute_errors,
        max_absolute_error=max_absolute_error,
        task_time_seconds=task_time_seconds,
        num_qubits=num_qubits,
        circuit_depth=max(circuit_depths),
        gate_count=max(gate_counts),
        transpiled_depth=max(transpiled_depths),
        transpiled_gate_count=max(transpiled_gate_counts),
        shots=shots,
        repetitions=repetitions,
        backend="AerSimulator",
        pass_threshold=tolerance,
        pass_fail=pass_fail,
        threshold_reason=(
            "Each swap-test keyword score is compared with its exact NumPy "
            "squared-similarity score; the maximum absolute error must stay "
            "within tolerance."
        ),
        measurement_mode="shot-based swap-test ancilla readout per candidate entity",
        claim_scope="implementation-level validation; no quantum-advantage claim",
        claim_note=(
            "Small keyword-search validation over the running example using "
            "amplitude encoding and swap-test similarity; not a full RDF "
            "keyword search engine and not a quantum-advantage claim."
        ),
    )
