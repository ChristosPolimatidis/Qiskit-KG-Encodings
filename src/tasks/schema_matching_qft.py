from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from src.running_example import RDF_TYPE, RDFS_SUBCLASS_OF, get_predicate_phase_map


DEFAULT_SHOTS = 4096
DEFAULT_SEED_SIMULATOR = 86420
DEFAULT_SCHEMA_SIMILARITY_TOLERANCE = 0.08
SCHEMA_PHASE_STRATEGIES = (
    "exact_uri",
    "synonym_dictionary",
    "negative_control",
)
TYPE_ALIAS = "http://example.org/schema/typeAlias"
SUBCLASS_ALIAS = "http://example.org/schema/subClassAlias"
UNMAPPED_A = "http://example.org/schema/unmappedA"
UNMAPPED_B = "http://example.org/schema/unmappedB"
NO_RELATION = "none"


@dataclass(frozen=True, slots=True)
class SchemaMatchingQFTResult:
    """Result bundle for the Chapter 9 schema-pattern QFT validation."""

    task_name: str
    relation_a_label: str
    relation_b_label: str
    phase_assignment_strategy: str
    phase_assignment_assumptions: list[str]
    phase_assignments: list[dict[str, Any]]
    phase_pattern_a: list[float]
    phase_pattern_b: list[float]
    fourier_magnitudes_a: list[float]
    fourier_magnitudes_b: list[float]
    fourier_probabilities_a: dict[str, float]
    fourier_probabilities_b: dict[str, float]
    counts_pattern_a: dict[str, int]
    counts_pattern_b: dict[str, int]
    measured_probabilities_a: dict[str, float]
    measured_probabilities_b: dict[str, float]
    fourier_pattern_similarity: float
    exact_distribution_similarity: float
    measured_distribution_similarity: float
    absolute_error: float
    relative_error: float | None
    negative_control_similarity: float
    strategy_results: dict[str, dict[str, Any]]
    negative_control: dict[str, Any]
    task_time_seconds: float
    num_qubits: int
    circuit_depth: int
    gate_count: int
    transpiled_depth: int
    transpiled_gate_count: int
    shots: int
    backend: str
    pass_threshold: float
    pass_fail: str
    threshold_reason: str
    measurement_mode: str
    claim_scope: str
    method_note: str
    claim_note: str


def _operation_count(circuit: QuantumCircuit) -> int:
    return sum(int(count) for count in circuit.count_ops().values())


def _phase_state(pattern: list[float]) -> np.ndarray:
    dimension = len(pattern)
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("QFT phase patterns must have power-of-two length.")
    return np.exp(1j * np.asarray(pattern, dtype=float)) / math.sqrt(dimension)


def _qft_circuit(pattern: list[float], *, name: str = "SchemaPatternQFT") -> QuantumCircuit:
    state = _phase_state(pattern)
    num_qubits = int(math.log2(len(state)))
    circuit = QuantumCircuit(num_qubits, name=name)
    circuit.initialize(state, range(num_qubits))
    circuit.append(QFTGate(num_qubits), range(num_qubits))
    return circuit


def _qft_magnitudes(pattern: list[float]) -> tuple[np.ndarray, QuantumCircuit]:
    circuit = _qft_circuit(pattern)
    transformed = Statevector.from_instruction(circuit).data
    return np.abs(transformed), circuit


def _qft_probabilities(pattern: list[float]) -> dict[str, float]:
    circuit = _qft_circuit(pattern)
    return {
        str(bitstring): float(probability)
        for bitstring, probability in sorted(
            Statevector.from_instruction(circuit).probabilities_dict().items()
        )
    }


def _counts_to_probabilities(counts: dict[str, int], shots: int) -> dict[str, float]:
    if shots < 1:
        raise ValueError("shots must be at least 1.")
    return {
        bitstring: count / shots
        for bitstring, count in sorted(counts.items())
    }


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if np.isclose(denominator, 0.0):
        raise ValueError("Cannot compare zero Fourier vectors.")
    similarity = float(np.dot(vector_a, vector_b) / denominator)
    return max(0.0, min(1.0, similarity))


def _dict_cosine_similarity(
    distribution_a: dict[str, float],
    distribution_b: dict[str, float],
) -> float:
    keys = sorted(set(distribution_a) | set(distribution_b))
    vector_a = np.asarray([distribution_a.get(key, 0.0) for key in keys], dtype=float)
    vector_b = np.asarray([distribution_b.get(key, 0.0) for key in keys], dtype=float)
    return _cosine_similarity(vector_a, vector_b)


def _combined_schema_circuit(
    pattern_a: list[float],
    pattern_b: list[float],
) -> QuantumCircuit:
    state_a = _phase_state(pattern_a)
    state_b = _phase_state(pattern_b)
    register_qubits = int(math.log2(len(state_a)))
    circuit = QuantumCircuit(register_qubits * 2, name="SchemaMatchingQFT")
    register_a = list(range(register_qubits))
    register_b = list(range(register_qubits, register_qubits * 2))
    circuit.initialize(state_a, register_a)
    circuit.initialize(state_b, register_b)
    qft_gate = QFTGate(register_qubits)
    circuit.append(qft_gate, register_a)
    circuit.append(qft_gate, register_b)
    return circuit


def schema_phase_assignments(strategy: str) -> list[dict[str, Any]]:
    """Return explicit phase assignment assumptions for a schema strategy."""

    if strategy not in SCHEMA_PHASE_STRATEGIES:
        raise ValueError(f"Unknown schema phase strategy: {strategy}")
    phase_map = get_predicate_phase_map()
    theta_type = float(phase_map[RDF_TYPE])
    theta_subclass = float(phase_map[RDFS_SUBCLASS_OF])
    if strategy == "exact_uri":
        return [
            {
                "strategy": strategy,
                "relation": RDF_TYPE,
                "canonical_relation": RDF_TYPE,
                "phase": theta_type,
                "assumption": "relations share the exact same URI before encoding",
            },
            {
                "strategy": strategy,
                "relation": RDFS_SUBCLASS_OF,
                "canonical_relation": RDFS_SUBCLASS_OF,
                "phase": theta_subclass,
                "assumption": "relations share the exact same URI before encoding",
            },
        ]
    if strategy == "synonym_dictionary":
        return [
            {
                "strategy": strategy,
                "relation": RDF_TYPE,
                "canonical_relation": RDF_TYPE,
                "phase": theta_type,
                "assumption": "provided synonym dictionary maps this label to rdf:type",
            },
            {
                "strategy": strategy,
                "relation": TYPE_ALIAS,
                "canonical_relation": RDF_TYPE,
                "phase": theta_type,
                "assumption": "provided synonym dictionary maps this label to rdf:type",
            },
            {
                "strategy": strategy,
                "relation": RDFS_SUBCLASS_OF,
                "canonical_relation": RDFS_SUBCLASS_OF,
                "phase": theta_subclass,
                "assumption": "provided synonym dictionary maps this label to rdfs:subClassOf",
            },
            {
                "strategy": strategy,
                "relation": SUBCLASS_ALIAS,
                "canonical_relation": RDFS_SUBCLASS_OF,
                "phase": theta_subclass,
                "assumption": "provided synonym dictionary maps this label to rdfs:subClassOf",
            },
        ]
    return [
        {
            "strategy": strategy,
            "relation": RDF_TYPE,
            "canonical_relation": RDF_TYPE,
            "phase": theta_type,
            "assumption": "left schema pattern uses known running-example phases",
        },
        {
            "strategy": strategy,
            "relation": RDFS_SUBCLASS_OF,
            "canonical_relation": RDFS_SUBCLASS_OF,
            "phase": theta_subclass,
            "assumption": "left schema pattern uses known running-example phases",
        },
        {
            "strategy": strategy,
            "relation": UNMAPPED_A,
            "canonical_relation": None,
            "phase": math.pi,
            "assumption": "no semantic prior is available for this relation",
        },
        {
            "strategy": strategy,
            "relation": UNMAPPED_B,
            "canonical_relation": None,
            "phase": 0.0,
            "assumption": "no semantic prior is available for this relation",
        },
    ]


def _assignment_map(strategy: str) -> dict[str, float]:
    assignments = schema_phase_assignments(strategy)
    phase_by_relation = {
        str(row["relation"]): float(row["phase"])
        for row in assignments
    }
    phase_by_relation[NO_RELATION] = 0.0
    return phase_by_relation


def schema_phase_patterns(strategy: str) -> tuple[list[str], list[str], list[float], list[float]]:
    phase_by_relation = _assignment_map(strategy)
    labels_a = [RDF_TYPE, RDFS_SUBCLASS_OF, RDF_TYPE, NO_RELATION]
    if strategy == "exact_uri":
        labels_b = [RDF_TYPE, RDFS_SUBCLASS_OF, RDF_TYPE, NO_RELATION]
    elif strategy == "synonym_dictionary":
        labels_b = [TYPE_ALIAS, SUBCLASS_ALIAS, TYPE_ALIAS, NO_RELATION]
    elif strategy == "negative_control":
        labels_b = [UNMAPPED_A, UNMAPPED_B, UNMAPPED_A, UNMAPPED_B]
    else:
        raise ValueError(f"Unknown schema phase strategy: {strategy}")
    pattern_a = [phase_by_relation[label] for label in labels_a]
    pattern_b = [phase_by_relation[label] for label in labels_b]
    return labels_a, labels_b, pattern_a, pattern_b


def _measure_qft_pattern(
    pattern: list[float],
    *,
    shots: int,
    seed_simulator: int,
    name: str,
) -> tuple[dict[str, int], dict[str, float], QuantumCircuit]:
    simulator = AerSimulator(seed_simulator=seed_simulator)
    circuit = _qft_circuit(pattern, name=name)
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()
    transpiled_circuit = transpile(measured_circuit, simulator, optimization_level=0)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = dict(sorted(result.get_counts().items()))
    return counts, _counts_to_probabilities(counts, shots), transpiled_circuit


def _strategy_result(
    strategy: str,
    *,
    shots: int,
    seed_simulator: int,
) -> dict[str, Any]:
    labels_a, labels_b, pattern_a, pattern_b = schema_phase_patterns(strategy)
    magnitudes_a, circuit_a = _qft_magnitudes(pattern_a)
    magnitudes_b, circuit_b = _qft_magnitudes(pattern_b)
    exact_probabilities_a = _qft_probabilities(pattern_a)
    exact_probabilities_b = _qft_probabilities(pattern_b)
    counts_a, measured_probabilities_a, transpiled_a = _measure_qft_pattern(
        pattern_a,
        shots=shots,
        seed_simulator=seed_simulator,
        name=f"SchemaQFT{strategy}A",
    )
    counts_b, measured_probabilities_b, transpiled_b = _measure_qft_pattern(
        pattern_b,
        shots=shots,
        seed_simulator=seed_simulator + 1,
        name=f"SchemaQFT{strategy}B",
    )
    exact_magnitude_similarity = _cosine_similarity(magnitudes_a, magnitudes_b)
    exact_distribution_similarity = _dict_cosine_similarity(
        exact_probabilities_a,
        exact_probabilities_b,
    )
    measured_distribution_similarity = _dict_cosine_similarity(
        measured_probabilities_a,
        measured_probabilities_b,
    )
    return {
        "strategy": strategy,
        "labels_a": labels_a,
        "labels_b": labels_b,
        "phase_pattern_a": [float(value) for value in pattern_a],
        "phase_pattern_b": [float(value) for value in pattern_b],
        "phase_assignments": schema_phase_assignments(strategy),
        "fourier_magnitudes_a": [float(value) for value in magnitudes_a],
        "fourier_magnitudes_b": [float(value) for value in magnitudes_b],
        "fourier_probabilities_a": exact_probabilities_a,
        "fourier_probabilities_b": exact_probabilities_b,
        "counts_pattern_a": counts_a,
        "counts_pattern_b": counts_b,
        "measured_probabilities_a": measured_probabilities_a,
        "measured_probabilities_b": measured_probabilities_b,
        "fourier_pattern_similarity": exact_magnitude_similarity,
        "exact_distribution_similarity": exact_distribution_similarity,
        "measured_distribution_similarity": measured_distribution_similarity,
        "absolute_error": abs(
            measured_distribution_similarity - exact_distribution_similarity
        ),
        "pattern_a_circuit_depth": circuit_a.depth(),
        "pattern_b_circuit_depth": circuit_b.depth(),
        "pattern_a_gate_count": _operation_count(circuit_a),
        "pattern_b_gate_count": _operation_count(circuit_b),
        "pattern_a_transpiled_depth": transpiled_a.depth(),
        "pattern_b_transpiled_depth": transpiled_b.depth(),
        "pattern_a_transpiled_gate_count": _operation_count(transpiled_a),
        "pattern_b_transpiled_gate_count": _operation_count(transpiled_b),
    }


def run_schema_matching_qft_task(
    *,
    phase_assignment_strategy: str = "synonym_dictionary",
    shots: int = DEFAULT_SHOTS,
    seed_simulator: int = DEFAULT_SEED_SIMULATOR,
    similarity_tolerance: float | None = None,
) -> SchemaMatchingQFTResult:
    """Run a toy QFT validation over schema relation phase patterns."""

    if phase_assignment_strategy not in SCHEMA_PHASE_STRATEGIES:
        raise ValueError(
            f"phase_assignment_strategy must be one of {SCHEMA_PHASE_STRATEGIES}."
        )
    task_start = time.perf_counter()
    tolerance = (
        DEFAULT_SCHEMA_SIMILARITY_TOLERANCE
        if similarity_tolerance is None and shots >= 1000
        else (0.20 if similarity_tolerance is None else similarity_tolerance)
    )
    strategy_results = {
        strategy: _strategy_result(
            strategy,
            shots=shots,
            seed_simulator=seed_simulator + (100 * position),
        )
        for position, strategy in enumerate(SCHEMA_PHASE_STRATEGIES)
    }
    selected = strategy_results[phase_assignment_strategy]
    negative_control = strategy_results["negative_control"]
    pattern_a = selected["phase_pattern_a"]
    pattern_b = selected["phase_pattern_b"]
    circuit = _combined_schema_circuit(pattern_a, pattern_b)
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()
    simulator = AerSimulator(seed_simulator=seed_simulator)
    transpiled_circuit = transpile(measured_circuit, simulator, optimization_level=0)
    absolute_error = float(selected["absolute_error"])
    relative_error = (
        absolute_error / abs(float(selected["exact_distribution_similarity"]))
        if not math.isclose(float(selected["exact_distribution_similarity"]), 0.0)
        else None
    )
    pass_fail = "pass" if absolute_error <= tolerance else "fail"
    assumptions = sorted(
        {
            str(row["assumption"])
            for row in selected["phase_assignments"]
        }
    )
    task_time_seconds = time.perf_counter() - task_start

    return SchemaMatchingQFTResult(
        task_name="schema_matching_qft",
        relation_a_label="schema_pattern_type_subclass_type",
        relation_b_label=f"{phase_assignment_strategy}_comparison_pattern",
        phase_assignment_strategy=phase_assignment_strategy,
        phase_assignment_assumptions=assumptions,
        phase_assignments=selected["phase_assignments"],
        phase_pattern_a=pattern_a,
        phase_pattern_b=pattern_b,
        fourier_magnitudes_a=selected["fourier_magnitudes_a"],
        fourier_magnitudes_b=selected["fourier_magnitudes_b"],
        fourier_probabilities_a=selected["fourier_probabilities_a"],
        fourier_probabilities_b=selected["fourier_probabilities_b"],
        counts_pattern_a=selected["counts_pattern_a"],
        counts_pattern_b=selected["counts_pattern_b"],
        measured_probabilities_a=selected["measured_probabilities_a"],
        measured_probabilities_b=selected["measured_probabilities_b"],
        fourier_pattern_similarity=float(selected["fourier_pattern_similarity"]),
        exact_distribution_similarity=float(selected["exact_distribution_similarity"]),
        measured_distribution_similarity=float(
            selected["measured_distribution_similarity"]
        ),
        absolute_error=absolute_error,
        relative_error=relative_error,
        negative_control_similarity=float(
            negative_control["measured_distribution_similarity"]
        ),
        strategy_results=strategy_results,
        negative_control=negative_control,
        task_time_seconds=task_time_seconds,
        num_qubits=circuit.num_qubits,
        circuit_depth=circuit.depth(),
        gate_count=_operation_count(circuit),
        transpiled_depth=transpiled_circuit.depth(),
        transpiled_gate_count=_operation_count(transpiled_circuit),
        shots=shots,
        backend="AerSimulator",
        pass_threshold=tolerance,
        pass_fail=pass_fail,
        threshold_reason=(
            "Measured QFT distribution similarity must stay within the "
            "configured tolerance of the exact Fourier distribution similarity. "
            "The negative control records behavior when no semantic prior is available."
        ),
        measurement_mode="shot-based QFT register measurements",
        claim_scope="implementation-level schema-pattern validation; no quantum-advantage claim",
        method_note=(
            "QFT maps small relation phase patterns into measured Fourier "
            "distributions. Phase assignments are explicit assumptions, with "
            "exact-URI, synonym-dictionary, and no-prior negative-control cases."
        ),
        claim_note=(
            "This validates the Table 1 Schema Matching -> Phase -> QFT "
            "mapping as a toy schema-pattern check, not full schema matching."
        ),
    )
