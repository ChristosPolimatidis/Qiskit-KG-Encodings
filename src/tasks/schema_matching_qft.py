from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector

from src.running_example import RDF_TYPE, RDFS_SUBCLASS_OF, get_predicate_phase_map


@dataclass(frozen=True, slots=True)
class SchemaMatchingQFTResult:
    """Result bundle for the Chapter 9 schema-pattern QFT validation."""

    task_name: str
    relation_a_label: str
    relation_b_label: str
    phase_pattern_a: list[float]
    phase_pattern_b: list[float]
    fourier_magnitudes_a: list[float]
    fourier_magnitudes_b: list[float]
    fourier_pattern_similarity: float
    task_time_seconds: float
    num_qubits: int
    circuit_depth: int
    gate_count: int
    method_note: str
    claim_note: str


def _operation_count(circuit: QuantumCircuit) -> int:
    return sum(int(count) for count in circuit.count_ops().values())


def _phase_state(pattern: list[float]) -> np.ndarray:
    dimension = len(pattern)
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("QFT phase patterns must have power-of-two length.")
    return np.exp(1j * np.asarray(pattern, dtype=float)) / math.sqrt(dimension)


def _qft_magnitudes(pattern: list[float]) -> tuple[np.ndarray, QuantumCircuit]:
    state = _phase_state(pattern)
    num_qubits = int(math.log2(len(state)))
    circuit = QuantumCircuit(num_qubits, name="SchemaPatternQFT")
    circuit.initialize(state, range(num_qubits))
    circuit.append(QFTGate(num_qubits), range(num_qubits))
    transformed = Statevector.from_instruction(circuit).data
    return np.abs(transformed), circuit


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if np.isclose(denominator, 0.0):
        raise ValueError("Cannot compare zero Fourier magnitude vectors.")
    return float(np.dot(vector_a, vector_b) / denominator)


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


def run_schema_matching_qft_task() -> SchemaMatchingQFTResult:
    """Run a toy QFT validation over two relation phase patterns."""

    task_start = time.perf_counter()
    phase_map = get_predicate_phase_map()
    theta_type = float(phase_map[RDF_TYPE])
    theta_subclass = float(phase_map[RDFS_SUBCLASS_OF])
    phase_pattern_a = [theta_type, theta_subclass, theta_type, 0.0]
    phase_pattern_b = [theta_type, theta_subclass, theta_subclass, 0.0]

    magnitudes_a, _ = _qft_magnitudes(phase_pattern_a)
    magnitudes_b, _ = _qft_magnitudes(phase_pattern_b)
    similarity = _cosine_similarity(magnitudes_a, magnitudes_b)
    circuit = _combined_schema_circuit(phase_pattern_a, phase_pattern_b)
    task_time_seconds = time.perf_counter() - task_start

    return SchemaMatchingQFTResult(
        task_name="schema_matching_qft",
        relation_a_label="schema_pattern_type_subclass_type",
        relation_b_label="schema_pattern_type_subclass_subclass",
        phase_pattern_a=[float(value) for value in phase_pattern_a],
        phase_pattern_b=[float(value) for value in phase_pattern_b],
        fourier_magnitudes_a=[float(value) for value in magnitudes_a],
        fourier_magnitudes_b=[float(value) for value in magnitudes_b],
        fourier_pattern_similarity=similarity,
        task_time_seconds=task_time_seconds,
        num_qubits=circuit.num_qubits,
        circuit_depth=circuit.depth(),
        gate_count=_operation_count(circuit),
        method_note=(
            "QFT maps small relation phase patterns into Fourier magnitude "
            "signatures for comparison."
        ),
        claim_note=(
            "This validates the Table 1 Schema Matching -> Phase -> QFT "
            "mapping as a toy schema-pattern check, not full schema matching."
        ),
    )
