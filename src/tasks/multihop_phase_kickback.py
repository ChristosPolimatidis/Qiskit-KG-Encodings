from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.running_example import (
    RDF_TYPE,
    RDFS_SUBCLASS_OF,
    get_predicate_phase_map,
    get_running_example_triples,
)


@dataclass(frozen=True, slots=True)
class MultiHopPhaseKickbackResult:
    """Result bundle for the Chapter 9 multi-hop phase-kickback validation."""

    task_name: str
    path: list[dict[str, str]]
    phase_terms: dict[str, float]
    expected_composed_phase: float
    observed_composed_phase: float
    phase_error: float
    task_time_seconds: float
    num_qubits: int
    circuit_depth: int
    gate_count: int
    method_note: str
    claim_note: str


def _operation_count(circuit: QuantumCircuit) -> int:
    return sum(int(count) for count in circuit.count_ops().values())


def _wrapped_phase(value: float) -> float:
    return float(value % (2 * math.pi))


def _phase_error(expected: float, observed: float) -> float:
    difference = (observed - expected + math.pi) % (2 * math.pi) - math.pi
    return float(abs(difference))


def build_multihop_phase_kickback_circuit(
    *,
    theta_type: float,
    theta_subclass: float,
) -> QuantumCircuit:
    """Build controlled phase operations for the two-hop running-example path."""

    circuit = QuantumCircuit(2, name="MultiHopPhaseKickback")
    control = 0
    target = 1
    circuit.x(control)
    circuit.x(target)
    circuit.cp(theta_type, control, target)
    circuit.cp(theta_subclass, control, target)
    return circuit


def run_multihop_phase_kickback_task() -> MultiHopPhaseKickbackResult:
    """Validate phase accumulation for one two-hop path.

    This is a phase-kickback/phase-accumulation check for the path
    Aristotle rdf:type Person, then Person rdfs:subClassOf Mortal. It is not a
    full RDFS reasoner.
    """

    task_start = time.perf_counter()
    triples = get_running_example_triples()
    path = [triples[0], triples[1]]
    phase_map = get_predicate_phase_map()
    theta_type = float(phase_map[RDF_TYPE])
    theta_subclass = float(phase_map[RDFS_SUBCLASS_OF])
    expected_phase = _wrapped_phase(theta_type + theta_subclass)

    circuit = build_multihop_phase_kickback_circuit(
        theta_type=theta_type,
        theta_subclass=theta_subclass,
    )
    statevector = Statevector.from_instruction(circuit).data
    observed_phase = _wrapped_phase(float(np.angle(statevector[3])))
    phase_error = _phase_error(expected_phase, observed_phase)
    task_time_seconds = time.perf_counter() - task_start

    return MultiHopPhaseKickbackResult(
        task_name="multihop_phase_kickback",
        path=[triple.to_dict() for triple in path],
        phase_terms={
            RDF_TYPE: theta_type,
            RDFS_SUBCLASS_OF: theta_subclass,
        },
        expected_composed_phase=expected_phase,
        observed_composed_phase=observed_phase,
        phase_error=phase_error,
        task_time_seconds=task_time_seconds,
        num_qubits=circuit.num_qubits,
        circuit_depth=circuit.depth(),
        gate_count=_operation_count(circuit),
        method_note=(
            "Controlled phase operations accumulate predicate phases along the "
            "two-hop path."
        ),
        claim_note=(
            "This validates the Table 1 Multi-hop Reasoning -> Phase -> Phase "
            "Kickback mapping. It is not a full RDFS reasoner."
        ),
    )
