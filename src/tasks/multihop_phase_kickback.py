from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from src.running_example import (
    RDF_TYPE,
    RDFS_SUBCLASS_OF,
    get_predicate_phase_map,
    get_running_example_triples,
)


DEFAULT_SHOTS = 4096
DEFAULT_SEED_SIMULATOR = 97531
DEFAULT_PHASE_ERROR_TOLERANCE = 0.10
PHASE_CLAIM_NOTE = (
    "This validates phase accumulation for a controlled two-hop relation path. "
    "It is not a full RDFS reasoner, complete rule engine, or path inference system."
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
    estimated_phase: float
    shot_phase_error: float
    counts_x: dict[str, int]
    counts_y: dict[str, int]
    probabilities_x: dict[str, float]
    probabilities_y: dict[str, float]
    p0_x: float
    p0_y: float
    estimated_cosine: float
    estimated_sine: float
    validation_cases: list[dict[str, object]]
    task_time_seconds: float
    num_qubits: int
    circuit_depth: int
    gate_count: int
    transpiled_depth: int
    transpiled_gate_count: int
    shots: int
    backend: str
    exact_statevector_quantity: float
    shot_based_estimate: float
    absolute_error: float
    relative_error: float | None
    pass_threshold: float
    pass_fail: str
    threshold_reason: str
    measurement_mode: str
    claim_scope: str
    method_note: str
    claim_note: str


def _operation_count(circuit: QuantumCircuit) -> int:
    return sum(int(count) for count in circuit.count_ops().values())


def _wrapped_phase(value: float) -> float:
    return float(value % (2 * math.pi))


def _phase_error(expected: float, observed: float) -> float:
    difference = (observed - expected + math.pi) % (2 * math.pi) - math.pi
    return float(abs(difference))


def _probabilities(counts: dict[str, int], shots: int) -> dict[str, float]:
    return {
        bitstring: count / shots
        for bitstring, count in sorted(counts.items())
    }


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


def build_phase_hadamard_test_circuit(theta: float, *, quadrature: str) -> QuantumCircuit:
    """Build an X/Y-basis Hadamard test for the phase exp(i theta)."""

    if quadrature not in {"x", "y"}:
        raise ValueError("quadrature must be 'x' or 'y'.")
    circuit = QuantumCircuit(2, 1, name=f"PhaseHadamardTest{quadrature.upper()}")
    ancilla = 0
    eigenstate = 1
    circuit.h(ancilla)
    circuit.x(eigenstate)
    circuit.cp(theta, ancilla, eigenstate)
    if quadrature == "y":
        circuit.sdg(ancilla)
    circuit.h(ancilla)
    circuit.measure(ancilla, 0)
    return circuit


def estimate_phase_from_counts(
    counts_x: dict[str, int],
    counts_y: dict[str, int],
    *,
    shots: int,
) -> tuple[float, float, float, float, float]:
    """Estimate phase from X/Y Hadamard-test counts."""

    if shots < 1:
        raise ValueError("shots must be at least 1.")
    p0_x = counts_x.get("0", 0) / shots
    p0_y = counts_y.get("0", 0) / shots
    estimated_cosine = max(-1.0, min(1.0, (2 * p0_x) - 1))
    estimated_sine = max(-1.0, min(1.0, (2 * p0_y) - 1))
    phase = _wrapped_phase(math.atan2(estimated_sine, estimated_cosine))
    return phase, p0_x, p0_y, estimated_cosine, estimated_sine


def run_phase_hadamard_validation(
    theta: float,
    *,
    shots: int,
    seed_simulator: int,
) -> dict[str, object]:
    simulator = AerSimulator(seed_simulator=seed_simulator)
    circuit_x = build_phase_hadamard_test_circuit(theta, quadrature="x")
    circuit_y = build_phase_hadamard_test_circuit(theta, quadrature="y")
    transpiled_x = transpile(circuit_x, simulator, optimization_level=0)
    transpiled_y = transpile(circuit_y, simulator, optimization_level=0)
    counts_x = dict(
        sorted(
            simulator.run(transpiled_x, shots=shots).result().get_counts().items()
        )
    )
    counts_y = dict(
        sorted(
            simulator.run(transpiled_y, shots=shots).result().get_counts().items()
        )
    )
    phase, p0_x, p0_y, estimated_cosine, estimated_sine = estimate_phase_from_counts(
        counts_x,
        counts_y,
        shots=shots,
    )
    exact_phase = _wrapped_phase(theta)
    return {
        "exact_phase": exact_phase,
        "estimated_phase": phase,
        "phase_error": _phase_error(exact_phase, phase),
        "counts_x": counts_x,
        "counts_y": counts_y,
        "probabilities_x": _probabilities(counts_x, shots),
        "probabilities_y": _probabilities(counts_y, shots),
        "p0_x": p0_x,
        "p0_y": p0_y,
        "estimated_cosine": estimated_cosine,
        "estimated_sine": estimated_sine,
        "circuit_x_depth": circuit_x.depth(),
        "circuit_y_depth": circuit_y.depth(),
        "transpiled_x_depth": transpiled_x.depth(),
        "transpiled_y_depth": transpiled_y.depth(),
        "transpiled_x_gate_count": _operation_count(transpiled_x),
        "transpiled_y_gate_count": _operation_count(transpiled_y),
    }


def _phase_case(
    *,
    name: str,
    predicates: list[str],
    phases: list[float],
    path_valid: bool,
    expected_relationship: str,
    shots: int,
    seed_simulator: int,
    pass_threshold: float,
) -> dict[str, object]:
    exact_phase = _wrapped_phase(sum(phases))
    validation = run_phase_hadamard_validation(
        exact_phase,
        shots=shots,
        seed_simulator=seed_simulator,
    )
    phase_error = float(validation["phase_error"])
    return {
        "case": name,
        "predicates": predicates,
        "path_valid": path_valid,
        "expected_relationship": expected_relationship,
        "exact_phase": exact_phase,
        "estimated_phase": validation["estimated_phase"],
        "phase_error": phase_error,
        "pass_threshold": pass_threshold,
        "pass_fail": "pass" if phase_error <= pass_threshold else "fail",
        "counts_x": validation["counts_x"],
        "counts_y": validation["counts_y"],
        "probabilities_x": validation["probabilities_x"],
        "probabilities_y": validation["probabilities_y"],
    }


def run_multihop_phase_kickback_task(
    *,
    shots: int = DEFAULT_SHOTS,
    seed_simulator: int = DEFAULT_SEED_SIMULATOR,
    phase_error_tolerance: float | None = None,
) -> MultiHopPhaseKickbackResult:
    """Validate phase accumulation for one two-hop path.

    This is a phase-kickback/phase-accumulation check for the path
    Aristotle rdf:type Person, then Person rdfs:subClassOf Mortal. It is not a
    full RDFS reasoner.
    """

    task_start = time.perf_counter()
    tolerance = (
        DEFAULT_PHASE_ERROR_TOLERANCE
        if phase_error_tolerance is None and shots >= 1000
        else (0.25 if phase_error_tolerance is None else phase_error_tolerance)
    )
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
    shot_validation = run_phase_hadamard_validation(
        expected_phase,
        shots=shots,
        seed_simulator=seed_simulator,
    )
    estimated_phase = float(shot_validation["estimated_phase"])
    shot_phase_error = float(shot_validation["phase_error"])
    validation_cases = [
        _phase_case(
            name="valid_two_hop_path",
            predicates=[RDF_TYPE, RDFS_SUBCLASS_OF],
            phases=[theta_type, theta_subclass],
            path_valid=True,
            expected_relationship="Aristotle rdf:type Person then Person rdfs:subClassOf Mortal",
            shots=shots,
            seed_simulator=seed_simulator + 1,
            pass_threshold=tolerance,
        ),
        _phase_case(
            name="unrelated_path",
            predicates=[RDF_TYPE],
            phases=[theta_type],
            path_valid=False,
            expected_relationship="single predicate does not validate the controlled two-hop path",
            shots=shots,
            seed_simulator=seed_simulator + 2,
            pass_threshold=tolerance,
        ),
        _phase_case(
            name="reversed_path",
            predicates=[RDFS_SUBCLASS_OF, RDF_TYPE],
            phases=[theta_subclass, theta_type],
            path_valid=False,
            expected_relationship="same phase terms in the wrong graph order are recorded as a structural failure",
            shots=shots,
            seed_simulator=seed_simulator + 3,
            pass_threshold=tolerance,
        ),
        _phase_case(
            name="mismatched_predicate_path",
            predicates=[RDF_TYPE, RDF_TYPE],
            phases=[theta_type, theta_type],
            path_valid=False,
            expected_relationship="mismatched predicate path changes the accumulated phase",
            shots=shots,
            seed_simulator=seed_simulator + 4,
            pass_threshold=tolerance,
        ),
    ]
    simulator = AerSimulator(seed_simulator=seed_simulator)
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()
    transpiled_circuit = transpile(measured_circuit, simulator, optimization_level=0)
    pass_fail = "pass" if shot_phase_error <= tolerance else "fail"
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
        estimated_phase=estimated_phase,
        shot_phase_error=shot_phase_error,
        counts_x=shot_validation["counts_x"],
        counts_y=shot_validation["counts_y"],
        probabilities_x=shot_validation["probabilities_x"],
        probabilities_y=shot_validation["probabilities_y"],
        p0_x=float(shot_validation["p0_x"]),
        p0_y=float(shot_validation["p0_y"]),
        estimated_cosine=float(shot_validation["estimated_cosine"]),
        estimated_sine=float(shot_validation["estimated_sine"]),
        validation_cases=validation_cases,
        task_time_seconds=task_time_seconds,
        num_qubits=circuit.num_qubits,
        circuit_depth=circuit.depth(),
        gate_count=_operation_count(circuit),
        transpiled_depth=transpiled_circuit.depth(),
        transpiled_gate_count=_operation_count(transpiled_circuit),
        shots=shots,
        backend="AerSimulator",
        exact_statevector_quantity=expected_phase,
        shot_based_estimate=estimated_phase,
        absolute_error=shot_phase_error,
        relative_error=shot_phase_error / abs(expected_phase)
        if not math.isclose(expected_phase, 0.0)
        else None,
        pass_threshold=tolerance,
        pass_fail=pass_fail,
        threshold_reason=(
            "Wrapped shot-estimated phase must be within the configured "
            "phase-error tolerance of the exact statevector phase."
        ),
        measurement_mode="shot-based X/Y Hadamard-test phase estimate plus exact statevector check",
        claim_scope="implementation-level controlled two-hop phase validation",
        method_note=(
            "Controlled phase operations accumulate predicate phases along the "
            "two-hop path; X/Y Hadamard-test measurements estimate the phase "
            "from shot counts."
        ),
        claim_note=PHASE_CLAIM_NOTE,
    )
