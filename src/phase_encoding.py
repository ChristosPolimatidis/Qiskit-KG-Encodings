from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from qiskit import QuantumCircuit

from src.amplitude_encoding import next_power_of_two_length
from src.models import KGEncodingContext, TripleRecord


PhaseMarkFunction = Callable[[TripleRecord, int, KGEncodingContext], float]


def zero_phase_marker(
    triple: TripleRecord,
    triple_index: int,
    context: KGEncodingContext,
) -> float:
    """Default marker that applies no phase shift."""

    del triple, triple_index, context
    return 0.0


def predicate_phase_marker(
    predicate_uri: str,
    phase_value: float = math.pi,
) -> PhaseMarkFunction:
    """Return a marker that flags triples with a matching predicate URI."""

    def mark_fn(
        triple: TripleRecord,
        triple_index: int,
        context: KGEncodingContext,
    ) -> float:
        del triple_index, context
        return phase_value if triple.predicate == predicate_uri else 0.0

    return mark_fn


def subject_phase_marker(
    subject_uri: str,
    phase_value: float = math.pi,
) -> PhaseMarkFunction:
    """Return a marker that flags triples with a matching subject URI."""

    def mark_fn(
        triple: TripleRecord,
        triple_index: int,
        context: KGEncodingContext,
    ) -> float:
        del triple_index, context
        return phase_value if triple.subject == subject_uri else 0.0

    return mark_fn


def condition_phase_marker(
    condition_fn: Callable[[TripleRecord, int, KGEncodingContext], bool],
    phase_value: float = math.pi,
) -> PhaseMarkFunction:
    """Convert a boolean condition into a phase-marking function."""

    def mark_fn(
        triple: TripleRecord,
        triple_index: int,
        context: KGEncodingContext,
    ) -> float:
        return phase_value if condition_fn(triple, triple_index, context) else 0.0

    return mark_fn


def build_uniform_index_state(triple_count: int) -> np.ndarray:
    """Create a uniform superposition over the valid triple indices."""

    padded_length = next_power_of_two_length(triple_count)
    state_vector = np.zeros(padded_length, dtype=complex)
    state_vector[:triple_count] = 1.0
    return state_vector / np.linalg.norm(state_vector)


def compute_phase_angles(
    context: KGEncodingContext,
    mark_fn: PhaseMarkFunction,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Compute per-index phase angles and a readable list of marked triples."""

    phase_angles = np.zeros(next_power_of_two_length(context.triple_count), dtype=float)
    marked_triples: list[dict[str, object]] = []

    for triple_index, triple in enumerate(context.triples):
        phase_value = float(mark_fn(triple, triple_index, context))
        phase_angles[triple_index] = phase_value

        if not np.isclose(np.mod(phase_value, 2 * math.pi), 0.0):
            marked_triples.append(
                {
                    "triple_index": triple_index,
                    "basis_state": format(
                        triple_index,
                        f"0{int(math.log2(len(phase_angles)))}b",
                    ),
                    "phase_value": phase_value,
                    "triple": triple.to_dict(),
                }
            )

    return phase_angles, marked_triples


def apply_phase_oracle(
    circuit: QuantumCircuit,
    phase_angles: np.ndarray,
) -> None:
    """Apply a dense diagonal phase oracle.

    This is intentionally simple and readable for small local simulations.
    """

    diagonal_entries = np.exp(1j * np.asarray(phase_angles, dtype=float))
    circuit.unitary(
        np.diag(diagonal_entries),
        circuit.qubits,
        label="PhaseOracle",
    )


def build_phase_marked_circuit(
    context: KGEncodingContext,
    mark_fn: PhaseMarkFunction,
) -> tuple[QuantumCircuit, np.ndarray, list[dict[str, object]], np.ndarray]:
    """Prepare a uniform index state and apply the chosen phase shifts."""

    initial_state = build_uniform_index_state(context.triple_count)
    num_qubits = int(math.log2(len(initial_state)))
    phase_angles, marked_triples = compute_phase_angles(context=context, mark_fn=mark_fn)

    circuit = QuantumCircuit(num_qubits, name="PhaseMarked")
    circuit.initialize(initial_state, circuit.qubits)
    apply_phase_oracle(circuit=circuit, phase_angles=phase_angles)
    return circuit, phase_angles, marked_triples, initial_state


def build_phase_interference_circuit(
    context: KGEncodingContext,
    mark_fn: PhaseMarkFunction,
) -> tuple[QuantumCircuit, np.ndarray, list[dict[str, object]], np.ndarray]:
    """Prepare the marked phase state and apply a Hadamard mixing step."""

    circuit, phase_angles, marked_triples, initial_state = build_phase_marked_circuit(
        context=context,
        mark_fn=mark_fn,
    )
    circuit = build_phase_interference_from_marked_circuit(circuit)
    return circuit, phase_angles, marked_triples, initial_state


def build_phase_interference_from_marked_circuit(
    marked_circuit: QuantumCircuit,
) -> QuantumCircuit:
    """Apply the mixing step to an already prepared phase-marked circuit."""

    circuit = marked_circuit.copy()
    circuit.h(range(circuit.num_qubits))
    circuit.name = "PhaseInterference"
    return circuit
