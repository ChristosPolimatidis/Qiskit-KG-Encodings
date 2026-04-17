from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

from src.models import KGEncodingContext, TripleRecord


REGISTER_ORDER = ("subject", "predicate", "object")


def int_to_bitstring(value: int, width: int) -> str:
    """Convert an integer to a zero-padded binary string."""

    if value < 0:
        raise ValueError("Only non-negative IDs can be converted to bitstrings.")
    return format(value, f"0{width}b")


def triple_to_basis_components(
    triple: TripleRecord,
    context: KGEncodingContext,
) -> dict[str, object]:
    """Return IDs and bitstrings for a triple's basis representation."""

    subject_id, predicate_id, object_id = context.triple_numeric_ids(triple)

    subject_bits = int_to_bitstring(subject_id, context.entity_bit_width)
    predicate_bits = int_to_bitstring(predicate_id, context.predicate_bit_width)
    object_bits = int_to_bitstring(object_id, context.entity_bit_width)

    return {
        "subject_id": subject_id,
        "predicate_id": predicate_id,
        "object_id": object_id,
        "subject_bits": subject_bits,
        "predicate_bits": predicate_bits,
        "object_bits": object_bits,
        "register_order": REGISTER_ORDER,
    }


def triple_to_basis_bitstring(
    triple: TripleRecord,
    context: KGEncodingContext,
) -> str:
    """Build the displayed basis bitstring for a triple.

    The bitstring is shown in big-endian form as:
    subject_bits || predicate_bits || object_bits
    """

    components = triple_to_basis_components(triple=triple, context=context)
    return (
        f"{components['subject_bits']}"
        f"{components['predicate_bits']}"
        f"{components['object_bits']}"
    )


def build_basis_state_circuit(bitstring: str, name: str | None = None) -> QuantumCircuit:
    """Prepare a computational basis state for a displayed bitstring.

    Qiskit prints measured classical bitstrings in reverse qubit index order. This
    function sets qubits so that the default measured result matches `bitstring`.
    """

    if not bitstring:
        raise ValueError("The basis bitstring cannot be empty.")

    circuit = QuantumCircuit(len(bitstring), name=name or "BasisEncoding")
    for display_index, bit in enumerate(bitstring):
        if bit == "1":
            circuit.x(len(bitstring) - 1 - display_index)
    return circuit


def encode_triples_as_basis_states(
    context: KGEncodingContext,
) -> list[dict[str, object]]:
    """Convert all triples into basis-encoding metadata."""

    encoded_states: list[dict[str, object]] = []
    for triple_index, triple in enumerate(context.triples):
        components = triple_to_basis_components(triple=triple, context=context)
        bitstring = triple_to_basis_bitstring(triple=triple, context=context)
        encoded_states.append(
            {
                "triple_index": triple_index,
                "triple": triple.to_dict(),
                **components,
                "bitstring": bitstring,
            }
        )
    return encoded_states


def build_uniform_encoded_state_circuit(
    context: KGEncodingContext,
) -> QuantumCircuit:
    """Prepare a uniform superposition over the encoded triple basis states.

    This uses dense state initialization, which is practical for the small graphs
    targeted by the thesis example and simple local simulations.
    """

    encoded_states = encode_triples_as_basis_states(context=context)
    return build_uniform_encoded_state_circuit_from_encoded_states(
        encoded_states=encoded_states,
        num_qubits=context.total_basis_qubits,
    )


def build_uniform_encoded_state_circuit_from_encoded_states(
    encoded_states: list[dict[str, object]],
    num_qubits: int,
) -> QuantumCircuit:
    """Prepare a uniform superposition from precomputed encoded basis states."""

    vector_length = 2**num_qubits

    state_vector = np.zeros(vector_length, dtype=complex)
    amplitude = 1 / math.sqrt(len(encoded_states))

    for encoded_state in encoded_states:
        state_index = int(encoded_state["bitstring"], 2)
        state_vector[state_index] = amplitude

    circuit = QuantumCircuit(num_qubits, name="BasisUniformSuperposition")
    circuit.initialize(state_vector, circuit.qubits)
    return circuit
