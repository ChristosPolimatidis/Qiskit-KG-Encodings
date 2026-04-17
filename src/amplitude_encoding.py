from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

from src.models import TripleRecord


def next_power_of_two_length(length: int) -> int:
    """Return the padded vector length, using at least one qubit."""

    if length < 1:
        raise ValueError("At least one triple is required for amplitude encoding.")
    if length == 1:
        return 2
    return 1 << math.ceil(math.log2(length))


def build_amplitude_vector(
    triples: list[TripleRecord],
    weights: list[float] | np.ndarray | None = None,
    strategy: str = "uniform",
) -> np.ndarray:
    """Build the unnormalized amplitude vector over triple indices.

    Basis state |i> corresponds to the i-th triple in the current triple list.
    """

    triple_count = len(triples)
    if triple_count < 1:
        raise ValueError("Amplitude encoding requires at least one triple.")

    if weights is not None:
        vector = np.asarray(weights, dtype=float)
        if vector.shape != (triple_count,):
            raise ValueError(
                "The custom weight vector must have one value per triple."
            )
        return vector

    if strategy == "uniform":
        return np.ones(triple_count, dtype=float)
    if strategy == "linear":
        return np.arange(1, triple_count + 1, dtype=float)

    raise ValueError(
        f"Unsupported amplitude strategy '{strategy}'. "
        "Use 'uniform', 'linear', or provide explicit weights."
    )


def pad_to_power_of_two(vector: np.ndarray) -> np.ndarray:
    """Pad a vector with zeros to the next power-of-two length."""

    padded_length = next_power_of_two_length(len(vector))
    padded_vector = np.zeros(padded_length, dtype=complex)
    padded_vector[: len(vector)] = vector
    return padded_vector


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector so it can be used as a quantum state."""

    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0.0):
        raise ValueError("The amplitude vector cannot be the zero vector.")
    return vector / norm


def prepare_amplitude_state(
    vector: np.ndarray,
    name: str | None = None,
) -> QuantumCircuit:
    """Prepare an amplitude-encoded quantum state from an input vector."""

    padded_vector = pad_to_power_of_two(np.asarray(vector, dtype=complex))
    normalized_vector = normalize_vector(padded_vector)
    return prepare_amplitude_state_from_normalized_vector(
        normalized_vector=normalized_vector,
        name=name,
    )


def prepare_amplitude_state_from_normalized_vector(
    normalized_vector: np.ndarray,
    name: str | None = None,
) -> QuantumCircuit:
    """Prepare an amplitude-encoded circuit from a normalized state vector."""

    normalized_vector = np.asarray(normalized_vector, dtype=complex)
    num_qubits = int(math.log2(len(normalized_vector)))

    circuit = QuantumCircuit(num_qubits, name=name or "AmplitudeEncoding")
    circuit.initialize(normalized_vector, circuit.qubits)
    return circuit


def build_amplitude_encoding_artifacts(
    triples: list[TripleRecord],
    weights: list[float] | np.ndarray | None = None,
    strategy: str = "uniform",
) -> dict[str, object]:
    """Build reusable metadata for the amplitude encoding pipeline."""

    raw_vector = build_amplitude_vector(
        triples=triples,
        weights=weights,
        strategy=strategy,
    )
    padded_vector = pad_to_power_of_two(raw_vector)
    normalized_vector = normalize_vector(padded_vector)
    num_qubits = int(math.log2(len(normalized_vector)))

    return {
        "strategy": "custom" if weights is not None else strategy,
        "raw_vector": raw_vector,
        "padded_vector": padded_vector,
        "normalized_vector": normalized_vector,
        "num_qubits": num_qubits,
        "index_labels": [
            {
                "triple_index": index,
                "basis_state": format(index, f"0{num_qubits}b"),
                "triple": triple.to_dict(),
            }
            for index, triple in enumerate(triples)
        ],
        "padded_basis_states": [
            format(index, f"0{num_qubits}b")
            for index in range(len(triples), len(padded_vector))
        ],
    }
