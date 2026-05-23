from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from qiskit import QuantumCircuit

from src.models import TripleRecord
from src.running_example import (
    PAPER_INDEX_DIMENSION,
    SEQUENTIAL_INDEX_DIMENSION,
    get_running_example_indices,
    get_running_example_triples,
)


INDEX_MODES = ("sequential", "paper")


@dataclass(frozen=True, slots=True)
class SparseIndexAmplitudeEncodingResult:
    """Result bundle for sparse index-aligned amplitude encoding."""

    statevector: np.ndarray
    num_qubits: int
    dimension: int
    nonzero_indices: list[int]
    normalized_amplitudes: list[float]
    measurement_probabilities: dict[str, float]
    preparation_time_seconds: float
    index_mode: str
    index_map: dict[TripleRecord, int]


def next_power_of_two_length(length: int) -> int:
    """Return the padded vector length, using at least one qubit."""

    if length < 1:
        raise ValueError("At least one triple is required for amplitude encoding.")
    if length == 1:
        return 2
    return 1 << math.ceil(math.log2(length))


def _is_running_example(triples: list[TripleRecord]) -> bool:
    return triples == get_running_example_triples()


def _sequential_index_map(triples: list[TripleRecord]) -> dict[TripleRecord, int]:
    return {
        triple: index
        for index, triple in enumerate(triples)
    }


def _resolve_sparse_index_map(
    triples: list[TripleRecord],
    index_mode: str,
    index_map: dict[TripleRecord, int] | None,
) -> tuple[dict[TripleRecord, int], int]:
    if index_mode not in INDEX_MODES:
        raise ValueError("Unsupported index mode. Use 'sequential' or 'paper'.")

    if index_map is not None:
        selected_index_map = dict(index_map)
        missing_triples = [
            triple for triple in triples if triple not in selected_index_map
        ]
        if missing_triples:
            raise ValueError("The custom index map must include every triple.")
        dimension = next_power_of_two_length(max(selected_index_map.values()) + 1)
        return selected_index_map, dimension

    if index_mode == "sequential":
        selected_index_map = (
            get_running_example_indices(mode="sequential")
            if _is_running_example(triples)
            else _sequential_index_map(triples)
        )
        dimension = (
            SEQUENTIAL_INDEX_DIMENSION
            if _is_running_example(triples)
            else next_power_of_two_length(len(triples))
        )
        return selected_index_map, dimension

    if not _is_running_example(triples):
        raise ValueError(
            "Paper index mode without a custom index map is defined for the "
            "canonical six-triple running example."
        )
    return get_running_example_indices(mode="paper"), PAPER_INDEX_DIMENSION


def _probabilities_dict(statevector: np.ndarray, num_qubits: int) -> dict[str, float]:
    probabilities = np.abs(statevector) ** 2
    return {
        format(index, f"0{num_qubits}b"): float(probability)
        for index, probability in enumerate(probabilities)
        if probability > 1e-15
    }


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


def build_sparse_index_amplitude_encoding(
    triples: list[TripleRecord],
    index_mode: str = "sequential",
    weights: list[float] | np.ndarray | None = None,
    index_map: dict[TripleRecord, int] | None = None,
) -> SparseIndexAmplitudeEncodingResult:
    """Build an amplitude state over a sparse index space.

    Non-existing indices receive amplitude zero. Sequential mode maps triples to
    contiguous indices unless the canonical running example mapping is used.
    Paper mode uses the running-example paper indices unless a custom index map
    is supplied.
    """

    start_time = time.perf_counter()
    selected_triples = list(triples)
    if not selected_triples:
        raise ValueError("Sparse index amplitude encoding requires at least one triple.")

    selected_index_map, dimension = _resolve_sparse_index_map(
        triples=selected_triples,
        index_mode=index_mode,
        index_map=index_map,
    )
    raw_amplitudes = build_amplitude_vector(
        triples=selected_triples,
        weights=weights,
        strategy="uniform",
    )
    normalized_amplitudes = normalize_vector(raw_amplitudes).astype(complex)
    statevector = np.zeros(dimension, dtype=complex)

    for triple, amplitude in zip(selected_triples, normalized_amplitudes):
        index = selected_index_map[triple]
        if index < 0 or index >= dimension:
            raise ValueError(
                f"Triple index {index} is outside dimension {dimension}."
            )
        statevector[index] = amplitude

    num_qubits = int(math.log2(dimension))
    nonzero_indices = sorted(
        index for index, value in enumerate(statevector) if abs(value) > 1e-15
    )
    return SparseIndexAmplitudeEncodingResult(
        statevector=statevector,
        num_qubits=num_qubits,
        dimension=dimension,
        nonzero_indices=nonzero_indices,
        normalized_amplitudes=[
            float(value.real)
            for value in normalized_amplitudes
        ],
        measurement_probabilities=_probabilities_dict(
            statevector=statevector,
            num_qubits=num_qubits,
        ),
        preparation_time_seconds=time.perf_counter() - start_time,
        index_mode=index_mode,
        index_map={
            triple: selected_index_map[triple]
            for triple in selected_triples
        },
    )


def build_paper_amplitude_encoding(
    triples: list[TripleRecord],
    index_mode: str = "paper",
    weights: list[float] | np.ndarray | None = None,
    index_map: dict[TripleRecord, int] | None = None,
) -> SparseIndexAmplitudeEncodingResult:
    """Alias for sparse index-aligned paper amplitude encoding."""

    return build_sparse_index_amplitude_encoding(
        triples=triples,
        index_mode=index_mode,
        weights=weights,
        index_map=index_map,
    )
