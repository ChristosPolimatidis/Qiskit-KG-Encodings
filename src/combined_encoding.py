from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from src.amplitude_encoding import next_power_of_two_length
from src.models import TripleRecord
from src.running_example import (
    PAPER_INDEX_DIMENSION,
    SEQUENTIAL_INDEX_DIMENSION,
    get_predicate_phase_map,
    get_running_example_indices,
    get_running_example_triples,
)


INDEX_MODES = ("sequential", "paper")


@dataclass(frozen=True, slots=True)
class CombinedEncodingResult:
    """Result bundle for combined amplitude and predicate-phase encoding."""

    statevector: np.ndarray
    num_qubits: int
    dimension: int
    amplitude_map: dict[int, float]
    phase_map: dict[str, float]
    nonzero_indices: list[int]
    preparation_time_seconds: float
    measurement_probabilities: dict[str, float]
    index_phase_map: dict[int, float]
    index_triple_map: dict[int, dict[str, str]]
    index_mode: str


def _validate_index_mode(index_mode: str) -> None:
    if index_mode not in INDEX_MODES:
        raise ValueError("Unsupported index mode. Use 'sequential' or 'paper'.")


def _is_default_running_example(triples: list[TripleRecord]) -> bool:
    return triples == get_running_example_triples()


def _sequential_indices(triples: list[TripleRecord]) -> dict[TripleRecord, int]:
    return {
        triple: index
        for index, triple in enumerate(triples)
    }


def _resolve_triples_and_indices(
    triples: list[TripleRecord] | None,
    index_mode: str,
    index_map: dict[TripleRecord, int] | None,
) -> tuple[list[TripleRecord], dict[TripleRecord, int], int]:
    _validate_index_mode(index_mode)

    selected_triples = get_running_example_triples() if triples is None else list(triples)
    if not selected_triples:
        raise ValueError("Combined encoding requires at least one triple.")

    if index_map is not None:
        selected_index_map = dict(index_map)
        missing_triples = [
            triple for triple in selected_triples if triple not in selected_index_map
        ]
        if missing_triples:
            raise ValueError("The custom index map must include every selected triple.")
        dimension = next_power_of_two_length(max(selected_index_map.values()) + 1)
        return selected_triples, selected_index_map, dimension

    if index_mode == "sequential":
        selected_index_map = (
            get_running_example_indices(mode="sequential")
            if _is_default_running_example(selected_triples)
            else _sequential_indices(selected_triples)
        )
        dimension = (
            SEQUENTIAL_INDEX_DIMENSION
            if _is_default_running_example(selected_triples)
            else next_power_of_two_length(len(selected_triples))
        )
        return selected_triples, selected_index_map, dimension

    if not _is_default_running_example(selected_triples):
        raise ValueError(
            "Paper index mode without a custom index map is defined for the "
            "canonical six-triple running example."
        )
    return (
        selected_triples,
        get_running_example_indices(mode="paper"),
        PAPER_INDEX_DIMENSION,
    )


def _normalized_amplitudes(
    triple_count: int,
    weights: list[float] | np.ndarray | None,
) -> np.ndarray:
    if weights is None:
        raw_amplitudes = np.ones(triple_count, dtype=float)
    else:
        raw_amplitudes = np.asarray(weights, dtype=float)
        if raw_amplitudes.shape != (triple_count,):
            raise ValueError("The weight vector must contain one value per triple.")
        if np.any(raw_amplitudes < 0):
            raise ValueError("Importance weights must be non-negative.")

    norm = np.linalg.norm(raw_amplitudes)
    if np.isclose(norm, 0.0):
        raise ValueError("At least one amplitude weight must be nonzero.")
    return raw_amplitudes / norm


def _probabilities_dict(statevector: np.ndarray, num_qubits: int) -> dict[str, float]:
    probabilities = np.abs(statevector) ** 2
    return {
        format(index, f"0{num_qubits}b"): float(probability)
        for index, probability in enumerate(probabilities)
        if probability > 1e-15
    }


def combined_amplitude_phase_encoding(
    triples: list[TripleRecord] | None = None,
    *,
    weights: list[float] | np.ndarray | None = None,
    index_mode: str = "sequential",
    index_map: dict[TripleRecord, int] | None = None,
    phase_map: dict[str, float] | None = None,
) -> CombinedEncodingResult:
    """Build |psi_G> = sum_k alpha_k exp(i theta_k) |k>.

    Non-existing indices receive amplitude zero. With no explicit triples, the
    canonical six-triple Chapter 9 running example is used.
    """

    start_time = time.perf_counter()
    selected_triples, selected_index_map, dimension = _resolve_triples_and_indices(
        triples=triples,
        index_mode=index_mode,
        index_map=index_map,
    )
    if dimension & (dimension - 1):
        raise ValueError("The index-space dimension must be a power of two.")

    num_qubits = int(math.log2(dimension))
    selected_phase_map = (
        get_predicate_phase_map() if phase_map is None else dict(phase_map)
    )
    normalized_amplitudes = _normalized_amplitudes(
        triple_count=len(selected_triples),
        weights=weights,
    )

    statevector = np.zeros(dimension, dtype=complex)
    amplitude_map: dict[int, float] = {}
    index_phase_map: dict[int, float] = {}
    index_triple_map: dict[int, dict[str, str]] = {}

    for triple, amplitude in zip(selected_triples, normalized_amplitudes):
        index = selected_index_map[triple]
        if index < 0 or index >= dimension:
            raise ValueError(
                f"Triple index {index} is outside dimension {dimension}."
            )
        if triple.predicate not in selected_phase_map:
            raise ValueError(
                f"No phase assignment exists for predicate '{triple.predicate}'."
            )
        phase = selected_phase_map[triple.predicate]
        statevector[index] = amplitude * np.exp(1j * phase)
        amplitude_map[index] = float(amplitude)
        index_phase_map[index] = float(phase)
        index_triple_map[index] = triple.to_dict()

    nonzero_indices = sorted(amplitude_map)
    preparation_time_seconds = time.perf_counter() - start_time

    return CombinedEncodingResult(
        statevector=statevector,
        num_qubits=num_qubits,
        dimension=dimension,
        amplitude_map=dict(sorted(amplitude_map.items())),
        phase_map=selected_phase_map,
        nonzero_indices=nonzero_indices,
        preparation_time_seconds=preparation_time_seconds,
        measurement_probabilities=_probabilities_dict(
            statevector=statevector,
            num_qubits=num_qubits,
        ),
        index_phase_map=dict(sorted(index_phase_map.items())),
        index_triple_map=dict(sorted(index_triple_map.items())),
        index_mode=index_mode,
    )


def build_combined_encoding(
    triples: list[TripleRecord] | None = None,
    *,
    weights: list[float] | np.ndarray | None = None,
    index_mode: str = "sequential",
    index_map: dict[TripleRecord, int] | None = None,
    phase_map: dict[str, float] | None = None,
) -> CombinedEncodingResult:
    """Alias for the Chapter 9 combined amplitude + phase encoding builder."""

    return combined_amplitude_phase_encoding(
        triples=triples,
        weights=weights,
        index_mode=index_mode,
        index_map=index_map,
        phase_map=phase_map,
    )
