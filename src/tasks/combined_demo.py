from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from src.combined_encoding import combined_amplitude_phase_encoding
from src.running_example import (
    EX,
    RDF_TYPE,
    RDFS_SUBCLASS_OF,
    get_predicate_phase_map,
    get_running_example_indices,
    get_running_example_triples,
)


DEFAULT_OUTPUT_PATH = Path("results/chapter9/tasks/combined_demo.json")
DEFAULT_CONFIDENCE_SCORES = (0.95, 0.90, 0.75, 0.80, 0.85, 0.70)


@dataclass(frozen=True, slots=True)
class CombinedDemoResult:
    """Result bundle for the Chapter 9 combined amplitude + phase demo."""

    task_name: str
    task_time_seconds: float
    num_qubits: int
    dimension: int
    preparation_time_seconds: float
    state_norm: float
    nonzero_indices: list[int]
    statevector: list[list[float]]
    amplitude_mode: str
    triple_encoding_details: list[dict[str, object]]
    decoded_index_mapping: dict[int, dict[str, str]]
    decoded_probabilities: list[dict[str, object]]
    composed_phase_examples: list[dict[str, object]]
    output_path: str | None
    index_mode: str
    claim_note: str


def _statevector_to_serializable(statevector: np.ndarray) -> list[list[float]]:
    return [
        [float(value.real), float(value.imag)]
        for value in statevector
    ]


def _bitstring_for_index(index: int, num_qubits: int) -> str:
    return format(index, f"0{num_qubits}b")


def _resolve_weights(
    confidence_scores: list[float] | tuple[float, ...] | None,
) -> tuple[str, list[float] | None]:
    if confidence_scores is None:
        return "uniform", None
    scores = [float(score) for score in confidence_scores]
    if len(scores) != len(get_running_example_triples()):
        raise ValueError("Confidence scores must contain one value per running-example triple.")
    return "confidence", scores


def _triple_details(
    *,
    statevector: np.ndarray,
    amplitude_map: dict[int, float],
    index_phase_map: dict[int, float],
    index_mode: str,
    num_qubits: int,
) -> list[dict[str, object]]:
    triple_to_index = get_running_example_indices(mode=index_mode)
    details: list[dict[str, object]] = []
    for triple in get_running_example_triples():
        index = triple_to_index[triple]
        coefficient = statevector[index]
        probability = float(abs(coefficient) ** 2)
        details.append(
            {
                "index": index,
                "bitstring": _bitstring_for_index(index, num_qubits),
                "triple": triple.to_dict(),
                "amplitude_magnitude": amplitude_map[index],
                "phase_angle": index_phase_map[index],
                "measurement_probability": probability,
                "coefficient": [float(coefficient.real), float(coefficient.imag)],
            }
        )
    return details


def _decoded_probabilities(
    triple_details: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "index": detail["index"],
            "bitstring": detail["bitstring"],
            "probability": detail["measurement_probability"],
            "triple": detail["triple"],
        }
        for detail in triple_details
    ]


def _phase_composition_examples() -> list[dict[str, object]]:
    triples = get_running_example_triples()
    phase_map = get_predicate_phase_map()
    first = triples[0]
    second = triples[1]
    composed_phase = phase_map[RDF_TYPE] + phase_map[RDFS_SUBCLASS_OF]

    return [
        {
            "label": "Aristotle type Person then Person subclassOf Mortal",
            "path": [first.to_dict(), second.to_dict()],
            "phase_terms": {
                RDF_TYPE: phase_map[RDF_TYPE],
                RDFS_SUBCLASS_OF: phase_map[RDFS_SUBCLASS_OF],
            },
            "composed_phase": composed_phase,
            "composed_phase_mod_2pi": math.fmod(composed_phase, 2 * math.pi),
            "interpretation": (
                "Phase composition demonstration only; this is not a complete "
                "implementation of RDFS reasoning."
            ),
        }
    ]


def _serializable_result(result: CombinedDemoResult) -> dict[str, object]:
    return {
        "task_name": result.task_name,
        "task_time_seconds": result.task_time_seconds,
        "num_qubits": result.num_qubits,
        "dimension": result.dimension,
        "preparation_time_seconds": result.preparation_time_seconds,
        "state_norm": result.state_norm,
        "nonzero_indices": result.nonzero_indices,
        "statevector": result.statevector,
        "amplitude_mode": result.amplitude_mode,
        "triple_encoding_details": result.triple_encoding_details,
        "decoded_index_mapping": {
            str(index): triple
            for index, triple in result.decoded_index_mapping.items()
        },
        "decoded_probabilities": result.decoded_probabilities,
        "composed_phase_examples": result.composed_phase_examples,
        "output_path": result.output_path,
        "claim_note": result.claim_note,
    }


def save_combined_demo_result(
    result: CombinedDemoResult,
    output_path: str | Path,
) -> Path:
    """Persist a combined-demo result as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_serializable_result(result), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return path


def run_combined_demo_task(
    *,
    index_mode: str = "sequential",
    confidence_scores: list[float] | tuple[float, ...] | None = None,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
) -> CombinedDemoResult:
    """Run the Chapter 9 combined amplitude + predicate-phase demonstration."""

    task_start = time.perf_counter()
    amplitude_mode, weights = _resolve_weights(confidence_scores)
    encoding = combined_amplitude_phase_encoding(
        weights=weights,
        index_mode=index_mode,
    )
    triple_details = _triple_details(
        statevector=encoding.statevector,
        amplitude_map=encoding.amplitude_map,
        index_phase_map=encoding.index_phase_map,
        index_mode=index_mode,
        num_qubits=encoding.num_qubits,
    )
    decoded_mapping = {
        index: triple
        for index, triple in encoding.index_triple_map.items()
    }
    task_time_seconds = time.perf_counter() - task_start
    output_string = str(Path(output_path)) if output_path is not None else None

    result = CombinedDemoResult(
        task_name="combined_amplitude_phase_demo",
        task_time_seconds=task_time_seconds,
        num_qubits=encoding.num_qubits,
        dimension=encoding.dimension,
        preparation_time_seconds=encoding.preparation_time_seconds,
        state_norm=float(np.linalg.norm(encoding.statevector)),
        nonzero_indices=encoding.nonzero_indices,
        statevector=_statevector_to_serializable(encoding.statevector),
        amplitude_mode=amplitude_mode,
        triple_encoding_details=triple_details,
        decoded_index_mapping=decoded_mapping,
        decoded_probabilities=_decoded_probabilities(triple_details),
        composed_phase_examples=_phase_composition_examples(),
        output_path=output_string,
        index_mode=index_mode,
        claim_note=(
            "This is a compact combined amplitude + phase encoding demonstration "
            "for Chapter 9. The phase-composition example is illustrative and is "
            "not full RDFS reasoning."
        ),
    )

    if output_path is not None:
        save_combined_demo_result(result, output_path)

    return result


def run_confidence_weighted_combined_demo(
    *,
    index_mode: str = "sequential",
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
) -> CombinedDemoResult:
    """Run the combined demo with deterministic example confidence scores."""

    return run_combined_demo_task(
        index_mode=index_mode,
        confidence_scores=DEFAULT_CONFIDENCE_SCORES,
        output_path=output_path,
    )
