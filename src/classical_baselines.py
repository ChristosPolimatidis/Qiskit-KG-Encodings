from __future__ import annotations

from dataclasses import dataclass, asdict
import time
from typing import Any

import numpy as np

from src.running_example import get_running_example_indices, get_running_example_triples
from src.tasks.amplitude_similarity import (
    DEFAULT_ENTITY_A,
    DEFAULT_ENTITY_B,
    classical_squared_similarity,
    entity_context_vector,
    normalize_and_pad,
)
from src.tasks.keyword_search import (
    ENTITY_FEATURE_VECTORS,
    QUERY_VECTOR,
    classical_keyword_scores,
)
from src.tasks.link_prediction_distance import (
    DEFAULT_HEAD_ENTITY,
    DEFAULT_RELATION,
    DEFAULT_TAIL_ENTITY,
    ENTITY_VECTORS,
    RELATION_VECTORS,
)


BASELINE_SCOPE = (
    "Implementation-level deterministic sanity baselines for the same toy "
    "inputs used by the quantum validation tasks. These are not graph-database "
    "benchmarks and do not establish quantum advantage."
)


@dataclass(frozen=True, slots=True)
class ClassicalBaselineResult:
    task: str
    baseline_method: str
    input_size: int
    expected_value: Any
    measured_value: Any
    runtime_seconds: float
    notes: str
    claim_scope: str = BASELINE_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_exact_lookup_baseline() -> ClassicalBaselineResult:
    triples = get_running_example_triples()
    target = triples[-1]
    start = time.perf_counter()
    triple_to_index = get_running_example_indices(mode="sequential")
    index_to_triple = {
        index: triple
        for triple, index in triple_to_index.items()
    }
    target_index = triple_to_index[target]
    measured = index_to_triple[target_index].to_dict()
    elapsed = time.perf_counter() - start
    return ClassicalBaselineResult(
        task="search_grover_lookup",
        baseline_method="python_dict_exact_lookup",
        input_size=len(triples),
        expected_value=target.to_dict(),
        measured_value=measured,
        runtime_seconds=elapsed,
        notes="Exact dictionary lookup over deterministic running-example indices.",
    )


def run_entity_similarity_baseline(
    *,
    entity_a: str = DEFAULT_ENTITY_A,
    entity_b: str = DEFAULT_ENTITY_B,
) -> ClassicalBaselineResult:
    start = time.perf_counter()
    vector_a = normalize_and_pad(entity_context_vector(entity_a))
    vector_b = normalize_and_pad(entity_context_vector(entity_b))
    similarity = classical_squared_similarity(vector_a, vector_b)
    elapsed = time.perf_counter() - start
    return ClassicalBaselineResult(
        task="entity_matching_swap_test",
        baseline_method="numpy_exact_squared_dot_product",
        input_size=int(vector_a.size),
        expected_value=similarity,
        measured_value=similarity,
        runtime_seconds=elapsed,
        notes="Exact NumPy squared inner product for the same feature vectors.",
    )


def run_link_distance_baseline(
    *,
    head_entity: str = DEFAULT_HEAD_ENTITY,
    relation: str = DEFAULT_RELATION,
    tail_entity: str = DEFAULT_TAIL_ENTITY,
) -> ClassicalBaselineResult:
    start = time.perf_counter()
    head = np.asarray(ENTITY_VECTORS[head_entity], dtype=float)
    relation_vector = np.asarray(RELATION_VECTORS[relation], dtype=float)
    tail = np.asarray(ENTITY_VECTORS[tail_entity], dtype=float)
    source = head + relation_vector
    distance = float(np.linalg.norm(source - tail))
    elapsed = time.perf_counter() - start
    return ClassicalBaselineResult(
        task="link_prediction_distance_estimation",
        baseline_method="numpy_exact_euclidean_distance",
        input_size=int(source.size),
        expected_value=distance,
        measured_value=distance,
        runtime_seconds=elapsed,
        notes="Exact NumPy Euclidean distance for the same toy TransE vectors.",
    )


def run_keyword_search_baseline() -> ClassicalBaselineResult:
    start = time.perf_counter()
    scores = classical_keyword_scores()
    ranking = sorted(scores, key=lambda entity: (-scores[entity], entity))
    elapsed = time.perf_counter() - start
    return ClassicalBaselineResult(
        task="keyword_search_swap_test",
        baseline_method="numpy_exact_keyword_vector_scores",
        input_size=len(QUERY_VECTOR) * len(ENTITY_FEATURE_VECTORS),
        expected_value=scores,
        measured_value={"ranking": ranking, "top_result": ranking[0], "scores": scores},
        runtime_seconds=elapsed,
        notes="Exact NumPy scores for the same query and entity feature vectors.",
    )


def run_all_classical_baselines() -> list[ClassicalBaselineResult]:
    return [
        run_exact_lookup_baseline(),
        run_entity_similarity_baseline(),
        run_link_distance_baseline(),
        run_keyword_search_baseline(),
    ]
