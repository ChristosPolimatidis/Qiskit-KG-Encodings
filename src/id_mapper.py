from __future__ import annotations

import math
from pathlib import Path

from src.models import KGEncodingContext, TripleRecord


THESIS_ENTITY_ORDER = [
    "http://example.org/Aristotle",
    "http://example.org/Person",
    "http://example.org/Mortal",
    "http://example.org/Athens",
    "http://example.org/City",
    "http://example.org/Place",
    "http://example.org/Philosophy",
]

THESIS_PREDICATE_ORDER = [
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://example.org/livesAt",
    "http://example.org/teaches",
]

THESIS_TRIPLE_ORDER = [
    TripleRecord(
        "http://example.org/Aristotle",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://example.org/Person",
    ),
    TripleRecord(
        "http://example.org/Person",
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "http://example.org/Mortal",
    ),
    TripleRecord(
        "http://example.org/Aristotle",
        "http://example.org/livesAt",
        "http://example.org/Athens",
    ),
    TripleRecord(
        "http://example.org/Athens",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://example.org/City",
    ),
    TripleRecord(
        "http://example.org/City",
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "http://example.org/Place",
    ),
    TripleRecord(
        "http://example.org/Aristotle",
        "http://example.org/teaches",
        "http://example.org/Philosophy",
    ),
]


def required_bit_width(item_count: int) -> int:
    """Compute the bit width needed to represent a mapping space."""

    if item_count < 1:
        raise ValueError("At least one mapped item is required.")
    return max(1, math.ceil(math.log2(max(2, item_count))))


def build_ordered_vocabulary(
    values: list[str],
    preferred_order: list[str] | None = None,
) -> list[str]:
    """Build a deterministic vocabulary with an optional preferred prefix order."""

    unique_values = set(values)
    ordered_values: list[str] = []

    if preferred_order is not None:
        ordered_values.extend(
            value for value in preferred_order if value in unique_values
        )

    remaining_values = sorted(unique_values.difference(ordered_values))
    ordered_values.extend(remaining_values)
    return ordered_values


def order_triples(
    triples: list[TripleRecord],
    fixed_thesis_mapping: bool = False,
) -> list[TripleRecord]:
    """Return deterministic triples, optionally using the thesis example order."""

    lexicographic_order = sorted(triples, key=lambda triple: triple.as_tuple())
    if not fixed_thesis_mapping:
        return lexicographic_order

    thesis_positions = {
        triple.as_tuple(): index for index, triple in enumerate(THESIS_TRIPLE_ORDER)
    }
    return sorted(
        triples,
        key=lambda triple: (
            0 if triple.as_tuple() in thesis_positions else 1,
            thesis_positions.get(triple.as_tuple(), 0),
            triple.as_tuple(),
        ),
    )


def build_encoding_context(
    triples: list[TripleRecord],
    fixed_thesis_mapping: bool = False,
    dataset_path: str | Path | None = None,
) -> KGEncodingContext:
    """Create the shared encoding context for a parsed graph."""

    if not triples:
        raise ValueError("The RDF graph does not contain any triples.")

    ordered_triples = order_triples(
        triples=triples,
        fixed_thesis_mapping=fixed_thesis_mapping,
    )

    entity_values = [triple.subject for triple in ordered_triples] + [
        triple.object for triple in ordered_triples
    ]
    predicate_values = [triple.predicate for triple in ordered_triples]

    entity_order = (
        THESIS_ENTITY_ORDER if fixed_thesis_mapping else None
    )
    predicate_order = (
        THESIS_PREDICATE_ORDER if fixed_thesis_mapping else None
    )

    ordered_entities = build_ordered_vocabulary(
        values=entity_values,
        preferred_order=entity_order,
    )
    ordered_predicates = build_ordered_vocabulary(
        values=predicate_values,
        preferred_order=predicate_order,
    )

    entity_to_id = {
        entity: identifier for identifier, entity in enumerate(ordered_entities)
    }
    predicate_to_id = {
        predicate: identifier
        for identifier, predicate in enumerate(ordered_predicates)
    }

    return KGEncodingContext(
        triples=ordered_triples,
        entity_to_id=entity_to_id,
        predicate_to_id=predicate_to_id,
        entity_bit_width=required_bit_width(len(entity_to_id)),
        predicate_bit_width=required_bit_width(len(predicate_to_id)),
        fixed_thesis_mapping=fixed_thesis_mapping,
        dataset_path=str(Path(dataset_path).resolve()) if dataset_path else None,
    )
