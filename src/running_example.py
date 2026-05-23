from __future__ import annotations

import math

from rdflib import Graph, URIRef

from src.models import TripleRecord


EX = "http://example.org/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"

SEQUENTIAL_INDEX_DIMENSION = 8
PAPER_INDEX_DIMENSION = 256

RUNNING_EXAMPLE_TRIPLES: tuple[TripleRecord, ...] = (
    TripleRecord(f"{EX}Aristotle", RDF_TYPE, f"{EX}Person"),
    TripleRecord(f"{EX}Person", RDFS_SUBCLASS_OF, f"{EX}Mortal"),
    TripleRecord(f"{EX}Aristotle", f"{EX}livesAt", f"{EX}Athens"),
    TripleRecord(f"{EX}Athens", RDF_TYPE, f"{EX}City"),
    TripleRecord(f"{EX}City", RDFS_SUBCLASS_OF, f"{EX}Place"),
    TripleRecord(f"{EX}Aristotle", f"{EX}teaches", f"{EX}Philosophy"),
)

PAPER_TRIPLE_INDICES: tuple[int, ...] = (1, 17, 27, 37, 88, 124)

PREDICATE_PHASE_MAP: dict[str, float] = {
    RDF_TYPE: math.pi / 4,
    RDFS_SUBCLASS_OF: math.pi / 2,
    f"{EX}livesAt": 3 * math.pi / 4,
    f"{EX}teaches": math.pi,
}


def get_running_example_triples() -> list[TripleRecord]:
    """Return the canonical six paper triples in deterministic order."""

    return list(RUNNING_EXAMPLE_TRIPLES)


def get_running_example_graph() -> Graph:
    """Return the canonical running example as an RDFLib graph."""

    graph = Graph()
    for triple in RUNNING_EXAMPLE_TRIPLES:
        graph.add(
            (
                URIRef(triple.subject),
                URIRef(triple.predicate),
                URIRef(triple.object),
            )
        )
    return graph


def get_running_example_indices(mode: str = "sequential") -> dict[TripleRecord, int]:
    """Return the index assignment for the running example triples.

    The sequential mode uses indices 0 through 5 and is padded to dimension 8 by
    index-based encodings. The paper mode uses the paper's sparse indices and
    therefore requires dimension 256, or 8 qubits, for index-based encodings.
    """

    if mode == "sequential":
        return {
            triple: index
            for index, triple in enumerate(RUNNING_EXAMPLE_TRIPLES)
        }

    if mode == "paper":
        return {
            triple: PAPER_TRIPLE_INDICES[index]
            for index, triple in enumerate(RUNNING_EXAMPLE_TRIPLES)
        }

    raise ValueError("Unsupported index mode. Use 'sequential' or 'paper'.")


def get_predicate_phase_map() -> dict[str, float]:
    """Return deterministic phase assignments for running-example predicates."""

    return dict(PREDICATE_PHASE_MAP)
