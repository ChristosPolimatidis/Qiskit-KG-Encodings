from __future__ import annotations

from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef

from src.models import TripleRecord


SUPPORTED_RDF_FORMATS = {
    ".ttl": "turtle",
    ".rdf": "xml",
    ".xml": "xml",
    ".nt": "nt",
}

FALLBACK_RDF_FORMATS = ("turtle", "xml", "nt")


def infer_rdf_format(file_path: str | Path) -> str:
    """Infer the RDF parser format from the file suffix."""

    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_RDF_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_RDF_FORMATS))
        raise ValueError(
            f"Unsupported RDF file suffix '{suffix}'. Supported suffixes: {supported}."
        )
    return SUPPORTED_RDF_FORMATS[suffix]


def term_to_identifier(term: object) -> str:
    """Convert an RDF term into a stable string identifier."""

    if isinstance(term, URIRef):
        return str(term)
    if isinstance(term, BNode):
        return f"_:{term}"
    if isinstance(term, Literal):
        return term.n3()
    return str(term)


def load_rdf_graph(file_path: str | Path, rdf_format: str | None = None) -> Graph:
    """Load an RDF graph from disk."""

    path = Path(file_path)
    graph = Graph()
    if rdf_format is not None:
        graph.parse(path, format=rdf_format)
        return graph

    first_format = infer_rdf_format(path)
    candidate_formats = [first_format]
    candidate_formats.extend(
        rdf_format
        for rdf_format in FALLBACK_RDF_FORMATS
        if rdf_format not in candidate_formats
    )

    parse_errors: list[str] = []
    for candidate_format in candidate_formats:
        graph = Graph()
        try:
            graph.parse(path, format=candidate_format)
            return graph
        except Exception as exc:
            parse_errors.append(f"{candidate_format}: {type(exc).__name__}: {exc}")

    raise ValueError(
        f"Could not parse RDF file '{path}' with supported formats. "
        f"Tried: {'; '.join(parse_errors)}"
    )


def extract_triples(graph: Graph) -> list[TripleRecord]:
    """Extract triples from an RDF graph into the project's internal model."""

    triples = [
        TripleRecord(
            subject=term_to_identifier(subject),
            predicate=term_to_identifier(predicate),
            object=term_to_identifier(obj),
        )
        for subject, predicate, obj in graph
    ]
    return sorted(triples, key=lambda triple: triple.as_tuple())


def load_triples(file_path: str | Path, rdf_format: str | None = None) -> list[TripleRecord]:
    """Load a graph file and return deterministic triple records."""

    graph = load_rdf_graph(file_path=file_path, rdf_format=rdf_format)
    return extract_triples(graph)
