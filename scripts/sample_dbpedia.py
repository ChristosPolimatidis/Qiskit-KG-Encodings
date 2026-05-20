from __future__ import annotations

import argparse
from pathlib import Path

from rdflib import Graph


OUTPUT_SIZES = (100, 1000, 5000, 10000)
SUPPORTED_FORMATS = {
    ".ttl": "turtle",
    ".nt": "nt",
}


def infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(f"Unsupported input suffix '{suffix}'. Supported: {supported}.")
    return SUPPORTED_FORMATS[suffix]


def load_turtle_triples(input_path: Path) -> list[tuple[object, object, object]]:
    graph = Graph()
    graph.parse(input_path, format="turtle")
    return sorted(graph, key=lambda triple: tuple(str(term) for term in triple))


def load_nt_sample_triples(
    input_path: Path,
    max_triples: int,
) -> list[tuple[object, object, object]]:
    lines: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(raw_line)
            if len(lines) >= max_triples:
                break

    graph = Graph()
    graph.parse(data="".join(lines), format="nt")
    return sorted(graph, key=lambda triple: tuple(str(term) for term in triple))


def write_sample(
    triples: list[tuple[object, object, object]],
    output_path: Path,
) -> None:
    graph = Graph()
    for subject, predicate, obj in triples:
        graph.add((subject, predicate, obj))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(graph.serialize(format="turtle"), encoding="utf-8")


def sample_dbpedia(input_path: str | Path, output_dir: str | Path) -> list[Path]:
    path = Path(input_path)
    input_format = infer_format(path)
    max_size = max(OUTPUT_SIZES)

    if input_format == "nt":
        triples = load_nt_sample_triples(path, max_triples=max_size)
    else:
        triples = load_turtle_triples(path)

    if len(triples) < max_size:
        raise ValueError(
            f"Input contains {len(triples)} triples after parsing, but {max_size} are required."
        )

    output_base = Path(output_dir)
    generated_paths: list[Path] = []
    for size in OUTPUT_SIZES:
        output_path = output_base / f"dbpedia_{size}.ttl"
        write_sample(triples[:size], output_path)
        generated_paths.append(output_path)

    return generated_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create optional DBpedia Turtle samples for realism checks."
    )
    parser.add_argument(
        "input",
        help="Input DBpedia .ttl or .nt file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dbpedia",
        help="Directory where the configured dbpedia_<size>.ttl files are written.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    for path in sample_dbpedia(args.input, args.output_dir):
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
