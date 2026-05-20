from __future__ import annotations

import argparse
from pathlib import Path


DATASET_CONFIGS: dict[int, tuple[int, int, int]] = {
    100: (100, 50, 5),
    1000: (1000, 500, 10),
    5000: (5000, 2500, 15),
    10000: (10000, 5000, 20),
}

PREFIX_LINE = "@prefix ex: <http://example.org/> ."


def synthetic_triple_line(
    triple_index: int,
    entity_count: int,
    predicate_count: int,
) -> str:
    """Return one deterministic synthetic RDF triple in Turtle syntax."""

    subject_id = triple_index % entity_count
    predicate_id = triple_index % predicate_count
    cycle_offset = triple_index // entity_count
    object_id = (subject_id + cycle_offset + 1) % entity_count
    return f"ex:e{subject_id} ex:p{predicate_id} ex:e{object_id} ."


def generate_dataset(
    output_path: str | Path,
    triple_count: int,
    entity_count: int,
    predicate_count: int,
) -> Path:
    """Generate a deterministic Turtle dataset and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        PREFIX_LINE,
        "",
        *(
            synthetic_triple_line(
                triple_index=index,
                entity_count=entity_count,
                predicate_count=predicate_count,
            )
            for index in range(triple_count)
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_all(output_dir: str | Path = "data/scaling") -> list[Path]:
    """Generate every configured synthetic scaling dataset."""

    output_base = Path(output_dir)
    generated_paths: list[Path] = []

    for size, (triple_count, entity_count, predicate_count) in DATASET_CONFIGS.items():
        output_path = output_base / f"synthetic_{size}.ttl"
        generated_paths.append(
            generate_dataset(
                output_path=output_path,
                triple_count=triple_count,
                entity_count=entity_count,
                predicate_count=predicate_count,
            )
        )

    return generated_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic RDF datasets for scaling tests."
    )
    parser.add_argument(
        "--output-dir",
        default="data/scaling",
        help="Directory where the configured synthetic_<size>.ttl files are written.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    for path in generate_all(args.output_dir):
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
