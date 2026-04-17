from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.models import KGEncodingContext


def ensure_results_directories(base_directory: str | Path = "results") -> dict[str, Path]:
    """Create and return the figures/logs result directories."""

    base_path = Path(base_directory)
    figures_path = base_path / "figures"
    logs_path = base_path / "logs"

    figures_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_path,
        "figures": figures_path,
        "logs": logs_path,
    }


def save_json(data: dict[str, object], output_path: str | Path) -> None:
    """Persist a JSON log file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def plot_bar_chart(
    values: dict[str, float],
    title: str,
    output_path: str | Path,
    xlabel: str = "State",
    ylabel: str = "Value",
) -> None:
    """Plot a simple bar chart for counts or probabilities."""

    if not values:
        return

    labels = list(values.keys())
    data = list(values.values())

    figure_width = max(8, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(figure_width, 4.8))
    ax.bar(labels, data, color="#35618f")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def format_context_summary(context: KGEncodingContext) -> str:
    """Return a readable text summary for CLI output."""

    entity_lines = [
        f"    {identifier}: {entity}"
        for identifier, entity in sorted(context.id_to_entity.items())
    ]
    predicate_lines = [
        f"    {identifier}: {predicate}"
        for identifier, predicate in sorted(context.id_to_predicate.items())
    ]

    lines = [
        f"Dataset: {context.dataset_path or 'N/A'}",
        f"Triples: {context.triple_count}",
        f"Entities/objects: {context.entity_count} ({context.entity_bit_width} bits)",
        f"Predicates: {context.predicate_count} ({context.predicate_bit_width} bits)",
        f"Fixed thesis mapping: {context.fixed_thesis_mapping}",
        "Entity/object ID mapping:",
        *entity_lines,
        "Predicate ID mapping:",
        *predicate_lines,
    ]
    return "\n".join(lines)
