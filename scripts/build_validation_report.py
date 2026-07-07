from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ARTIFACT_FIELDS = [
    "relative_path",
    "kind",
    "size_bytes",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "command-log"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "figure"
    if suffix in {".md", ".txt"}:
        return "report"
    if suffix in {".tex"}:
        return "latex-table"
    if suffix == ".xml":
        return "test-report"
    return "artifact"


def build_artifact_index(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": str(path.relative_to(run_dir)),
                "kind": artifact_kind(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_threshold_rows(run_dir: Path) -> list[dict[str, str]]:
    paths = list(run_dir.rglob("chapter9_validation_thresholds.csv"))
    if not paths:
        return []
    with paths[0].open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_command_rows(run_dir: Path) -> list[dict[str, Any]]:
    command_log = run_dir / "command_log.jsonl"
    if not command_log.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in command_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


def validation_summary_payload(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json") or {}
    environment = read_json(run_dir / "environment.json")
    if environment is None:
        environment_paths = list(run_dir.rglob("environment.json"))
        environment = read_json(environment_paths[0]) if environment_paths else {}
    artifact_rows = build_artifact_index(run_dir)
    threshold_rows = read_threshold_rows(run_dir)
    command_rows = read_command_rows(run_dir)
    figure_count = sum(1 for row in artifact_rows if row["kind"] == "figure")
    table_count = sum(
        1
        for row in artifact_rows
        if row["kind"] in {"table", "latex-table"}
    )
    pass_count = sum(1 for row in threshold_rows if row.get("pass_fail") == "pass")
    fail_count = sum(1 for row in threshold_rows if row.get("pass_fail") == "fail")
    warnings = []
    if not threshold_rows:
        warnings.append("No chapter9_validation_thresholds.csv file was found.")
    if not command_rows:
        warnings.append("No command_log.jsonl file was found.")
    return {
        "run_dir": str(run_dir),
        "run_profile": manifest.get("profile") or manifest.get("Profile"),
        "manifest_status": manifest.get("status"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "commands": command_rows,
        "command_count": len(command_rows),
        "environment": environment,
        "artifact_count": len(artifact_rows),
        "table_count": table_count,
        "figure_count": figure_count,
        "threshold_rows": threshold_rows,
        "threshold_pass_count": pass_count,
        "threshold_fail_count": fail_count,
        "warnings": warnings,
        "claim_scope": (
            "Implementation-level simulator and classical sanity evidence only; "
            "not a quantum-advantage claim."
        ),
    }


def write_markdown_summary(payload: dict[str, Any], artifact_index_path: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Validation Summary",
        "",
        payload["claim_scope"],
        "",
        "## Run",
        "",
        f"- Run directory: `{payload['run_dir']}`",
        f"- Profile: `{payload.get('run_profile')}`",
        f"- Manifest status: `{payload.get('manifest_status')}`",
        f"- Started: `{payload.get('started_at')}`",
        f"- Finished: `{payload.get('finished_at')}`",
        f"- Commands recorded: `{payload.get('command_count')}`",
        "",
        "## Artifacts",
        "",
        f"- Artifact index: `{artifact_index_path}`",
        f"- Total artifacts: `{payload.get('artifact_count')}`",
        f"- Tables: `{payload.get('table_count')}`",
        f"- Figures: `{payload.get('figure_count')}`",
        "",
        "## Thresholds",
        "",
        f"- Pass: `{payload.get('threshold_pass_count')}`",
        f"- Fail: `{payload.get('threshold_fail_count')}`",
        "",
    ]
    threshold_rows = payload.get("threshold_rows") or []
    if threshold_rows:
        lines.extend(
            [
                "| task | repetition | estimated_value | absolute_error | pass_threshold | pass_fail |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in threshold_rows:
            lines.append(
                "| "
                + " | ".join(
                    str(row.get(field, ""))
                    for field in (
                        "task",
                        "repetition",
                        "estimated_value",
                        "absolute_error",
                        "pass_threshold",
                        "pass_fail",
                    )
                )
                + " |"
            )
        lines.append("")
    warnings = payload.get("warnings") or []
    lines.extend(["## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a validation summary and artifact index for one run directory."
    )
    parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifact_index_path = reports_dir / "artifact_index.csv"
    summary_md_path = reports_dir / "validation_summary.md"
    summary_json_path = reports_dir / "validation_summary.json"

    artifact_rows = build_artifact_index(run_dir)
    write_csv(artifact_rows, artifact_index_path, ARTIFACT_FIELDS)
    payload = validation_summary_payload(run_dir)
    payload["artifact_index"] = str(artifact_index_path)
    summary_json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_markdown_summary(payload, artifact_index_path, summary_md_path)

    print("Validation report complete")
    print(f"  Summary MD: {summary_md_path}")
    print(f"  Summary JSON: {summary_json_path}")
    print(f"  Artifact index: {artifact_index_path}")
    return payload


if __name__ == "__main__":
    main()
