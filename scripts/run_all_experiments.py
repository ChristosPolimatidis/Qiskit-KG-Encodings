from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_scaling_experiments import (
    DEFAULT_REAL_KG_DIR,
    ENCODINGS,
    PHASE_MARKER_MODES,
    build_runtime_args,
    load_raw_results,
    real_dataset_specs,
    synthetic_dataset_specs,
    update_experiment_outputs,
    write_experiment_outputs,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run synthetic, real KG, and combined encoding experiments."
    )
    parser.add_argument(
        "--synthetic-sizes",
        nargs="+",
        type=int,
        default=[100, 1000, 5000],
        help="Synthetic dataset sizes to generate/run.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/scaling",
        help="Directory containing synthetic_<size>.ttl files.",
    )
    parser.add_argument(
        "--real-dir",
        default=str(DEFAULT_REAL_KG_DIR),
        help="Directory containing local real KG files.",
    )
    parser.add_argument(
        "--real-files",
        nargs="+",
        default=None,
        help="Optional subset of local real KG files to run.",
    )
    parser.add_argument(
        "--results-root",
        default="results/scaling",
        help="Root directory for synthetic, real, and combined outputs.",
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        choices=ENCODINGS,
        default=list(ENCODINGS),
        help="Encoding families to run.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions per dataset/encoding configuration.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of measurement shots.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600,
        help="Per-run timeout. Timed-out runs are recorded and the batch continues.",
    )
    parser.add_argument(
        "--max-real-triples",
        type=int,
        default=None,
        help="Deterministically truncate each real KG file to this many triples.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new rows to existing synthetic/real raw CSVs and rebuild combined outputs.",
    )
    parser.add_argument(
        "--regenerate-plots-only",
        action="store_true",
        help="Do not run experiments; rebuild summaries and plots from existing raw CSVs.",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Do not run or regenerate the synthetic result group. Existing synthetic raw rows are still used for combined plots when present.",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Do not run or regenerate the real KG result group. Existing real raw rows are still used for combined plots when present.",
    )
    parser.add_argument(
        "--rdf-format",
        default=None,
        help="Optional rdflib parser format override. Leave unset for mixed real KG files.",
    )
    parser.add_argument(
        "--weight-strategy",
        choices=("uniform", "linear"),
        default="uniform",
        help="Amplitude weight strategy used by the experiments.",
    )
    parser.add_argument(
        "--phase-marker-mode",
        choices=PHASE_MARKER_MODES,
        default=None,
        help=(
            "Phase predicate selection mode. If omitted, synthetic datasets use "
            "synthetic-default and real datasets use most-common-predicate."
        ),
    )
    parser.add_argument(
        "--phase-mark-predicate",
        default=None,
        help=(
            "Custom predicate URI marked in phase encoding. Supplying this value "
            "uses custom phase selection unless --phase-marker-mode overrides it."
        ),
    )
    parser.add_argument(
        "--phase-angle",
        type=float,
        default=math.pi,
        help="Phase angle used for marked triples.",
    )
    parser.add_argument(
        "--max-basis-simulation-qubits",
        type=int,
        default=22,
        help="Guardrail for dense basis statevector simulation.",
    )
    parser.add_argument(
        "--max-phase-diagonal-qubits",
        type=int,
        default=10,
        help="Guardrail for the dense phase-oracle matrix.",
    )
    parser.add_argument(
        "--max-metric-qubits",
        type=int,
        default=14,
        help=(
            "Guardrail for decomposed/transpiled circuit metrics. Logical metrics "
            "are still recorded above this limit."
        ),
    )
    parser.add_argument(
        "--decompose-reps",
        type=int,
        default=1,
        help="Number of Qiskit decompose() repetitions for decomposed metrics.",
    )
    parser.add_argument(
        "--compute-decomposed-metrics",
        dest="compute_decomposed_metrics",
        action="store_true",
        default=True,
        help="Compute guarded decomposed circuit metrics.",
    )
    parser.add_argument(
        "--no-compute-decomposed-metrics",
        dest="compute_decomposed_metrics",
        action="store_false",
        help="Skip decomposed circuit metrics.",
    )
    parser.add_argument(
        "--compute-transpiled-metrics",
        dest="compute_transpiled_metrics",
        action="store_true",
        default=True,
        help="Compute guarded transpiled circuit metrics.",
    )
    parser.add_argument(
        "--no-compute-transpiled-metrics",
        dest="compute_transpiled_metrics",
        action="store_false",
        help="Skip transpiled circuit metrics.",
    )
    parser.add_argument(
        "--no-generate-missing",
        action="store_true",
        help="Do not generate missing synthetic datasets automatically.",
    )
    return parser


def main() -> None:
    mp.freeze_support()
    args = build_argument_parser().parse_args()
    if args.skip_synthetic and args.skip_real:
        raise ValueError("At least one of synthetic or real experiments must be enabled.")

    runtime_args = build_runtime_args(args)
    results_root = Path(args.results_root)

    synthetic_specs = []
    real_specs = []
    if not args.regenerate_plots_only:
        synthetic_specs = synthetic_dataset_specs(
            sizes=args.synthetic_sizes,
            data_dir=Path(args.data_dir),
            generate_missing=not args.no_generate_missing,
        ) if not args.skip_synthetic else []
        real_specs = (
            real_dataset_specs(Path(args.real_dir), filenames=args.real_files)
            if not args.skip_real
            else []
        )

    synthetic_raw = results_root / "synthetic" / "scaling_raw_results.csv"
    synthetic_summary = results_root / "synthetic" / "scaling_summary.csv"
    synthetic_plots = []
    if args.skip_synthetic:
        synthetic_rows = load_raw_results(synthetic_raw)
        print("Skipped synthetic group.")
    else:
        synthetic_rows, synthetic_raw, synthetic_summary, synthetic_plots = update_experiment_outputs(
            dataset_specs=synthetic_specs,
            encodings=list(args.encodings),
            repetitions=args.repetitions,
            args=runtime_args,
            output_dir=results_root / "synthetic",
            group="synthetic",
            timeout_seconds=args.timeout_seconds,
            append=args.append,
            regenerate_plots_only=args.regenerate_plots_only,
        )

    real_raw = results_root / "real" / "scaling_raw_results.csv"
    real_summary = results_root / "real" / "scaling_summary.csv"
    real_plots = []
    if args.skip_real:
        real_rows = load_raw_results(real_raw)
        print("Skipped real KG group.")
    else:
        real_rows, real_raw, real_summary, real_plots = update_experiment_outputs(
            dataset_specs=real_specs,
            encodings=list(args.encodings),
            repetitions=args.repetitions,
            args=runtime_args,
            output_dir=results_root / "real",
            group="real",
            timeout_seconds=args.timeout_seconds,
            append=args.append,
            regenerate_plots_only=args.regenerate_plots_only,
        )

    combined_rows = [*synthetic_rows, *real_rows]
    if args.regenerate_plots_only and not combined_rows:
        combined_rows = load_raw_results(
            results_root / "combined" / "scaling_raw_results.csv"
        )
        if not combined_rows:
            raise FileNotFoundError(
                "Cannot regenerate combined plots because no synthetic, real, or combined raw CSV exists."
            )
    combined_raw, combined_summary, combined_plots = write_experiment_outputs(
        rows=combined_rows,
        output_dir=results_root / "combined",
        group="combined",
    )

    if args.skip_synthetic:
        print(f"Synthetic group skipped; loaded {len(synthetic_rows)} existing rows from {synthetic_raw}")
    else:
        print(f"Saved synthetic raw results: {synthetic_raw}")
        print(f"Saved synthetic summary: {synthetic_summary}")
        for path in synthetic_plots:
            print(f"Saved synthetic plot: {path}")

    if args.skip_real:
        print(f"Real KG group skipped; loaded {len(real_rows)} existing rows from {real_raw}")
    else:
        print(f"Saved real raw results: {real_raw}")
        print(f"Saved real summary: {real_summary}")
        for path in real_plots:
            print(f"Saved real plot: {path}")

    print(f"Saved combined raw results: {combined_raw}")
    print(f"Saved combined summary: {combined_summary}")
    for path in combined_plots:
        print(f"Saved combined plot: {path}")


if __name__ == "__main__":
    main()
