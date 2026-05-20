from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from src.amplitude_encoding import (
    build_amplitude_encoding_artifacts,
    prepare_amplitude_state_from_normalized_vector,
)
from src.basis_encoding import (
    build_basis_state_circuit,
    build_uniform_encoded_state_circuit_from_encoded_states,
    encode_triples_as_basis_states,
)
from src.id_mapper import build_encoding_context
from src.kg_parser import load_triples
from src.phase_encoding import (
    build_phase_interference_from_marked_circuit,
    build_phase_marked_circuit,
    predicate_phase_marker,
    subject_phase_marker,
    zero_phase_marker,
)
from src.visualization import (
    ensure_results_directories,
    format_context_summary,
    plot_bar_chart,
    save_json,
)


DEFAULT_DATASET = REPO_ROOT / "data" / "running_example.ttl"
DEFAULT_PHASE_DEMO_PREDICATE = "http://example.org/teaches"


def current_utc_timestamp() -> str:
    """Return a compact UTC timestamp for run metadata."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def duration_record(seconds: float) -> dict[str, float]:
    """Store durations in both seconds and milliseconds."""

    return {
        "seconds": round(seconds, 6),
        "milliseconds": round(seconds * 1000, 3),
    }


def summarize_duration_samples(samples: list[float]) -> dict[str, object]:
    """Build total/average/min/max timing statistics for repeated stages."""

    if not samples:
        zero_duration = duration_record(0.0)
        return {
            "count": 0,
            "total": zero_duration,
            "average": zero_duration,
            "minimum": zero_duration,
            "maximum": zero_duration,
        }

    return {
        "count": len(samples),
        "total": duration_record(sum(samples)),
        "average": duration_record(sum(samples) / len(samples)),
        "minimum": duration_record(min(samples)),
        "maximum": duration_record(max(samples)),
    }


def timed_call(function, *args, **kwargs):
    """Execute a callable and measure how long it takes."""

    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start_time
    return result, elapsed


def parse_weights(raw_weights: str | None) -> list[float] | None:
    """Parse a comma-separated weight vector."""

    if raw_weights is None:
        return None
    return [float(item.strip()) for item in raw_weights.split(",") if item.strip()]


def add_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a measured copy of a circuit."""

    measured_circuit = circuit.copy()
    measured_circuit.measure_all()
    return measured_circuit


def simulate_counts(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    """Run a measured circuit on the local Aer simulator."""

    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    return dict(sorted(counts.items()))


def simulate_probabilities(circuit: QuantumCircuit) -> dict[str, float]:
    """Compute exact state probabilities from the final statevector."""

    statevector = Statevector.from_instruction(circuit)
    probabilities = statevector.probabilities_dict()
    return {
        state: float(probability)
        for state, probability in sorted(probabilities.items())
        if probability > 0
    }


def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """Normalize measurement counts into probabilities."""

    total_shots = sum(counts.values())
    if total_shots == 0:
        return {}
    return {
        state: count / total_shots
        for state, count in sorted(counts.items())
    }


def default_output_prefix(encoding: str, input_path: str | Path) -> str:
    """Build a default filename stem for logs and figures."""

    input_stem = Path(input_path).stem
    return f"{input_stem}_{encoding}"


def select_phase_marker(args: argparse.Namespace):
    """Choose the phase-marking rule for the current run."""

    if args.mark_predicate:
        return predicate_phase_marker(
            predicate_uri=args.mark_predicate,
            phase_value=args.phase_angle,
        )

    if args.mark_subject:
        return subject_phase_marker(
            subject_uri=args.mark_subject,
            phase_value=args.phase_angle,
        )

    if Path(args.input).resolve() == DEFAULT_DATASET.resolve():
        return predicate_phase_marker(
            predicate_uri=DEFAULT_PHASE_DEMO_PREDICATE,
            phase_value=args.phase_angle,
        )

    return zero_phase_marker


def run_basis_pipeline(
    context,
    args: argparse.Namespace,
    output_prefix: str,
    results_dirs: dict[str, Path],
) -> dict[str, object]:
    """Run the basis-encoding pipeline."""

    pipeline_start = time.perf_counter()
    encoded_states, metadata_elapsed = timed_call(
        encode_triples_as_basis_states,
        context=context,
    )

    triple_runs = []
    triple_state_prep_samples: list[float] = []
    triple_ideal_samples: list[float] = []
    triple_measure_prep_samples: list[float] = []
    triple_measure_samples: list[float] = []
    triple_total_samples: list[float] = []

    for encoded_state in encoded_states:
        triple_start = time.perf_counter()
        circuit, state_prep_elapsed = timed_call(
            build_basis_state_circuit,
            bitstring=encoded_state["bitstring"],
            name=f"BasisTriple{encoded_state['triple_index']}",
        )
        probabilities, ideal_elapsed = timed_call(simulate_probabilities, circuit)
        measured_circuit, measurement_prep_elapsed = timed_call(add_measurements, circuit)
        counts, measurement_elapsed = timed_call(
            simulate_counts,
            measured_circuit,
            shots=args.shots,
        )
        triple_total_elapsed = time.perf_counter() - triple_start

        triple_state_prep_samples.append(state_prep_elapsed)
        triple_ideal_samples.append(ideal_elapsed)
        triple_measure_prep_samples.append(measurement_prep_elapsed)
        triple_measure_samples.append(measurement_elapsed)
        triple_total_samples.append(triple_total_elapsed)

        triple_runs.append(
            {
                **encoded_state,
                "ideal_probabilities": probabilities,
                "counts": counts,
                "timings": {
                    "state_preparation_circuit_build": duration_record(
                        state_prep_elapsed
                    ),
                    "ideal_statevector_simulation": duration_record(ideal_elapsed),
                    "measurement_circuit_preparation": duration_record(
                        measurement_prep_elapsed
                    ),
                    "measurement_simulation": duration_record(measurement_elapsed),
                    "triple_total": duration_record(triple_total_elapsed),
                },
            }
        )

    uniform_start = time.perf_counter()
    uniform_circuit, uniform_state_prep_elapsed = timed_call(
        build_uniform_encoded_state_circuit_from_encoded_states,
        encoded_states=encoded_states,
        num_qubits=context.total_basis_qubits,
    )
    uniform_probabilities, uniform_ideal_elapsed = timed_call(
        simulate_probabilities,
        uniform_circuit,
    )
    uniform_measured_circuit, uniform_measure_prep_elapsed = timed_call(
        add_measurements,
        uniform_circuit,
    )
    uniform_counts, uniform_measure_elapsed = timed_call(
        simulate_counts,
        uniform_measured_circuit,
        shots=args.shots,
    )
    uniform_total_elapsed = time.perf_counter() - uniform_start

    plot_elapsed = 0.0
    if not args.skip_plots:
        _, plot_elapsed = timed_call(
            plot_bar_chart,
            values=uniform_probabilities,
            title="Basis Encoding: Uniform Superposition Over Encoded Triples",
            output_path=results_dirs["figures"] / f"{output_prefix}_uniform_basis.png",
            xlabel="Basis state",
            ylabel="Probability",
        )

    pipeline_elapsed = time.perf_counter() - pipeline_start

    return {
        "encoding": "basis",
        "context": context.to_serializable_dict(),
        "triple_runs": triple_runs,
        "uniform_superposition": {
            "ideal_probabilities": uniform_probabilities,
            "counts": uniform_counts,
            "empirical_probabilities": counts_to_probabilities(uniform_counts),
        },
        "timings": {
            "encoding_pipeline": {
                "encoded_state_metadata_build": duration_record(metadata_elapsed),
                "per_triple_summary": {
                    "state_preparation_circuit_build": summarize_duration_samples(
                        triple_state_prep_samples
                    ),
                    "ideal_statevector_simulation": summarize_duration_samples(
                        triple_ideal_samples
                    ),
                    "measurement_circuit_preparation": summarize_duration_samples(
                        triple_measure_prep_samples
                    ),
                    "measurement_simulation": summarize_duration_samples(
                        triple_measure_samples
                    ),
                    "triple_total": summarize_duration_samples(triple_total_samples),
                },
                "uniform_superposition": {
                    "state_preparation_circuit_build": duration_record(
                        uniform_state_prep_elapsed
                    ),
                    "ideal_statevector_simulation": duration_record(
                        uniform_ideal_elapsed
                    ),
                    "measurement_circuit_preparation": duration_record(
                        uniform_measure_prep_elapsed
                    ),
                    "measurement_simulation": duration_record(uniform_measure_elapsed),
                    "total": duration_record(uniform_total_elapsed),
                },
                "plot_generation": duration_record(plot_elapsed),
                "pipeline_total": duration_record(pipeline_elapsed),
            }
        },
    }


def run_amplitude_pipeline(
    context,
    args: argparse.Namespace,
    output_prefix: str,
    results_dirs: dict[str, Path],
) -> dict[str, object]:
    """Run the amplitude-encoding pipeline."""

    pipeline_start = time.perf_counter()

    weights, weights_parse_elapsed = timed_call(parse_weights, args.weights)
    artifacts, artifacts_elapsed = timed_call(
        build_amplitude_encoding_artifacts,
        triples=context.triples,
        weights=weights,
        strategy=args.weight_strategy,
    )
    circuit, state_prep_elapsed = timed_call(
        prepare_amplitude_state_from_normalized_vector,
        normalized_vector=artifacts["normalized_vector"],
        name="AmplitudeEncoding",
    )
    ideal_probabilities, ideal_elapsed = timed_call(simulate_probabilities, circuit)
    measured_circuit, measurement_prep_elapsed = timed_call(add_measurements, circuit)
    counts, measurement_elapsed = timed_call(
        simulate_counts,
        measured_circuit,
        shots=args.shots,
    )

    plot_elapsed = 0.0
    if not args.skip_plots:
        _, plot_elapsed = timed_call(
            plot_bar_chart,
            values=ideal_probabilities,
            title="Amplitude Encoding: Basis-State Probabilities",
            output_path=results_dirs["figures"] / f"{output_prefix}_probabilities.png",
            xlabel="Basis state |i>",
            ylabel="Probability",
        )

    pipeline_elapsed = time.perf_counter() - pipeline_start

    return {
        "encoding": "amplitude",
        "context": context.to_serializable_dict(),
        "strategy": artifacts["strategy"],
        "raw_vector": artifacts["raw_vector"].tolist(),
        "padded_vector": [
            [float(value.real), float(value.imag)]
            for value in artifacts["padded_vector"]
        ],
        "normalized_vector": [
            [float(value.real), float(value.imag)]
            for value in artifacts["normalized_vector"]
        ],
        "index_labels": artifacts["index_labels"],
        "padded_basis_states": artifacts["padded_basis_states"],
        "ideal_probabilities": ideal_probabilities,
        "counts": counts,
        "empirical_probabilities": counts_to_probabilities(counts),
        "timings": {
            "encoding_pipeline": {
                "weights_parsing": duration_record(weights_parse_elapsed),
                "vector_and_metadata_build": duration_record(artifacts_elapsed),
                "state_preparation_circuit_build": duration_record(state_prep_elapsed),
                "ideal_statevector_simulation": duration_record(ideal_elapsed),
                "measurement_circuit_preparation": duration_record(
                    measurement_prep_elapsed
                ),
                "measurement_simulation": duration_record(measurement_elapsed),
                "plot_generation": duration_record(plot_elapsed),
                "pipeline_total": duration_record(pipeline_elapsed),
            }
        },
    }


def run_phase_pipeline(
    context,
    args: argparse.Namespace,
    output_prefix: str,
    results_dirs: dict[str, Path],
) -> dict[str, object]:
    """Run the phase-encoding pipeline."""

    pipeline_start = time.perf_counter()

    mark_fn, marker_selection_elapsed = timed_call(select_phase_marker, args)
    phase_marked_output, marked_state_prep_elapsed = timed_call(
        build_phase_marked_circuit,
        context=context,
        mark_fn=mark_fn,
    )
    before_circuit, phase_angles, marked_triples, initial_state = phase_marked_output
    after_circuit, interference_prep_elapsed = timed_call(
        build_phase_interference_from_marked_circuit,
        marked_circuit=before_circuit,
    )

    before_probabilities, before_ideal_elapsed = timed_call(
        simulate_probabilities,
        before_circuit,
    )
    after_probabilities, after_ideal_elapsed = timed_call(
        simulate_probabilities,
        after_circuit,
    )
    before_measured_circuit, before_measure_prep_elapsed = timed_call(
        add_measurements,
        before_circuit,
    )
    before_counts, before_measure_elapsed = timed_call(
        simulate_counts,
        before_measured_circuit,
        shots=args.shots,
    )
    after_measured_circuit, after_measure_prep_elapsed = timed_call(
        add_measurements,
        after_circuit,
    )
    after_counts, after_measure_elapsed = timed_call(
        simulate_counts,
        after_measured_circuit,
        shots=args.shots,
    )

    plot_before_elapsed = 0.0
    plot_after_elapsed = 0.0
    if not args.skip_plots:
        _, plot_before_elapsed = timed_call(
            plot_bar_chart,
            values=before_probabilities,
            title="Phase Encoding: Before Mixing",
            output_path=results_dirs["figures"] / f"{output_prefix}_before_mixing.png",
            xlabel="Basis state |i>",
            ylabel="Probability",
        )
        _, plot_after_elapsed = timed_call(
            plot_bar_chart,
            values=after_probabilities,
            title="Phase Encoding: After Hadamard Mixing",
            output_path=results_dirs["figures"] / f"{output_prefix}_after_mixing.png",
            xlabel="Basis state |i>",
            ylabel="Probability",
        )

    pipeline_elapsed = time.perf_counter() - pipeline_start

    return {
        "encoding": "phase",
        "context": context.to_serializable_dict(),
        "phase_angle": args.phase_angle,
        "marked_triples": marked_triples,
        "phase_angles": phase_angles.tolist(),
        "initial_uniform_state": [
            [float(value.real), float(value.imag)] for value in initial_state
        ],
        "before_mixing": {
            "ideal_probabilities": before_probabilities,
            "counts": before_counts,
            "empirical_probabilities": counts_to_probabilities(before_counts),
        },
        "after_mixing": {
            "ideal_probabilities": after_probabilities,
            "counts": after_counts,
            "empirical_probabilities": counts_to_probabilities(after_counts),
        },
        "timings": {
            "encoding_pipeline": {
                "mark_rule_selection": duration_record(marker_selection_elapsed),
                "phase_marked_state_preparation": duration_record(
                    marked_state_prep_elapsed
                ),
                "interference_circuit_preparation": duration_record(
                    interference_prep_elapsed
                ),
                "before_mixing": {
                    "ideal_statevector_simulation": duration_record(
                        before_ideal_elapsed
                    ),
                    "measurement_circuit_preparation": duration_record(
                        before_measure_prep_elapsed
                    ),
                    "measurement_simulation": duration_record(before_measure_elapsed),
                },
                "after_mixing": {
                    "ideal_statevector_simulation": duration_record(
                        after_ideal_elapsed
                    ),
                    "measurement_circuit_preparation": duration_record(
                        after_measure_prep_elapsed
                    ),
                    "measurement_simulation": duration_record(after_measure_elapsed),
                },
                "plot_generation": {
                    "before_mixing_plot": duration_record(plot_before_elapsed),
                    "after_mixing_plot": duration_record(plot_after_elapsed),
                    "total": duration_record(plot_before_elapsed + plot_after_elapsed),
                },
                "pipeline_total": duration_record(pipeline_elapsed),
            }
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the project CLI parser."""

    parser = argparse.ArgumentParser(
        description="Reusable Qiskit-based Knowledge Graph encoding pipeline."
    )
    parser.add_argument(
        "--encoding",
        required=True,
        choices=("basis", "amplitude", "phase"),
        help="Encoding family to run.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_DATASET),
        help="Path to the RDF input file. Turtle (.ttl) is supported by default.",
    )
    parser.add_argument(
        "--rdf-format",
        default=None,
        help="Optional rdflib parser format override.",
    )
    parser.add_argument(
        "--fixed-thesis-mapping",
        action="store_true",
        help="Use the optional running-example mapping/order instead of the generic deterministic one.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated custom amplitude weights, e.g. 2,1,3,1,1,2.",
    )
    parser.add_argument(
        "--weight-strategy",
        default="uniform",
        choices=("uniform", "linear"),
        help="Fallback strategy used when --weights is not supplied.",
    )
    parser.add_argument(
        "--mark-predicate",
        default=None,
        help="Predicate URI to mark during phase encoding.",
    )
    parser.add_argument(
        "--mark-subject",
        default=None,
        help="Subject URI to mark during phase encoding.",
    )
    parser.add_argument(
        "--phase-angle",
        type=float,
        default=math.pi,
        help="Phase shift applied by the chosen phase-marking rule.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=2048,
        help="Number of simulator shots for measured circuits.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "results"),
        help="Base output directory for logs and figures.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional custom filename prefix for logs and plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """CLI entry point."""

    program_start = time.perf_counter()
    run_started_at_utc = current_utc_timestamp()

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    triples, load_elapsed = timed_call(
        load_triples,
        file_path=args.input,
        rdf_format=args.rdf_format,
    )
    context, context_elapsed = timed_call(
        build_encoding_context,
        triples=triples,
        fixed_thesis_mapping=args.fixed_thesis_mapping,
        dataset_path=args.input,
    )
    results_dirs, results_dir_elapsed = timed_call(
        ensure_results_directories,
        args.results_dir,
    )

    output_prefix = args.output_prefix or default_output_prefix(
        encoding=args.encoding,
        input_path=args.input,
    )

    print(format_context_summary(context))

    if args.encoding == "basis":
        payload, pipeline_elapsed = timed_call(
            run_basis_pipeline,
            context=context,
            args=args,
            output_prefix=output_prefix,
            results_dirs=results_dirs,
        )
    elif args.encoding == "amplitude":
        payload, pipeline_elapsed = timed_call(
            run_amplitude_pipeline,
            context=context,
            args=args,
            output_prefix=output_prefix,
            results_dirs=results_dirs,
        )
    else:
        payload, pipeline_elapsed = timed_call(
            run_phase_pipeline,
            context=context,
            args=args,
            output_prefix=output_prefix,
            results_dirs=results_dirs,
        )

    log_path = results_dirs["logs"] / f"{output_prefix}.json"

    payload.setdefault("timings", {})
    payload["timings"]["setup"] = {
        "load_rdf_and_parse_triples": duration_record(load_elapsed),
        "build_id_mappings_and_context": duration_record(context_elapsed),
        "prepare_results_directories": duration_record(results_dir_elapsed),
    }
    payload["timings"]["output"] = {
        "json_log_path": str(log_path.resolve()),
        "json_log_write": duration_record(0.0),
    }
    payload["timings"]["program"] = {
        "run_started_at_utc": run_started_at_utc,
        "selected_encoding": args.encoding,
        "encoding_dispatch_total": duration_record(pipeline_elapsed),
        "total_runtime": duration_record(0.0),
    }

    _, log_save_elapsed = timed_call(save_json, payload, log_path)
    total_elapsed = time.perf_counter() - program_start

    payload["timings"]["output"]["json_log_write"] = duration_record(log_save_elapsed)
    payload["timings"]["program"]["run_finished_at_utc"] = current_utc_timestamp()
    payload["timings"]["program"]["total_runtime"] = duration_record(total_elapsed)
    payload["timings"]["program"]["timing_note"] = (
        "json_log_write was measured on a first write pass so it could be embedded "
        "in the final JSON log."
    )

    save_json(payload, log_path)

    print(f"Saved log: {log_path}")
    if not args.skip_plots:
        print(f"Saved figures under: {results_dirs['figures']}")

    return payload


if __name__ == "__main__":
    main()
