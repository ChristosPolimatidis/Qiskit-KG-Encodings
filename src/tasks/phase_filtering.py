from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.models import TripleRecord
from src.running_example import (
    EX,
    PAPER_INDEX_DIMENSION,
    SEQUENTIAL_INDEX_DIMENSION,
    get_running_example_indices,
    get_running_example_triples,
)
from src.tasks.basis_lookup import (
    apply_diffuser,
    apply_index_phase_oracle,
)


DEFAULT_MARK_PREDICATE = f"{EX}teaches"
DEFAULT_SHOTS = 2048
DEFAULT_SEED_SIMULATOR = 13579
TOP_RESULT_LIMIT = 5

RuleFunction = Callable[[TripleRecord], bool]


@dataclass(frozen=True, slots=True)
class PhaseFilteringResult:
    """Result bundle for the Chapter 9 phase filtering validation task."""

    task_name: str
    selected_predicate: str | None
    task_time_seconds: float
    num_qubits: int
    circuit_depth: int
    gate_count: int
    transpiled_depth: int
    transpiled_gate_count: int
    marked_indices: list[int]
    measurement_counts: dict[str, int]
    decoded_top_results: list[dict[str, object]]
    marked_probability_before: float
    marked_probability_after: float
    shots: int
    backend: str
    index_mode: str
    interference_step: str
    amplification_report: str
    claim_note: str


def _bitstring_for_index(index: int, num_qubits: int) -> str:
    return format(index, f"0{num_qubits}b")


def _operation_count(circuit: QuantumCircuit) -> int:
    return sum(int(count) for count in circuit.count_ops().values())


def _dimension_for_index_mode(index_mode: str) -> int:
    if index_mode == "sequential":
        return SEQUENTIAL_INDEX_DIMENSION
    if index_mode == "paper":
        return PAPER_INDEX_DIMENSION
    raise ValueError("Unsupported index mode. Use 'sequential' or 'paper'.")


def _index_to_triple_map(index_mode: str = "sequential") -> dict[int, TripleRecord]:
    return {
        index: triple
        for triple, index in get_running_example_indices(mode=index_mode).items()
    }


def select_marked_indices(
    *,
    predicate_uri: str | None = DEFAULT_MARK_PREDICATE,
    rule_fn: RuleFunction | None = None,
    index_mode: str = "sequential",
) -> list[int]:
    """Select running-example indices marked by predicate or custom rule."""

    if predicate_uri is None and rule_fn is None:
        raise ValueError("A predicate URI or rule function is required.")

    triple_to_index = get_running_example_indices(mode=index_mode)
    marked_indices: list[int] = []
    for triple in get_running_example_triples():
        predicate_match = predicate_uri is not None and triple.predicate == predicate_uri
        rule_match = rule_fn(triple) if rule_fn is not None else False
        if predicate_match or rule_match:
            marked_indices.append(triple_to_index[triple])
    return sorted(marked_indices)


def apply_marking_oracle(
    circuit: QuantumCircuit,
    marked_indices: list[int],
) -> None:
    """Apply a sign flip to every marked displayed index."""

    num_qubits = circuit.num_qubits
    for index in marked_indices:
        apply_index_phase_oracle(
            circuit,
            _bitstring_for_index(index, num_qubits),
        )


def build_phase_filtering_circuit(
    marked_indices: list[int],
    *,
    num_qubits: int = 3,
    apply_interference: bool = True,
) -> QuantumCircuit:
    """Build the phase oracle plus optional diffuser-style interference circuit."""

    if not marked_indices:
        raise ValueError("At least one marked index is required.")
    invalid_indices = [
        index for index in marked_indices if index < 0 or index >= 2**num_qubits
    ]
    if invalid_indices:
        raise ValueError(f"Marked indices outside the index space: {invalid_indices}")

    circuit = QuantumCircuit(num_qubits, name="PhaseFilteringDemo")
    circuit.h(range(num_qubits))
    apply_marking_oracle(circuit, marked_indices)
    if apply_interference:
        apply_diffuser(circuit)
    return circuit


def decode_top_results(
    measurement_counts: dict[str, int],
    *,
    marked_indices: list[int],
    shots: int,
    limit: int = TOP_RESULT_LIMIT,
    index_mode: str = "sequential",
) -> list[dict[str, object]]:
    """Decode high-probability measured indices back to RDF triples."""

    index_to_triple = _index_to_triple_map(index_mode=index_mode)
    sorted_counts = sorted(
        measurement_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    marked_index_set = set(marked_indices)
    decoded: list[dict[str, object]] = []
    for bitstring, count in sorted_counts[:limit]:
        index = int(bitstring, 2)
        triple = index_to_triple.get(index)
        decoded.append(
            {
                "bitstring": bitstring,
                "index": index,
                "count": count,
                "probability": count / shots,
                "marked": index in marked_index_set,
                "triple": triple.to_dict() if triple is not None else None,
            }
        )
    return decoded


def _marked_probability_from_counts(
    counts: dict[str, int],
    marked_indices: list[int],
    shots: int,
    num_qubits: int,
) -> float:
    marked_bitstrings = {
        _bitstring_for_index(index, num_qubits)
        for index in marked_indices
    }
    marked_counts = sum(
        count for bitstring, count in counts.items() if bitstring in marked_bitstrings
    )
    return marked_counts / shots


def _amplification_report(before: float, after: float) -> str:
    if after > before + 0.10:
        return "marked triples amplified after diffuser-style interference"
    if after < before - 0.10:
        return "marked triples suppressed after diffuser-style interference"
    return "marked triples remain distinguishable but are not strongly amplified"


def run_phase_filtering_task(
    *,
    predicate_uri: str | None = DEFAULT_MARK_PREDICATE,
    rule_fn: RuleFunction | None = None,
    index_mode: str = "sequential",
    shots: int = DEFAULT_SHOTS,
    seed_simulator: int = DEFAULT_SEED_SIMULATOR,
) -> PhaseFilteringResult:
    """Run predicate/rule phase marking followed by diffuser interference."""

    task_start = time.perf_counter()
    dimension = _dimension_for_index_mode(index_mode)
    num_qubits = (dimension.bit_length() - 1)
    if 2**num_qubits != dimension:
        raise ValueError("The selected running-example dimension must be a power of two.")

    marked_indices = select_marked_indices(
        predicate_uri=predicate_uri,
        rule_fn=rule_fn,
        index_mode=index_mode,
    )
    circuit = build_phase_filtering_circuit(
        marked_indices=marked_indices,
        num_qubits=num_qubits,
        apply_interference=True,
    )
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()

    simulator = AerSimulator(seed_simulator=seed_simulator)
    transpiled_circuit = transpile(measured_circuit, simulator, optimization_level=0)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = dict(sorted(result.get_counts().items()))

    marked_probability_before = len(marked_indices) / dimension
    marked_probability_after = _marked_probability_from_counts(
        counts=counts,
        marked_indices=marked_indices,
        shots=shots,
        num_qubits=num_qubits,
    )
    task_time_seconds = time.perf_counter() - task_start

    return PhaseFilteringResult(
        task_name="phase_predicate_filtering",
        selected_predicate=predicate_uri,
        task_time_seconds=task_time_seconds,
        num_qubits=num_qubits,
        circuit_depth=circuit.depth(),
        gate_count=_operation_count(circuit),
        transpiled_depth=transpiled_circuit.depth(),
        transpiled_gate_count=_operation_count(transpiled_circuit),
        marked_indices=marked_indices,
        measurement_counts=counts,
        decoded_top_results=decode_top_results(
            measurement_counts=counts,
            marked_indices=marked_indices,
            shots=shots,
            index_mode=index_mode,
        ),
        marked_probability_before=marked_probability_before,
        marked_probability_after=marked_probability_after,
        shots=shots,
        backend="AerSimulator",
        index_mode=index_mode,
        interference_step="diffuser-style inversion about the mean",
        amplification_report=_amplification_report(
            marked_probability_before,
            marked_probability_after,
        ),
        claim_note=(
            "This is a small predicate/rule phase-filtering validation task, "
            "not a quantum-advantage claim."
        ),
    )
