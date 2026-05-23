from __future__ import annotations

from dataclasses import dataclass
import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.models import TripleRecord
from src.running_example import (
    SEQUENTIAL_INDEX_DIMENSION,
    get_running_example_indices,
    get_running_example_triples,
)


DEFAULT_SHOTS = 2048
DEFAULT_GROVER_ITERATIONS = 2
DEFAULT_SEED_SIMULATOR = 12345


@dataclass(frozen=True, slots=True)
class BasisLookupResult:
    """Result bundle for the Chapter 9 basis lookup validation task."""

    task_name: str
    task_description: str
    target_index: int
    target_bitstring: str
    target_triple: dict[str, str]
    decoded_index: int
    decoded_result: dict[str, str] | None
    counts: dict[str, int]
    task_time_seconds: float
    num_qubits: int
    circuit_depth: int
    gate_count: int
    transpiled_depth: int
    transpiled_gate_count: int
    success_probability: float
    shots: int
    backend: str
    grover_iterations: int
    index_mode: str
    claim_note: str


def _bitstring_for_index(index: int, num_qubits: int) -> str:
    return format(index, f"0{num_qubits}b")


def _qubit_for_display_bit(display_index: int, num_qubits: int) -> int:
    return num_qubits - 1 - display_index


def _apply_multi_controlled_z(circuit: QuantumCircuit) -> None:
    if circuit.num_qubits == 1:
        circuit.z(0)
        return

    controls = list(range(circuit.num_qubits - 1))
    target = circuit.num_qubits - 1
    circuit.h(target)
    circuit.mcx(controls, target)
    circuit.h(target)


def apply_index_phase_oracle(circuit: QuantumCircuit, target_bitstring: str) -> None:
    """Flip the phase of exactly one displayed basis-state bitstring."""

    num_qubits = circuit.num_qubits
    if len(target_bitstring) != num_qubits:
        raise ValueError("The target bitstring width must match the circuit width.")

    zero_qubits = [
        _qubit_for_display_bit(display_index, num_qubits)
        for display_index, bit in enumerate(target_bitstring)
        if bit == "0"
    ]
    for qubit in zero_qubits:
        circuit.x(qubit)
    _apply_multi_controlled_z(circuit)
    for qubit in reversed(zero_qubits):
        circuit.x(qubit)


def apply_diffuser(circuit: QuantumCircuit) -> None:
    """Apply the standard inversion-about-the-mean diffuser."""

    circuit.h(range(circuit.num_qubits))
    circuit.x(range(circuit.num_qubits))
    _apply_multi_controlled_z(circuit)
    circuit.x(range(circuit.num_qubits))
    circuit.h(range(circuit.num_qubits))


def build_basis_lookup_circuit(
    target_index: int,
    *,
    num_qubits: int = 3,
    grover_iterations: int = DEFAULT_GROVER_ITERATIONS,
) -> QuantumCircuit:
    """Build a small Grover-style exact-index lookup circuit."""

    if target_index < 0 or target_index >= 2**num_qubits:
        raise ValueError("The target index is outside the circuit index space.")
    if grover_iterations < 1:
        raise ValueError("At least one Grover-style iteration is required.")

    target_bitstring = _bitstring_for_index(target_index, num_qubits)
    circuit = QuantumCircuit(num_qubits, name="BasisLookupGroverDemo")
    circuit.h(range(num_qubits))
    for _ in range(grover_iterations):
        apply_index_phase_oracle(circuit, target_bitstring)
        apply_diffuser(circuit)
    return circuit


def _decode_index(
    index: int,
    index_to_triple: dict[int, TripleRecord],
) -> dict[str, str] | None:
    triple = index_to_triple.get(index)
    return triple.to_dict() if triple is not None else None


def _operation_count(circuit: QuantumCircuit) -> int:
    return sum(int(count) for count in circuit.count_ops().values())


def _most_likely_bitstring(counts: dict[str, int]) -> str:
    return max(counts, key=lambda bitstring: (counts[bitstring], bitstring))


def run_basis_lookup_task(
    *,
    target_triple: TripleRecord | None = None,
    target_index: int | None = None,
    shots: int = DEFAULT_SHOTS,
    grover_iterations: int = DEFAULT_GROVER_ITERATIONS,
    seed_simulator: int = DEFAULT_SEED_SIMULATOR,
) -> BasisLookupResult:
    """Run the Chapter 9 basis/index lookup validation task.

    This is an exact lookup demonstration on the small running example. It does
    not claim quantum advantage.
    """

    task_start = time.perf_counter()
    triples = get_running_example_triples()
    triple_to_index = get_running_example_indices(mode="sequential")
    index_to_triple = {
        index: triple
        for triple, index in triple_to_index.items()
    }

    if target_triple is None and target_index is None:
        target_triple = triples[-1]
    if target_triple is not None:
        if target_triple not in triple_to_index:
            raise ValueError("The target triple is not in the running example.")
        selected_target_index = triple_to_index[target_triple]
        selected_target_triple = target_triple
    else:
        if target_index is None or target_index not in index_to_triple:
            raise ValueError("The target index must identify a running-example triple.")
        selected_target_index = target_index
        selected_target_triple = index_to_triple[target_index]

    num_qubits = 3
    if 2**num_qubits != SEQUENTIAL_INDEX_DIMENSION:
        raise ValueError("The sequential running-example dimension must be 8.")

    circuit = build_basis_lookup_circuit(
        target_index=selected_target_index,
        num_qubits=num_qubits,
        grover_iterations=grover_iterations,
    )
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()

    simulator = AerSimulator(seed_simulator=seed_simulator)
    transpiled_circuit = transpile(measured_circuit, simulator, optimization_level=0)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = dict(sorted(result.get_counts().items()))

    most_likely_bitstring = _most_likely_bitstring(counts)
    decoded_index = int(most_likely_bitstring, 2)
    decoded_result = _decode_index(decoded_index, index_to_triple)
    success_probability = counts.get(
        _bitstring_for_index(selected_target_index, num_qubits),
        0,
    ) / shots
    task_time_seconds = time.perf_counter() - task_start

    return BasisLookupResult(
        task_name="basis_exact_triple_lookup",
        task_description=(
            "Small Chapter 9 sequential-only validation task: Grover-style "
            "oracle/diffuser demonstration over the 8-state indexed basis "
            "space for the six running-example triples."
        ),
        target_index=selected_target_index,
        target_bitstring=_bitstring_for_index(selected_target_index, num_qubits),
        target_triple=selected_target_triple.to_dict(),
        decoded_index=decoded_index,
        decoded_result=decoded_result,
        counts=counts,
        task_time_seconds=task_time_seconds,
        num_qubits=num_qubits,
        circuit_depth=circuit.depth(),
        gate_count=_operation_count(circuit),
        transpiled_depth=transpiled_circuit.depth(),
        transpiled_gate_count=_operation_count(transpiled_circuit),
        success_probability=success_probability,
        shots=shots,
        backend="AerSimulator",
        grover_iterations=grover_iterations,
        index_mode="sequential",
        claim_note=(
            "This is a sequential-only validation task for basis/index "
            "encoding. It stays in the compact 3-qubit sequential index space "
            "instead of the paper 8-qubit sparse index space because the goal "
            "is exact lookup validation, not a quantum-advantage claim."
        ),
    )
