# Qiskit KG Encodings

This repository implements a small, reusable Python framework for encoding RDF Knowledge Graphs into quantum states with Qiskit.

A short project manual is also available at [docs/manual.pdf](/home/mrdoumanis/Desktop/GitHub/Quskit-KG-Ecodings/docs/manual.pdf).

The project focuses on three encoding families:

- basis encoding
- amplitude encoding
- phase encoding

The included Turtle graph in [data/running_example.ttl](/home/mrdoumanis/Desktop/GitHub/Quskit-KG-Ecodings/data/running_example.ttl) is only a default demo dataset for the thesis running example. The code is intentionally written so the same pipeline can be reused later with a different RDF graph file.

## Features

- loads RDF graphs from Turtle files with `rdflib`
- extracts triples into a clean internal representation
- builds deterministic subject/object and predicate ID mappings
- computes bit widths automatically
- prepares reusable Qiskit circuits for basis, amplitude, and phase encodings
- runs locally with `qiskit-aer`
- saves figures and JSON logs under `results/`

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── running_example.ttl
├── experiments/
│   ├── exp_amplitude_running_example.py
│   ├── exp_basis_running_example.py
│   └── exp_phase_running_example.py
├── results/
│   ├── figures/
│   └── logs/
└── src/
    ├── amplitude_encoding.py
    ├── basis_encoding.py
    ├── id_mapper.py
    ├── kg_parser.py
    ├── main.py
    ├── models.py
    ├── phase_encoding.py
    └── visualization.py
```

## Installation

Create or activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Default Running Example

The repository ships with a small RDF/Turtle example:

- `ex:Aristotle rdf:type ex:Person`
- `ex:Person rdfs:subClassOf ex:Mortal`
- `ex:Aristotle ex:livesAt ex:Athens`
- `ex:Athens rdf:type ex:City`
- `ex:City rdfs:subClassOf ex:Place`
- `ex:Aristotle ex:teaches ex:Philosophy`

This file lives at [data/running_example.ttl](/home/mrdoumanis/Desktop/GitHub/Quskit-KG-Ecodings/data/running_example.ttl).

## CLI Usage

Run the reusable pipeline with:

```bash
python -m src.main --encoding basis --input data/running_example.ttl
python -m src.main --encoding amplitude --input data/running_example.ttl
python -m src.main --encoding phase --input data/running_example.ttl
```

Useful options:

- `--fixed-thesis-mapping`
  Uses an optional fixed mapping/ordering aligned with the included running example.
- `--weights 2,1,3,1,1,2`
  Supplies custom weights for amplitude encoding.
- `--mark-predicate http://example.org/teaches`
  Marks triples whose predicate matches that URI during phase encoding.
- `--mark-subject http://example.org/Aristotle`
  Marks triples whose subject matches that URI during phase encoding.
- `--phase-angle 3.141592653589793`
  Sets the phase shift applied by the selected marking rule.

Example:

```bash
python -m src.main \
  --encoding amplitude \
  --input data/running_example.ttl \
  --weights 2,1,3,1,1,2
```

## Running the Included Experiments

Thin wrappers around the reusable code are provided in `experiments/`:

```bash
python experiments/exp_basis_running_example.py
python experiments/exp_amplitude_running_example.py
python experiments/exp_phase_running_example.py
```

These scripts simply call the generic CLI code with sensible defaults for the thesis example.

## Using Another RDF Graph Later

The implementation is not hardcoded to the included demo graph. To use another Turtle file:

```bash
python -m src.main --encoding basis --input path/to/another_graph.ttl
```

The pipeline will:

1. load the RDF graph with `rdflib`
2. parse triples into `TripleRecord` objects
3. build subject/object and predicate ID mappings dynamically
4. compute the required bit widths automatically
5. build the chosen quantum representation
6. simulate it locally with Qiskit Aer
7. save figures and JSON logs

## Mapping Conventions

- subject and object terms share the same mapping space
- predicates use a separate mapping space
- default mappings are deterministic and lexicographically ordered for reproducibility
- the optional `--fixed-thesis-mapping` mode applies a preferred ordering for the included running example only

## Encoding Notes

### Basis Encoding

Each triple `(s, p, o)` is converted into:

```text
subject_bits || predicate_bits || object_bits
```

The displayed bitstring is big-endian for readability. The circuit builder maps qubits so that Qiskit's default measured bitstring matches that displayed string.

### Amplitude Encoding

Amplitude encoding works over triple indices:

- `|i>` corresponds to the `i`-th triple in the current context
- the weight vector is padded to the next power of two
- the padded vector is normalized automatically

If no custom vector is provided, the default strategy is uniform weights.

### Phase Encoding

Phase encoding also works over triple indices:

- the initial state is a uniform superposition over the available triple indices
- a user-supplied marking function decides which triples receive a phase shift
- a simple Hadamard mixing step is then applied to create interference

The current implementation uses a dense diagonal unitary for the phase oracle, which is fine for small teaching/thesis examples and local simulation, but not intended for large-scale production graphs.

## Outputs

The CLI saves:

- plots under [results/figures](/home/mrdoumanis/Desktop/GitHub/Quskit-KG-Ecodings/results/figures)
- JSON logs under [results/logs](/home/mrdoumanis/Desktop/GitHub/Quskit-KG-Ecodings/results/logs)

The JSON logs now also include detailed timing data for:

- RDF loading and triple parsing
- ID-mapping/context construction
- encoding-specific state preparation steps
- statevector and shot-based simulation stages
- plot generation
- JSON log writing
- total runtime of the whole program

## Dependencies

- `qiskit`
- `qiskit-aer`
- `rdflib`
- `numpy`
- `matplotlib`
