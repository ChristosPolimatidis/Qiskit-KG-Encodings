# Qiskit KG Encodings

User guide for the thesis project **Comparative Evaluation of Quantum Encodings
for Knowledge Graphs**.

This repository implements a small, reproducible Python/Qiskit workflow for
comparing three quantum encoding families for Knowledge Graph (KG) data:

- basis encoding
- amplitude encoding
- phase encoding

The goal is to measure how the encoding choice affects the practical
KG-to-quantum pipeline: RDF loading, triple parsing, ID mapping, state or circuit
preparation, local simulation, measurement, decoding, runtime, qubit
requirements, circuit depth, and gate count.

This project is **not** trying to prove quantum advantage. It is also **not** a
DBpedia database benchmark. Real KG inputs are used as a realism check, while
synthetic KGs are used for controlled scalability experiments.

## Contents

- [High-Level Workflow](#high-level-workflow)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Synthetic KG Datasets](#synthetic-kg-datasets)
- [Real KG Datasets](#real-kg-datasets)
- [Running Experiments](#running-experiments)
- [Chapter 9 Experiments](#chapter-9-experiments)
- [Incremental Experiments](#incremental-experiments)
- [Command-Line Reference](#command-line-reference)
- [Simulator Limits And Run Statuses](#simulator-limits-and-run-statuses)
- [Output Files](#output-files)
- [Plot Guide](#plot-guide)
- [Adding New Data Or Encodings](#adding-new-data-or-encodings)
- [Troubleshooting](#troubleshooting)
- [Using The Results In The Thesis](#using-the-results-in-the-thesis)
- [Reproducibility Checklist](#reproducibility-checklist)

## High-Level Workflow

The experiment pipeline follows the same conceptual steps for synthetic and real
KG datasets:

1. Load RDF / KG data from disk.
2. Parse RDF triples into internal `TripleRecord` objects.
3. Build deterministic identifier mappings:
   - one shared ID space for subjects and objects
   - one ID space for predicates
4. Apply the selected encoding:
   - basis encoding
   - amplitude encoding
   - phase encoding
5. Construct the quantum circuit or state representation.
6. Run local simulation when feasible.
7. Measure and decode counts where applicable.
8. Record runtime and circuit metrics.
9. Export raw and summary CSV files.
10. Generate comparative plots.

The reusable running-example CLI lives in [src/main.py](src/main.py). The
scaling and thesis experiment scripts live in [scripts/](scripts/).

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   |-- running_example.ttl
|   |-- scaling/
|       |-- synthetic_100.ttl
|       |-- synthetic_1000.ttl
|       |-- synthetic_5000.ttl
|       `-- synthetic_10000.ttl
|   `-- real_kgs/
|       |-- Aristotle.xml
|       |-- DecodedOntologies_V2.ttl
|       |-- exampleV3.ttl
|       `-- productsSmall.rdf
|-- docs/
|   |-- manual.md
|   |-- manual.pdf
|   `-- scaling_experiments.md
|-- experiments/
|   |-- exp_amplitude_running_example.py
|   |-- exp_basis_running_example.py
|   `-- exp_phase_running_example.py
|-- results/
|   |-- figures/
|   |-- logs/
|   `-- scaling/
|       |-- synthetic/
|       |-- real/
|       `-- combined/
|-- scripts/
|   |-- generate_scaling_datasets.py
|   |-- run_scaling_experiments.py
|   |-- run_all_experiments.py
|   `-- sample_dbpedia.py
|-- src/
|   |-- amplitude_encoding.py
|   |-- basis_encoding.py
|   |-- id_mapper.py
|   |-- kg_parser.py
|   |-- main.py
|   |-- models.py
|   |-- phase_encoding.py
|   `-- visualization.py
`-- tools/
```

### Folder Guide

| Folder | Purpose | Should users modify it? |
| --- | --- | --- |
| `src/` | Core KG parsing, ID mapping, encoding implementations, simulation helpers, and plotting/logging helpers. | Modify only when changing the implementation. |
| `scripts/` | Experiment automation: synthetic generation, scaling runs, all-in-one synthetic/real runs, optional DBpedia sampling. | Usually run these scripts; edit only to add experiment behavior. |
| `experiments/` | Thin wrappers for the small running example. | Useful for quick demos. |
| `data/` | Default demo KG, generated synthetic scaling KGs, and local real KG inputs. | Do not edit generated files manually unless you know why. |
| `data/scaling/` | Generated synthetic KG files. | Regenerate with `scripts/generate_scaling_datasets.py`. |
| `data/real_kgs/` | Local real KG files from the professor's QuantumEncoDeco resources, plus any user-added real KGs. | Put new real KG files here. |
| `results/` | Generated figures, logs, CSVs, and plots. | Usually do not edit manually. Safe to delete if you want a clean rerun. |
| `results/scaling/synthetic/` | Synthetic-only raw CSV, summary CSV, and plots. | Generated output. |
| `results/scaling/real/` | Real-KG-only raw CSV, summary CSV, and plots. | Generated output. |
| `results/scaling/combined/` | Combined synthetic + real raw CSV, summary CSV, and plots. | Generated output. |
| `docs/` | Extra documentation, especially deeper scaling experiment details. | Edit when documentation changes. |
| `tools/` | Utility tooling, such as manual PDF generation. | Optional developer utilities. |

## Installation

The commands below are written for Windows PowerShell.

### 1. Clone The Repository

```powershell
git clone <repository-url>
cd Quskit-KG-Ecodings
```

If the repository is already on your machine, open PowerShell in the repository
root.

### 2. Create And Activate A Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, see
[PowerShell Execution Policy](#powershell-execution-policy) in the
troubleshooting section.

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

The current requirements are:

- `qiskit`
- `qiskit-aer`
- `rdflib`
- `numpy`
- `matplotlib`

If Qiskit or Aer is missing in your environment, install them explicitly:

```powershell
pip install qiskit qiskit-aer
```

### 4. Test The Core CLI

```powershell
python -m src.main --encoding amplitude --input data/running_example.ttl --skip-plots
```

You should see a dataset summary printed in the terminal and a JSON log under:

```text
results/logs/
```

## Quick Start

The shortest useful path is:

```powershell
python scripts/generate_scaling_datasets.py
python scripts/run_scaling_experiments.py --sizes 100 --repetitions 1 --shots 256
```

After the command finishes, check:

```text
results/scaling/synthetic/scaling_raw_results.csv
results/scaling/synthetic/scaling_summary.csv
results/scaling/synthetic/plots/
```

The smoke test runs basis, amplitude, and phase encodings on the 100-triple
synthetic dataset once. It should generate runtime, qubit, logical circuit, and
where feasible decomposed/transpiled circuit plots under:

```text
results/scaling/synthetic/plots/
```

## Synthetic KG Datasets

Synthetic KGs are used for controlled scalability testing. They make it possible
to compare encodings under predictable conditions because the number of triples,
entities, and predicates is fixed.

The generator is:

```text
scripts/generate_scaling_datasets.py
```

Run:

```powershell
python scripts/generate_scaling_datasets.py
```

Generated files:

| File | Triples | Entities | Predicates |
| --- | ---: | ---: | ---: |
| `data/scaling/synthetic_100.ttl` | 100 | 50 | 5 |
| `data/scaling/synthetic_1000.ttl` | 1,000 | 500 | 10 |
| `data/scaling/synthetic_5000.ttl` | 5,000 | 2,500 | 15 |
| `data/scaling/synthetic_10000.ttl` | 10,000 | 5,000 | 20 |

Each file uses deterministic Turtle triples with the prefix:

```turtle
@prefix ex: <http://example.org/> .
```

Example generated triples:

```turtle
ex:e0 ex:p0 ex:e1 .
ex:e1 ex:p1 ex:e2 .
```

### Changing Or Adding Synthetic Sizes

Edit `DATASET_CONFIGS` in
[scripts/generate_scaling_datasets.py](scripts/generate_scaling_datasets.py):

```python
DATASET_CONFIGS = {
    100: (100, 50, 5),
    1000: (1000, 500, 10),
    5000: (5000, 2500, 15),
    10000: (10000, 5000, 20),
}
```

The tuple format is:

```text
dataset_size: (triple_count, entity_count, predicate_count)
```

After editing, regenerate:

```powershell
python scripts/generate_scaling_datasets.py
```

## Real KG Datasets

Real KG files should be placed in:

```text
data/real_kgs/
```

The currently expected real KG files are:

```text
data/real_kgs/exampleV3.ttl
data/real_kgs/productsSmall.rdf
data/real_kgs/DecodedOntologies_V2.ttl
data/real_kgs/Aristotle.xml
```

These files come from the professor's QuantumEncoDeco resources and are used as
a realism check. They are not the main controlled scalability experiment.

The parser currently supports:

| Extension | RDFLib format |
| --- | --- |
| `.ttl` | Turtle |
| `.rdf` | RDF/XML |
| `.xml` | RDF/XML |
| `.nt` | N-Triples |

The parser also tries fallback RDF formats when no explicit `--rdf-format` is
provided.

### Adding Your Own Real KG

Example: you have `my_graph.ttl`.

1. Copy it into `data/real_kgs/`:

   ```powershell
   Copy-Item .\my_graph.ttl .\data\real_kgs\my_graph.ttl
   ```

2. Run it explicitly with `--real-files`:

   ```powershell
   python scripts/run_all_experiments.py --skip-synthetic --real-files my_graph.ttl --repetitions 1 --shots 256 --timeout-seconds 300
   ```

3. Check:

   ```text
   results/scaling/real/scaling_raw_results.csv
   results/scaling/real/scaling_summary.csv
   results/scaling/real/plots/
   ```

If you want the file to be included by default when `--real-files` is omitted,
edit `REAL_KG_FILES` in
[scripts/run_scaling_experiments.py](scripts/run_scaling_experiments.py):

```python
REAL_KG_FILES = (
    "exampleV3.ttl",
    "productsSmall.rdf",
    "DecodedOntologies_V2.ttl",
    "Aristotle.xml",
)
```

## Running Experiments

### Synthetic-Only Experiments

Run 100 and 1,000 triples, five repetitions each:

```powershell
python scripts/run_scaling_experiments.py --sizes 100 1000 --repetitions 5
```

Run the default synthetic set, currently 100, 1,000, and 5,000:

```powershell
python scripts/run_scaling_experiments.py --repetitions 5
```

### Real-Only Experiments

Run all default real KG files, all encodings:

```powershell
python scripts/run_all_experiments.py --skip-synthetic --repetitions 5 --shots 1024 --timeout-seconds 600
```

Run one real file:

```powershell
python scripts/run_all_experiments.py --skip-synthetic --real-files exampleV3.ttl --repetitions 1 --shots 256 --timeout-seconds 300
```

### Combined Synthetic + Real Experiments

Run synthetic and real KG experiments and regenerate the combined outputs:

```powershell
python scripts/run_all_experiments.py --synthetic-sizes 100 1000 5000 --repetitions 5 --shots 1024 --timeout-seconds 600
```

This writes:

```text
results/scaling/synthetic/
results/scaling/real/
results/scaling/combined/
```

### All-In-One Recommended Command

```powershell
python scripts/run_all_experiments.py --synthetic-sizes 100 1000 5000 --repetitions 5 --shots 1024 --timeout-seconds 600
```

## Chapter 9 Experiments

The Chapter 9 experiment runner is separate from the older scalability
experiments. It focuses only on the canonical six-triple running example used in
the new thesis paper, and it writes paper-facing summary tables for the four
encoding families and the four small validation tasks.

This script does **not** run the old large synthetic/real scalability
experiments. To run those, use `scripts/run_scaling_experiments.py` or
`scripts/run_all_experiments.py` explicitly.

These Chapter 9 runs are simulator-based validation experiments using Qiskit and
Aer Simulator. They are useful for checking encoding behavior, table values, and
small task demonstrations. They are **not** quantum-advantage results.

### Sequential Index Mode

Sequential mode maps the six running-example triples to indices `0..5` and pads
index-based states to dimension `8`.

```powershell
python scripts/run_chapter9_experiments.py --index-mode sequential --shots 2048 --repetitions 5 --output-dir results/chapter9/sequential
```

### Paper Index Mode

Paper mode uses the sparse paper indices `T = {1, 17, 27, 37, 88, 124}`, so
index-based states use 8 qubits and dimension `256`.

```powershell
python scripts/run_chapter9_experiments.py --index-mode paper --shots 2048 --repetitions 5 --output-dir results/chapter9/paper
```

If you want the default output folder exactly, omit `--output-dir` or set it to
`results/chapter9`. Run one index mode at a time in that folder, because each
run rewrites the same table filenames.

### Chapter 9 Outputs

For an output directory such as `results/chapter9`, the script writes:

| Output | Purpose |
| --- | --- |
| `results/chapter9/table3_encoding_process.csv` | Paper-facing Table 3 data for encoding creation: encoding, variant, index mode, qubits, dimension, creation time, circuit metrics, and notes. |
| `results/chapter9/table3_encoding_process.tex` | Compact LaTeX version of Table 3. |
| `results/chapter9/table4_usage_tasks.csv` | Paper-facing Table 4 data for the Table 1 task-to-encoding mapping: Search/Grover lookup, Entity Matching/Swap Test, Link Prediction/Distance Estimation, Multi-hop Reasoning/Phase Kickback, and Schema Matching/QFT. |
| `results/chapter9/table4_usage_tasks.tex` | Compact LaTeX version of Table 4. |
| `results/chapter9/table6_circuit_statistics.csv` | Circuit statistics for the five Chapter 9 running-example task validations: qubits, depth, gate counts, transpiled metrics, shots, and repetitions. |
| `results/chapter9/table6_circuit_statistics.tex` | Compact LaTeX version of Table 6. |
| `results/chapter9/chapter9_raw_results.json` | Full raw run metadata, detailed rows, task payloads, and the additional combined amplitude-phase validation kept separate from Table 4. Use this as the audit trail, not as a table pasted into the paper. |
| `results/chapter9/environment.json` | Reproducibility metadata: Python, OS, CPU/RAM when available, package versions, command-line arguments, timestamp, seed, hostname, and git commit hash when available. |
| `results/chapter9/figures/` | Optional runtime and paper-facing plots, including Table 3 time/qubits, Table 4 task time, amplitude probabilities, and combined magnitude/phase. |

At the end of a run, the script prints a short summary including the machine
specification line, table paths, environment path, and a note that old
scalability experiments were not run.

## Incremental Experiments

Large datasets often require different simulator limits. The scripts support
incremental runs so that partial experiments do not overwrite previous results.

### Append New Synthetic Rows

First run the main clean experiment:

```powershell
python scripts/run_scaling_experiments.py --sizes 100 1000 --repetitions 5
```

Later add a 5,000-triple amplitude run without destroying previous rows:

```powershell
python scripts/run_scaling_experiments.py --sizes 5000 --encodings amplitude --repetitions 1 --append
```

When `--append` is used, the script:

- loads the existing raw CSV
- skips already-recorded matching configurations
- appends new rows
- recomputes the summary CSV
- regenerates plots from all rows

The append deduplication key uses:

```text
dataset_name + dataset_category + encoding + repetition + shots
+ simulator limits + metric limits + timeout + weight strategy
+ phase settings + rdf format
```

### Regenerate Plots Only

Do not run experiments. Rebuild summary and plots from the stored raw CSV:

```powershell
python scripts/run_scaling_experiments.py --regenerate-plots-only
```

For all groups:

```powershell
python scripts/run_all_experiments.py --regenerate-plots-only
```

### Use A Separate Output Directory

Use this when testing risky settings:

```powershell
python scripts/run_scaling_experiments.py --sizes 5000 --repetitions 1 --output-dir results/scaling/test_5000
```

For all-in-one experiments, use `--results-root`:

```powershell
python scripts/run_all_experiments.py --results-root results/scaling/test_all --synthetic-sizes 100 --real-files Aristotle.xml --repetitions 1
```

## Command-Line Reference

This section documents the actual command-line parameters currently implemented
in the repository.

### `scripts/generate_scaling_datasets.py`

| Parameter | Default | What It Does | Example |
| --- | --- | --- | --- |
| `--output-dir` | `data/scaling` | Directory where `synthetic_<size>.ttl` files are written. | `python scripts/generate_scaling_datasets.py --output-dir data/my_scaling` |

### `scripts/run_scaling_experiments.py`

This script runs **synthetic-only** scaling experiments.

| Parameter | Default | What It Does | When To Use It | Example |
| --- | --- | --- | --- | --- |
| `--data-dir` | `data/scaling` | Directory containing synthetic Turtle files. | Use when synthetic files are somewhere else. | `--data-dir data/my_scaling` |
| `--output-dir` / `--results-dir` | `results/scaling/synthetic` | Output folder for raw CSV, summary CSV, and plots. | Use for isolated test runs. | `--output-dir results/scaling/test_5000` |
| `--sizes` | `100 1000 5000` | Synthetic dataset sizes to run. | Use to select small or large datasets. | `--sizes 100 1000` |
| `--encodings` | `basis amplitude phase` | Encoding families to run. | Use for partial runs. | `--encodings amplitude phase` |
| `--repetitions` | `5` | Number of repetitions per size/encoding. | Use more for stable averages; use 1 for smoke tests. | `--repetitions 1` |
| `--shots` | `2048` | Measurement shots used for sampled counts. | Lower for speed, higher for smoother measurement estimates. | `--shots 1024` |
| `--timeout-seconds` | none | Optional per-run timeout. Timed-out rows are recorded. | Use for large or risky runs. | `--timeout-seconds 600` |
| `--append` | off | Append new configurations to the existing raw CSV and rebuild summary/plots. | Use for incremental experiments. | `--append` |
| `--regenerate-plots-only` | off | Do not run experiments; rebuild summary and plots from raw CSV. | Use after editing plotting code or combining rows. | `--regenerate-plots-only` |
| `--rdf-format` | none | Force an RDFLib parser format. | Rare for synthetic data. | `--rdf-format turtle` |
| `--weight-strategy` | `uniform` | Amplitude weights when explicit weights are not supplied. Choices: `uniform`, `linear`. | Use to compare amplitude weighting strategies. | `--weight-strategy linear` |
| `--phase-marker-mode` | category-dependent | Predicate selection mode for phase encoding. Choices: `synthetic-default`, `first-predicate`, `most-common-predicate`, `none`, `custom`. Synthetic runs default to `synthetic-default`; real KG runs default to `most-common-predicate`. | Use to avoid no-op real-KG phase runs or to create a no-op baseline. | `--phase-marker-mode most-common-predicate` |
| `--phase-mark-predicate` | none | Custom predicate URI marked by phase encoding. Supplying this normally makes the selection mode `custom`. | Use when the paper needs a specific predicate. | `--phase-marker-mode custom --phase-mark-predicate http://example.org/p1` |
| `--phase-angle` | pi | Phase shift applied to marked triples. | Use to test a different phase shift. | `--phase-angle 1.57079632679` |
| `--max-basis-simulation-qubits` | `22` | Guardrail for dense basis statevector simulation. | Increase only if your machine can handle it. | `--max-basis-simulation-qubits 24` |
| `--max-phase-diagonal-qubits` | `10` | Guardrail for the dense phase-oracle matrix. | Increase cautiously; memory grows quickly. | `--max-phase-diagonal-qubits 12` |
| `--max-metric-qubits` | `14` | Guardrail for decomposed/transpiled metric extraction. Logical metrics are still recorded above this limit. | Use to keep metric synthesis from dominating large runs. | `--max-metric-qubits 12` |
| `--decompose-reps` | `1` | Number of `circuit.decompose(...)` repetitions used for decomposed metrics. | Increase only for small circuits when you want deeper expansion. | `--decompose-reps 2` |
| `--compute-decomposed-metrics` | on | Compute guarded decomposed depth/count metrics. | Use default for thesis runs. | `--compute-decomposed-metrics` |
| `--no-compute-decomposed-metrics` | off | Skip decomposed metric extraction. | Use for very large or timing-sensitive runs. | `--no-compute-decomposed-metrics` |
| `--compute-transpiled-metrics` | on | Compute guarded transpiled depth/count metrics with Qiskit `transpile`. | Use default for small/medium runs. | `--compute-transpiled-metrics` |
| `--no-compute-transpiled-metrics` | off | Skip transpiled metric extraction. | Use when arbitrary `initialize` or `unitary` synthesis is too expensive. | `--no-compute-transpiled-metrics` |
| `--no-generate-missing` | off | Do not auto-generate missing synthetic files. | Use when you want missing data to be an error. | `--no-generate-missing` |

There is no backend-selection parameter in the scaling runner. It records
`Qiskit Statevector` as the backend because the scaling scripts use Qiskit's
`Statevector` object for local simulation and sampling.

### `scripts/run_all_experiments.py`

This script runs synthetic, real, and combined workflows.

| Parameter | Default | What It Does | When To Use It | Example |
| --- | --- | --- | --- | --- |
| `--synthetic-sizes` | `100 1000 5000` | Synthetic sizes to generate/run. | Use to include or exclude larger sizes. | `--synthetic-sizes 100 1000` |
| `--data-dir` | `data/scaling` | Synthetic data folder. | Use for alternate synthetic files. | `--data-dir data/my_scaling` |
| `--real-dir` | `data/real_kgs` | Folder containing real KG files. | Use if real files are elsewhere. | `--real-dir my_real_kgs` |
| `--real-files` | all files in `REAL_KG_FILES` | Optional subset of real files to run. | Use for one real KG or a custom file. | `--real-files my_graph.ttl` |
| `--results-root` | `results/scaling` | Root folder for `synthetic/`, `real/`, and `combined/`. | Use for isolated full runs. | `--results-root results/scaling/test_all` |
| `--encodings` | `basis amplitude phase` | Encoding families to run. | Use for partial runs. | `--encodings amplitude` |
| `--repetitions` | `5` | Repetitions per dataset/encoding. | Use 1 for smoke tests. | `--repetitions 1` |
| `--shots` | `1024` | Measurement shots. | Tune speed versus measurement stability. | `--shots 256` |
| `--timeout-seconds` | `600` | Per-run timeout. | Use to prevent large runs from hanging. | `--timeout-seconds 120` |
| `--append` | off | Append to existing synthetic/real CSVs and rebuild combined outputs. | Use for incremental all-in-one runs. | `--append` |
| `--regenerate-plots-only` | off | Rebuild summaries and plots from stored raw CSVs. | Use after appending or changing plots. | `--regenerate-plots-only` |
| `--skip-synthetic` | off | Skip synthetic runs/regeneration. Existing synthetic rows can still be used in combined output. | Use for real-only runs. | `--skip-synthetic` |
| `--skip-real` | off | Skip real KG runs/regeneration. Existing real rows can still be used in combined output. | Use for synthetic-only all-in-one runs. | `--skip-real` |
| `--rdf-format` | none | Force RDFLib parser format. | Usually leave unset for mixed real formats. | `--rdf-format turtle` |
| `--weight-strategy` | `uniform` | Amplitude weight strategy. Choices: `uniform`, `linear`. | Use for amplitude weighting experiments. | `--weight-strategy linear` |
| `--phase-marker-mode` | category-dependent | Predicate selection mode for phase encoding. Choices: `synthetic-default`, `first-predicate`, `most-common-predicate`, `none`, `custom`. Synthetic runs default to `synthetic-default`; real KG runs default to `most-common-predicate`. | Use to avoid no-op real-KG phase runs or to force a baseline. | `--phase-marker-mode first-predicate` |
| `--phase-mark-predicate` | none | Custom predicate URI marked by phase encoding. Supplying this normally makes the selection mode `custom`. | Use to adapt phase marking to a known KG predicate. | `--phase-marker-mode custom --phase-mark-predicate http://example.org/p1` |
| `--phase-angle` | pi | Phase shift for marked triples. | Use to test another phase angle. | `--phase-angle 3.141592653589793` |
| `--max-basis-simulation-qubits` | `22` | Dense basis simulation guardrail. | Increase cautiously for large machines. | `--max-basis-simulation-qubits 24` |
| `--max-phase-diagonal-qubits` | `10` | Dense phase-oracle guardrail. | Increase cautiously. | `--max-phase-diagonal-qubits 12` |
| `--max-metric-qubits` | `14` | Guardrail for decomposed/transpiled metric extraction. Logical metrics are still recorded above this limit. | Use to avoid costly metric synthesis. | `--max-metric-qubits 12` |
| `--decompose-reps` | `1` | Number of `circuit.decompose(...)` repetitions used for decomposed metrics. | Increase only for small circuits. | `--decompose-reps 2` |
| `--compute-decomposed-metrics` | on | Compute guarded decomposed depth/count metrics. | Use default for thesis runs. | `--compute-decomposed-metrics` |
| `--no-compute-decomposed-metrics` | off | Skip decomposed metric extraction. | Use for large timing-sensitive runs. | `--no-compute-decomposed-metrics` |
| `--compute-transpiled-metrics` | on | Compute guarded transpiled depth/count metrics with Qiskit `transpile`. | Use default for small/medium runs. | `--compute-transpiled-metrics` |
| `--no-compute-transpiled-metrics` | off | Skip transpiled metric extraction. | Use when transpilation becomes the bottleneck. | `--no-compute-transpiled-metrics` |
| `--no-generate-missing` | off | Do not generate missing synthetic files. | Use when data should already exist. | `--no-generate-missing` |

### `scripts/sample_dbpedia.py`

Optional realism-check sampler. It is not required for the main experiment.

| Parameter | Default | What It Does | Example |
| --- | --- | --- | --- |
| `input` | required | Input `.ttl` or `.nt` file. | `python scripts/sample_dbpedia.py path/to/dbpedia.ttl` |
| `--output-dir` | `data/dbpedia` | Output folder for `dbpedia_100.ttl`, `dbpedia_1000.ttl`, `dbpedia_5000.ttl`, and `dbpedia_10000.ttl`. | `--output-dir data/dbpedia_samples` |

### `src/main.py`

Reusable single-dataset CLI for the running example or a custom RDF file.

| Parameter | Default | What It Does |
| --- | --- | --- |
| `--encoding` | required | One of `basis`, `amplitude`, or `phase`. |
| `--input` | `data/running_example.ttl` | RDF input file. |
| `--rdf-format` | none | Optional RDFLib parser override. |
| `--fixed-thesis-mapping` | off | Use the fixed mapping for the included running example. |
| `--weights` | none | Comma-separated amplitude weights. |
| `--weight-strategy` | `uniform` | Amplitude fallback strategy: `uniform` or `linear`. |
| `--mark-predicate` | none | Predicate URI to mark in phase encoding. |
| `--mark-subject` | none | Subject URI to mark in phase encoding. |
| `--phase-angle` | pi | Phase shift for marked triples. |
| `--shots` | `2048` | Measurement shots. |
| `--results-dir` | `results` | Base folder for figures and JSON logs. |
| `--output-prefix` | derived from file and encoding | Prefix for output files. |
| `--skip-plots` | off | Skip figure generation. |

Example:

```powershell
python -m src.main --encoding phase --input data/running_example.ttl --mark-predicate http://example.org/teaches
```

## Phase Predicate Selection

Phase encoding is only meaningful when the phase oracle actually marks triples.
The synthetic generator always includes predicates such as
`http://example.org/p0`, so synthetic phase runs default to
`synthetic-default`, which marks `http://example.org/p0`.

Real KG files usually do not contain that synthetic predicate. For real KG
experiments, the default is therefore `most-common-predicate`: the runner counts
the parsed predicates and marks the predicate that appears most often. This
keeps real phase experiments from silently becoming no-op circuits.

Available modes:

| Mode | Behavior |
| --- | --- |
| `synthetic-default` | Mark `http://example.org/p0`. Best for generated synthetic KGs. |
| `first-predicate` | Mark the first predicate in the parsed deterministic triple list. |
| `most-common-predicate` | Mark the predicate with the highest count in the dataset. Default for real KGs. |
| `none` | Mark nothing. This is an explicit no-op/baseline phase run. |
| `custom` | Mark the URI supplied by `--phase-mark-predicate`; this mode requires that argument. |

The raw CSV records:

- `phase_marker_mode`
- `phase_requested_predicate`
- `phase_effective_predicate`
- `phase_marked_triples`
- `phase_total_triples`
- `phase_marked_fraction`
- `phase_warning`

If `phase_marked_triples` is `0`, the phase oracle did not mark any KG triple.
The row remains in the CSV, but `phase_warning` explains that the phase oracle
is a no-op for that dataset. Do not interpret such a row as a meaningful
predicate-marking phase experiment.

## Circuit Metric Levels

The scaling CSV reports circuit metrics at three levels because Qiskit circuits
can contain high-level instructions. In particular, `initialize(...)` and
`unitary(...)` may appear as one operation in the original circuit even though
their eventual physical implementation can require many gates.

| Metric family | Meaning |
| --- | --- |
| `logical_*` | Depth, gate count, and operation counts on the original high-level circuit. This is useful for comparing the implementation pipeline, but it is not a hardware gate estimate. |
| `decomposed_*` | Metrics after `circuit.decompose(reps=...)`, guarded by `--max-metric-qubits`. These expose more of the cost of high-level instructions when feasible. |
| `transpiled_*` | Metrics after Qiskit `transpile(..., basis_gates=["u", "cx"], optimization_level=0)`, also guarded by `--max-metric-qubits`. These are closer to a compiled circuit cost, but can be expensive for arbitrary state preparation or dense unitaries. |

The legacy columns `circuit_depth`, `gate_count`, and `operation_counts` are
kept for backward compatibility and are aliases for the logical metrics. In
thesis text, call them logical or high-level metrics, not hardware-level costs.

If decomposed or transpiled metrics are skipped or fail, the row records
`metric_status` as `metric_limit`, `not_requested`, or `error`, with details in
`metric_error`. Missing decomposed/transpiled points in plots mean the runner did
not fake those metrics.

## Simulator Limits And Run Statuses

Some configurations cannot run on a local dense simulator. This is expected and
is part of the thesis result.

### Why Basis Encoding Can Become Expensive

Basis encoding represents symbolic KG components directly:

```text
subject_bits || predicate_bits || object_bits
```

The qubit count depends on the number of unique entities and predicates, not
only the number of triples.

Examples from the synthetic generator:

- 5,000 triples with 2,500 entities and 15 predicates can require 28 basis qubits.
- 10,000 triples with 5,000 entities and 20 predicates can require 31 basis qubits.

Dense statevector memory grows as:

```text
2^n complex amplitudes
```

So a run may be recorded as `simulator_limit` instead of attempting an unsafe
simulation.

### Why Phase Encoding Can Become Expensive

The current phase implementation uses a dense diagonal phase oracle. A dense
matrix can become impractical as the padded triple-index space grows. The option
`--max-phase-diagonal-qubits` controls when the runner skips this dense oracle
path.

### Amplitude Encoding Costs

Amplitude encoding uses fewer qubits for many datasets because it works over
triple indices, but generic state preparation can still be computationally
expensive. It is not "free" just because the qubit count is smaller.

### Run Statuses

| Status | Meaning |
| --- | --- |
| `success` | The run completed and produced runtime/resource metrics. |
| `simulator_limit` | The runner skipped the simulation because a configured dense-simulation guardrail was exceeded. |
| `timeout` | The run exceeded `--timeout-seconds` and was terminated. |
| `parse_error` | RDF parsing failed for the dataset. The batch continued with the next configuration. |
| `error` | Missing file, invalid data, memory error, custom phase configuration error, or another exception occurred. The batch continued. |
| `skipped` | Not normally written by the current scripts; append mode prints "Skipping existing ..." for duplicate configurations. If a `skipped` row exists from manual or older data, it is preserved. |

Skipped, failed, timeout, and simulator-limit rows are not fake results. They
show practical feasibility limits and should be discussed honestly.

Circuit metric extraction has its own `metric_status` column. A run can have
`status=success` while `metric_status=metric_limit` if the main encoding and
simulation completed but decomposed/transpiled metric extraction was skipped by
`--max-metric-qubits`.

## Output Files

### Running Example Outputs

The single-dataset CLI writes:

```text
results/logs/
results/figures/
```

The JSON logs include timing data for RDF loading, ID mapping, encoding stages,
simulation stages, plotting, log writing, and total runtime.

### Scaling Outputs

Synthetic:

```text
results/scaling/synthetic/scaling_raw_results.csv
results/scaling/synthetic/scaling_summary.csv
results/scaling/synthetic/plots/
```

Real:

```text
results/scaling/real/scaling_raw_results.csv
results/scaling/real/scaling_summary.csv
results/scaling/real/plots/
```

Combined:

```text
results/scaling/combined/scaling_raw_results.csv
results/scaling/combined/scaling_summary.csv
results/scaling/combined/plots/
```

The current workflow writes scaling results only to the split `synthetic`,
`real`, and `combined` folders unless you explicitly pass a custom
`--output-dir` or `--results-root`.

### Raw CSV vs Summary CSV

| File | Meaning |
| --- | --- |
| `scaling_raw_results.csv` | One row per dataset, encoding, repetition, and simulator setting. This is the audit trail. |
| `scaling_summary.csv` | Grouped means and standard deviations by dataset and encoding. This is easier to plot and cite. |

Important raw CSV columns include:

- `dataset_category`
- `dataset_name`
- `dataset_path`
- `num_triples`
- `encoding`
- `repetition`
- `status`
- `error_message`
- `total_runtime`
- `rdf_parse_time`
- `id_mapping_time`
- `state_preparation_time`
- `circuit_construction_time`
- `simulation_time`
- `measurement_time`
- `qubits`
- `circuit_depth` (legacy alias for `logical_circuit_depth`)
- `gate_count` (legacy alias for `logical_gate_count`)
- `logical_circuit_depth`
- `logical_gate_count`
- `logical_operation_counts`
- `decomposed_circuit_depth`
- `decomposed_gate_count`
- `decomposed_operation_counts`
- `transpiled_circuit_depth`
- `transpiled_gate_count`
- `transpiled_operation_counts`
- `metric_status`
- `metric_error`
- `phase_marker_mode`
- `phase_requested_predicate`
- `phase_effective_predicate`
- `phase_marked_triples`
- `phase_total_triples`
- `phase_marked_fraction`
- `phase_warning`
- `shots`
- `backend`

## Plot Guide

### Synthetic Plots

Saved under:

```text
results/scaling/synthetic/plots/
```

| Plot | Meaning |
| --- | --- |
| `synthetic_total_runtime_vs_triples.png` | Completed full-run runtime as synthetic KG size grows. |
| `synthetic_state_preparation_time_vs_triples.png` | Time spent building encoding-specific state metadata or vectors. |
| `synthetic_simulation_time_vs_triples.png` | Local statevector simulation time when simulation succeeds. |
| `synthetic_qubits_vs_triples.png` | Qubit requirement for each encoding. |
| `synthetic_logical_circuit_depth_vs_triples.png` | High-level circuit depth before decomposition/transpilation. |
| `synthetic_logical_gate_count_vs_triples.png` | High-level operation count before decomposition/transpilation. |
| `synthetic_decomposed_circuit_depth_vs_triples.png` | Circuit depth after guarded `decompose(...)`, when available. |
| `synthetic_decomposed_gate_count_vs_triples.png` | Gate count after guarded `decompose(...)`, when available. |
| `synthetic_transpiled_circuit_depth_vs_triples.png` | Circuit depth after guarded Qiskit transpilation, when available. |
| `synthetic_transpiled_gate_count_vs_triples.png` | Gate count after guarded Qiskit transpilation, when available. |

### Real KG Plots

Saved under:

```text
results/scaling/real/plots/
```

| Plot | Meaning |
| --- | --- |
| `real_total_runtime_by_dataset.png` | Completed runtime by real KG dataset. |
| `real_state_preparation_time_by_dataset.png` | State preparation time by real KG dataset. |
| `real_simulation_time_by_dataset.png` | Simulation time by real KG dataset. |
| `real_qubits_by_dataset.png` | Qubit count by real KG dataset. |
| `real_logical_circuit_depth_by_dataset.png` | High-level circuit depth by real KG dataset. |
| `real_logical_gate_count_by_dataset.png` | High-level gate count by real KG dataset. |
| `real_decomposed_circuit_depth_by_dataset.png` | Decomposed circuit depth by real KG dataset, when available. |
| `real_decomposed_gate_count_by_dataset.png` | Decomposed gate count by real KG dataset, when available. |
| `real_transpiled_circuit_depth_by_dataset.png` | Transpiled circuit depth by real KG dataset, when available. |
| `real_transpiled_gate_count_by_dataset.png` | Transpiled gate count by real KG dataset, when available. |

Real plots use dataset names on the x-axis because the real files are a realism
check, not a controlled size ladder.

### Combined Plots

Saved under:

```text
results/scaling/combined/plots/
```

| Plot | Meaning |
| --- | --- |
| `combined_total_runtime_vs_triples.png` | Synthetic and real completed runtimes on one triple-count axis. |
| `combined_state_preparation_time_vs_triples.png` | State preparation trends for both dataset categories. |
| `combined_simulation_time_vs_triples.png` | Simulation-time trends for both dataset categories. |
| `combined_qubits_vs_triples.png` | Qubit requirements for both dataset categories. |
| `combined_logical_circuit_depth_vs_triples.png` | High-level circuit depth for both dataset categories. |
| `combined_logical_gate_count_vs_triples.png` | High-level gate count for both dataset categories. |
| `combined_decomposed_circuit_depth_vs_triples.png` | Decomposed circuit depth for both categories, when available. |
| `combined_decomposed_gate_count_vs_triples.png` | Decomposed gate count for both categories, when available. |
| `combined_transpiled_circuit_depth_vs_triples.png` | Transpiled circuit depth for both categories, when available. |
| `combined_transpiled_gate_count_vs_triples.png` | Transpiled gate count for both categories, when available. |

Combined legends distinguish category and encoding, for example:

```text
basis/synthetic
basis/real
amplitude/synthetic
amplitude/real
phase/synthetic
phase/real
```

Missing points usually mean the relevant metric was unavailable because the run
hit `simulator_limit`, `timeout`, or `error`. The raw CSV keeps those rows.

## Adding New Data Or Encodings

### Add A New KG Dataset

Example: add `my_graph.ttl`.

```powershell
Copy-Item .\my_graph.ttl .\data\real_kgs\my_graph.ttl
python scripts/run_all_experiments.py --skip-synthetic --real-files my_graph.ttl --repetitions 1 --shots 256 --timeout-seconds 300
```

Then inspect:

```text
results/scaling/real/scaling_raw_results.csv
results/scaling/real/scaling_summary.csv
results/scaling/real/plots/
```

To include the dataset by default, add it to `REAL_KG_FILES` in
[scripts/run_scaling_experiments.py](scripts/run_scaling_experiments.py).

### Add A New Synthetic Dataset Size

Edit `DATASET_CONFIGS` in
[scripts/generate_scaling_datasets.py](scripts/generate_scaling_datasets.py),
then regenerate:

```powershell
python scripts/generate_scaling_datasets.py
```

### Add A New Encoding

A practical path:

1. Create a new file such as `src/my_encoding.py`.
2. Implement functions that build the encoding metadata, circuit/state, and any
   measurement/decoding information.
3. Add the encoding name to `ENCODINGS` in
   [scripts/run_scaling_experiments.py](scripts/run_scaling_experiments.py).
4. Add a `run_my_encoding_scaling(...)` function similar to
   `run_basis_scaling`, `run_amplitude_scaling`, or `run_phase_scaling`.
5. Register it in `run_one_configuration(...)`.
6. Make sure these fields are populated when available:
   - `state_preparation_time`
   - `circuit_construction_time`
   - `simulation_time`
   - `measurement_time`
   - `qubits`
   - `logical_circuit_depth`
   - `logical_gate_count`
   - `decomposed_circuit_depth`, when feasible
   - `decomposed_gate_count`, when feasible
   - `transpiled_circuit_depth`, when feasible
   - `transpiled_gate_count`, when feasible
   - `status`
   - `error_message`
7. Regenerate plots. The plotting code already iterates over `ENCODINGS`.

Keep the first version simple. A new encoding should be comparable to the
existing three before it becomes optimized.

## Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and reinstall dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Qiskit Aer Not Installed

Install Qiskit and Aer explicitly:

```powershell
pip install qiskit qiskit-aer
```

### RDF Parse Error

Check that the file is valid RDF and that the suffix matches the content. The
parser supports `.ttl`, `.rdf`, `.xml`, and `.nt`. You can force a format:

```powershell
python scripts/run_all_experiments.py --skip-synthetic --real-files my_graph.rdf --rdf-format xml
```

### Unsupported RDF/XML File

Some `.xml` files are XML but not RDF/XML. If RDFLib cannot parse the file, the
row will be recorded with `status=parse_error`. Validate or convert the file to
Turtle if needed.

### Phase Run Marks Zero Triples

Check `phase_warning`, `phase_effective_predicate`, and
`phase_marked_triples` in the raw CSV. For real KGs, leave
`--phase-marker-mode` unset or use `--phase-marker-mode most-common-predicate`
unless you intentionally want a custom predicate or `none` baseline.

### Memory Error

Lower the simulator limits or accept the `simulator_limit` row as a feasibility
result:

```powershell
python scripts/run_scaling_experiments.py --sizes 5000 --max-basis-simulation-qubits 22
```

### Run Takes Too Long

Use a timeout, fewer shots, fewer repetitions, or fewer encodings:

```powershell
python scripts/run_all_experiments.py --encodings amplitude --repetitions 1 --shots 128 --timeout-seconds 120
```

### Plots Are Missing

Regenerate from existing raw CSVs:

```powershell
python scripts/run_scaling_experiments.py --regenerate-plots-only
python scripts/run_all_experiments.py --regenerate-plots-only
```

If a metric has no successful or available values, the plot may omit that series.

### CSV Was Overwritten

Use append mode for incremental experiments:

```powershell
python scripts/run_scaling_experiments.py --sizes 5000 --encodings amplitude --append
```

Use `--output-dir` or `--results-root` for isolated tests.

### PowerShell Execution Policy

If activation fails:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This changes the policy only for the current PowerShell process.

### `data/real_kgs/` Folder Not Found

Create it and add RDF files:

```powershell
New-Item -ItemType Directory -Force data\real_kgs
Copy-Item .\my_graph.ttl .\data\real_kgs\my_graph.ttl
```

### Empty Dataset / Zero Triples

The ID-mapping stage requires at least one triple. Check the RDF file and parse
format. Empty graphs are recorded as `status=error`.

## Using The Results In The Thesis

Use the results carefully:

- Use synthetic datasets for controlled scalability analysis.
- Use real KGs from `data/real_kgs/` as realism checks.
- Use combined plots to see whether real KG behavior is consistent with
  synthetic trends.
- Do not claim quantum advantage.
- Do not present this as a DBpedia database benchmark.
- Report simulator limits, timeouts, and errors honestly.
- Present `circuit_depth` and `gate_count` as logical/high-level aliases, and
  use decomposed/transpiled metrics where feasible for circuit-cost discussion.
- For real KG phase experiments, report the selected predicate and the marked
  fraction so a no-op phase run is not mistaken for a meaningful oracle.
- Emphasize trade-offs:
  - basis encoding is explicit and directly symbolic, but qubits can grow fast
  - amplitude encoding can be compact in qubits, but state preparation matters
  - phase encoding supports marking/interference workflows, but dense oracle
    construction can be expensive

A LaTeX-ready draft is available at:

```text
results/scaling/paper_scaling_section_draft.tex
```

## Reproducibility Checklist

Before reporting results, record:

- Python version:
  ```powershell
  python --version
  ```
- Dependency versions:
  ```powershell
  pip freeze
  ```
- Synthetic datasets generated:
  ```powershell
  python scripts/generate_scaling_datasets.py
  ```
- Real KG files present in `data/real_kgs/`
- Exact command used
- Dataset sizes
- Real KG filenames
- Encodings
- Repetitions
- Shots
- Timeout setting
- `--max-basis-simulation-qubits`
- `--max-phase-diagonal-qubits`
- `--max-metric-qubits`
- `--decompose-reps`
- Whether decomposed/transpiled metrics were enabled
- `--phase-marker-mode`
- `--phase-mark-predicate`, if used
- Output raw CSV paths
- Output summary CSV paths
- Output plot folders

For final thesis tables and figures, prefer the CSVs under:

```text
results/scaling/synthetic/
results/scaling/real/
results/scaling/combined/
```

## Extra Documentation

More detailed scaling notes are in:

```text
docs/scaling_experiments.md
```
