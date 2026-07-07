# Qiskit KG Encodings

This repository runs experiments for Qiskit-based knowledge graph encodings.
It is for producing numbers, plots, logs, JSON payloads, and result tables from
local simulations.

It is not a document editor or a text-analysis tool. The code does not rewrite
or evaluate documents. Run the experiments, inspect the generated result
folder, and use those outputs however you like outside this repository.

## What The Code Measures

The project compares small knowledge graph encoding workflows:

- basis encoding
- amplitude encoding
- phase encoding
- combined amplitude + phase checks
- task-level checks such as lookup, similarity, keyword search, schema matching,
  phase accumulation, scaling, and classical baselines

The experiments measure implementation-level behavior:

- RDF loading and parsing
- deterministic ID mapping
- state or circuit construction time
- local simulation time
- measurement counts and probabilities
- runtime
- qubit counts
- circuit depth
- gate counts
- pass/fail status for small validation checks
- scaling behavior across synthetic and local real datasets

The outputs are local simulator results and software measurements. Do not treat
them as hardware results or quantum-advantage evidence.

## Recommended Way To Run

Use the Windows PowerShell runner from the repository root:

```powershell
.\scripts\run_experiments_windows.ps1 -Profile light
.\scripts\run_experiments_windows.ps1 -Profile medium
.\scripts\run_experiments_windows.ps1 -Profile hard
```

You can choose a different results root:

```powershell
.\scripts\run_experiments_windows.ps1 -Profile medium -ResultsRoot results\runs
```

Install dependencies through the runner only when you need to:

```powershell
.\scripts\run_experiments_windows.ps1 -Profile light -InstallDeps
```

The runner detects Python in this order:

1. `.\.venv\Scripts\python.exe`
2. `python` from `PATH`

It also sets:

- `PYTHONPATH` to the repository root
- `MPLBACKEND=Agg` so plots are saved without opening windows

## Profiles

Only three profiles are supported.

| Profile | Purpose | Shots | Repetitions | Dataset behavior |
| --- | --- | ---: | ---: | --- |
| `light` | Smoke test that should finish quickly. | 1,000 | 1 | Small synthetic data only. |
| `medium` | Main desktop-sized run. | 10,000 | 3 | Small/medium synthetic data plus available local real datasets. |
| `hard` | Longer scaling run. | 10,000 | 5 | Larger synthetic data and available local real datasets. Optional large failures are recorded and the run continues where safe. |

## Result Folder

Every runner invocation creates a new timestamped folder:

```text
results/runs/<timestamp>_<profile>/
```

Inside it:

```text
logs/
tables/
plots/
json/
raw/
summary/
circuits/
histograms/
run_config.json
run_manifest.json
command_log.jsonl
```

The runner also creates:

```text
results/runs/<timestamp>_<profile>.zip
```

Old runs are not overwritten.

## Most Important Outputs

| File or folder | Purpose |
| --- | --- |
| `run_config.json` | Profile, timestamp, Python version, package versions, seed, shots, repetitions, dataset sizes, OS, command line, and git commit when available. |
| `run_manifest.json` | Every command and core experiment attempted, status, runtime, log path, output files, and error message when a step fails. |
| `command_log.jsonl` | Machine-readable command log, one JSON object per command. |
| `logs/*.log` | Full stdout/stderr for each runner step. |
| `summary/results_summary.csv` | Normalized numeric rows from core experiments, scaling runs, and classical baselines. |
| `summary/artifact_index.csv` | Index of generated files inside the run folder. |
| `tables/*.csv` | Experiment-level metric tables. |
| `json/*.json` | Structured payloads and run metadata. |
| `raw/` | Raw metrics and outputs from the experiment scripts. |
| `plots/*.png` | Summary plots. |
| `histograms/*.png` | Measurement-count histograms. |
| `circuits/*.json` | Circuit metric snapshots. |

If a command fails, the runner prints the failed command and the exact log path.

## Quick Start From A Fresh Clone

```powershell
git clone <repository-url>
cd Quskit-KG-Ecodings
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\scripts\run_experiments_windows.ps1 -Profile light -ResultsRoot results\runs
```

When the run finishes, open the printed result folder and start with:

```text
run_manifest.json
run_config.json
summary/results_summary.csv
summary/artifact_index.csv
logs/
```

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   |-- running_example.ttl
|   |-- scaling/
|   `-- real_kgs/
|-- docs/
|   |-- manual.md
|   `-- scaling_experiments.md
|-- experiments/
|-- results/
|-- scripts/
|-- src/
|   |-- tasks/
|   |-- amplitude_encoding.py
|   |-- basis_encoding.py
|   |-- combined_encoding.py
|   |-- id_mapper.py
|   |-- kg_parser.py
|   |-- main.py
|   |-- models.py
|   |-- phase_encoding.py
|   `-- visualization.py
|-- tests/
`-- tools/
```

## Folder Guide

| Folder | Purpose |
| --- | --- |
| `src/` | Core parsing, mapping, encoding, simulation, plotting, and task code. |
| `src/tasks/` | Small task-level experiments such as lookup, similarity, keyword search, schema matching, and phase accumulation. |
| `scripts/` | Command-line runners and result collectors. |
| `experiments/` | Thin wrappers for running the basic encoding pipelines on the included example graph. |
| `data/` | Included RDF inputs, generated synthetic data, and local real KG files. |
| `results/` | Generated outputs. Safe to delete if you want a clean rerun. |
| `docs/` | Additional usage notes. |
| `tests/` | Unit tests for task logic, scripts, and runner behavior. |

## Main Scripts

| Script | Use |
| --- | --- |
| `scripts/run_experiments_windows.ps1` | Main Windows runner. Creates one timestamped run folder and zip. |
| `scripts/run_experiment_suite.py` | Runs the core task-level experiment suite and writes normalized CSV/JSON/PNG outputs. Usually called by the Windows runner. |
| `scripts/collect_run_outputs.py` | Builds `summary/results_summary.csv` and `summary/artifact_index.csv` for one run folder. Usually called by the Windows runner. |
| `scripts/write_environment_info.py` | Writes Python/package/OS/git metadata for `run_config.json`. Usually called by the Windows runner. |
| `scripts/run_all_experiments.py` | Runs synthetic and real scaling groups. Usually called by the Windows runner. |
| `scripts/run_scaling_experiments.py` | Runs synthetic scaling only. Useful for focused scaling work. |
| `scripts/generate_scaling_datasets.py` | Generates deterministic synthetic Turtle datasets. |
| `scripts/run_classical_baselines.py` | Runs deterministic classical baseline checks. |

## Data

The included running example is:

```text
data/running_example.ttl
```

Generated synthetic datasets live under:

```text
data/scaling/
```

Local real KG files live under:

```text
data/real_kgs/
```

The runner uses local files only. It does not download datasets.

## Running Lower-Level Experiments

The profile runner is the normal entry point. Use lower-level scripts only when
you want focused development or debugging.

Run the reusable encoding CLI:

```powershell
python -m src.main --encoding basis --input data/running_example.ttl
python -m src.main --encoding amplitude --input data/running_example.ttl
python -m src.main --encoding phase --input data/running_example.ttl
```

Run the core suite directly:

```powershell
python scripts/run_experiment_suite.py --output-dir results\scratch_core --profile light --shots 1000 --repetitions 1
```

Run synthetic scaling directly:

```powershell
python scripts/generate_scaling_datasets.py
python scripts/run_scaling_experiments.py --sizes 100 1000 --repetitions 1 --shots 1000 --output-dir results\scratch_scaling
```

Run synthetic plus real scaling directly:

```powershell
python scripts/run_all_experiments.py --results-root results\scratch_all --synthetic-sizes 100 1000 --repetitions 1 --shots 1000 --timeout-seconds 300
```

Run classical baselines directly:

```powershell
python scripts/run_classical_baselines.py --output-dir results\scratch_baselines
```

## Core Experiment Suite

The core suite currently attempts:

- basis lookup
- amplitude similarity
- phase filtering
- entity matching style distance/similarity check
- keyword search
- schema matching
- multi-hop phase accumulation
- combined amplitude + phase encoding

Each successful experiment writes:

- JSON payload in `json/`
- raw JSON in `raw/`
- metrics CSV in `tables/`
- circuit metrics in `circuits/`
- histogram PNG when measurement counts are available

Missing or failing experiments are not faked. They are recorded as failures in
the manifest with the error message and log path.

## Scaling Experiments

Scaling rows are written under the run folder at:

```text
raw/scaling/synthetic/
raw/scaling/real/
raw/scaling/combined/
```

The important files are:

```text
scaling_raw_results.csv
scaling_summary.csv
plots/
```

For more detail, see [docs/scaling_experiments.md](docs/scaling_experiments.md).

## Tests

Run tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The Windows runner also runs tests automatically when the `tests/` folder is
present.

## Troubleshooting

### PowerShell Blocks Virtualenv Activation

Run PowerShell as your normal user and use:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Missing Dependencies

Install once:

```powershell
pip install -r requirements.txt
```

Or ask the runner to install:

```powershell
.\scripts\run_experiments_windows.ps1 -Profile light -InstallDeps
```

### A Runner Step Fails

Open:

```text
run_manifest.json
command_log.jsonl
logs/<step>.log
```

The runner records the exact command, exit code, runtime, status, stdout, and
stderr.

### A Large Experiment Is Skipped Or Limited

That can be a valid result. Some dense simulations or metric extractions are
guarded because statevector memory and circuit expansion can grow quickly. Use
the raw CSV status and error columns to distinguish successful rows from
timeouts, simulator limits, parse errors, and other failures.

## Adding A New Experiment

Prefer this pattern:

1. Put reusable logic under `src/`.
2. Put task-specific logic under `src/tasks/` when it is a named experiment.
3. Add a CLI or extend `scripts/run_experiment_suite.py`.
4. Accept standard arguments where relevant:
   - `--output-dir`
   - `--shots`
   - `--repetitions`
   - `--seed`
   - `--dataset`
   - `--dataset-size`
   - `--profile`
5. Write outputs only under the provided output directory.
6. Produce machine-readable CSV/JSON outputs.
7. Return a nonzero exit code on failure.
8. Add or update tests.

Do not scatter generated files around the repository root.

## Reproducibility Checklist

For any run you want to keep, save or archive:

- the whole timestamped run folder
- the generated zip file
- `run_config.json`
- `run_manifest.json`
- `summary/results_summary.csv`
- `summary/artifact_index.csv`
- all relevant raw CSV/JSON files

The runner already packages the run folder into a zip at the end.
