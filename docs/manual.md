# Qiskit KG Encodings Manual

This is a compact guide to the code layout and normal experiment workflow.

For the shortest path, run:

```powershell
.\scripts\run_experiments_windows.ps1 -Profile light -ResultsRoot results\runs
```

That command creates one timestamped result folder and a zip archive.

## Code Layout

Core modules live under `src/`.

| Module | Purpose |
| --- | --- |
| `kg_parser.py` | Load RDF files and extract triples. |
| `models.py` | Shared data structures. |
| `id_mapper.py` | Build deterministic entity and predicate mappings. |
| `basis_encoding.py` | Encode triples as computational basis states. |
| `amplitude_encoding.py` | Build amplitude vectors over triple indices. |
| `phase_encoding.py` | Build phase-marking circuits and phase helpers. |
| `combined_encoding.py` | Combine amplitude magnitudes with predicate phases. |
| `visualization.py` | Save JSON logs and plots for lower-level CLI runs. |
| `main.py` | Reusable single-encoding command-line entry point. |
| `tasks/` | Named task experiments used by the suite runner. |

Automation lives under `scripts/`.

| Script | Purpose |
| --- | --- |
| `run_experiments_windows.ps1` | Main Windows profile runner. |
| `run_experiment_suite.py` | Core task-level suite. |
| `collect_run_outputs.py` | Collect artifacts and build normalized summary CSV. |
| `write_environment_info.py` | Save Python/package/OS/git metadata. |
| `run_all_experiments.py` | Synthetic plus real scaling runner. |
| `run_scaling_experiments.py` | Synthetic-only scaling runner. |
| `generate_scaling_datasets.py` | Generate deterministic synthetic datasets. |
| `run_classical_baselines.py` | Run classical sanity baselines. |

## Main Runner

From the repository root:

```powershell
.\scripts\run_experiments_windows.ps1 -Profile light
.\scripts\run_experiments_windows.ps1 -Profile medium
.\scripts\run_experiments_windows.ps1 -Profile hard
```

With an explicit results root:

```powershell
.\scripts\run_experiments_windows.ps1 -Profile medium -ResultsRoot results\runs
```

Use `-InstallDeps` only when dependencies need to be installed:

```powershell
.\scripts\run_experiments_windows.ps1 -Profile light -InstallDeps
```

## Profiles

| Profile | Use | Shots | Repetitions |
| --- | --- | ---: | ---: |
| `light` | Quick smoke test. | 1,000 | 1 |
| `medium` | Main desktop-sized run. | 10,000 | 3 |
| `hard` | Longer scaling run. | 10,000 | 5 |

The profiles differ by shots, repetitions, synthetic sizes, real dataset use,
and timeout behavior.

## Result Folder

Each run creates:

```text
results/runs/<timestamp>_<profile>/
```

The folder contains:

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

Start with:

- `run_config.json`
- `run_manifest.json`
- `summary/results_summary.csv`
- `summary/artifact_index.csv`
- `logs/`

## Core Task Experiments

The core suite runs:

- basis lookup
- amplitude similarity
- phase filtering
- entity matching style distance/similarity check
- keyword search
- schema matching
- multi-hop phase accumulation
- combined amplitude + phase encoding

Each task writes structured outputs under the run folder. The collector merges
numeric rows into:

```text
summary/results_summary.csv
```

## Basic Encoding CLI

You can run one encoding directly:

```powershell
python -m src.main --encoding basis --input data/running_example.ttl
python -m src.main --encoding amplitude --input data/running_example.ttl
python -m src.main --encoding phase --input data/running_example.ttl
```

Useful options:

```powershell
--shots 1000
--results-dir results\scratch_single
--output-prefix my_run
--skip-plots
--weights 2,1,3,1,1,2
--mark-predicate http://example.org/teaches
--mark-subject http://example.org/Aristotle
```

Use the profile runner for complete experiment runs. Use `src.main` for focused
debugging of one encoding.

## Data

Default small RDF input:

```text
data/running_example.ttl
```

Synthetic scaling inputs:

```text
data/scaling/
```

Local real inputs:

```text
data/real_kgs/
```

Generate synthetic files with:

```powershell
python scripts/generate_scaling_datasets.py
```

## Scaling

The profile runner calls the scaling scripts automatically.

For direct synthetic-only scaling:

```powershell
python scripts/run_scaling_experiments.py --sizes 100 1000 --repetitions 1 --shots 1000 --output-dir results\scratch_scaling
```

For direct synthetic plus real scaling:

```powershell
python scripts/run_all_experiments.py --results-root results\scratch_all --synthetic-sizes 100 1000 --repetitions 1 --shots 1000 --timeout-seconds 300
```

See `docs/scaling_experiments.md` for scaling details.

## Testing

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The Windows runner also runs tests when the `tests/` folder exists.

## Failure Handling

The runner records every command in:

```text
command_log.jsonl
run_manifest.json
logs/<step>.log
```

Every log contains the command, working directory, stdout, stderr, and finish
time. A failed command records its exit code or timeout reason.

For large experiments, a skipped, timed-out, or simulator-limited row is still a
valid output row. Do not replace those rows with invented values.

## Adding Code

When adding experiments:

1. Put reusable logic under `src/`.
2. Put named task logic under `src/tasks/`.
3. Add CLI support in `scripts/` when needed.
4. Write all outputs under `--output-dir`.
5. Save CSV/JSON for metrics and raw details.
6. Save PNG for plots or histograms.
7. Return proper exit codes.
8. Add tests.

Keep generated outputs inside the active run folder.
