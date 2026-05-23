# Chapter 9 Run Summary

Results are simulator-based software-level observations and do not show quantum advantage.

## Command

```text
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts/run_chapter9_experiments.py --shots 2048 --repetitions 5 --index-mode paper --include-synthetic --synthetic-sizes 6 10 25 50 100 250 500 --synthetic-repetitions 3 --include-real --max-real-triples 500 --output-dir results/chapter9
```

## Command-Line Arguments

- `backend`: `aer_simulator`
- `include_combined`: `True`
- `include_real`: `True`
- `include_synthetic`: `True`
- `index_mode`: `paper`
- `max_real_triples`: `500`
- `output_dir`: `results/chapter9`
- `real_kg_files`: `None`
- `repetitions`: `5`
- `save_csv`: `True`
- `save_json`: `True`
- `seed`: `12345`
- `shots`: `2048`
- `synthetic_repetitions`: `3`
- `synthetic_sizes`: `[6, 10, 25, 50, 100, 250, 500]`

## Generated Tables

- results\chapter9\table3_encoding_process.csv
- results\chapter9\table3_encoding_process.tex
- results\chapter9\table4_usage_tasks.csv
- results\chapter9\table4_usage_tasks.tex
- results\chapter9\table6_circuit_statistics.csv
- results\chapter9\table6_circuit_statistics.tex
- results\chapter9\table7_synthetic_results.csv
- results\chapter9\table7_synthetic_results.tex
- results\chapter9\table8_real_kg_results.csv
- results\chapter9\table8_real_kg_results.tex

## Generated Plots

- results\chapter9\figures\table3_encoding_process_runtime.png
- results\chapter9\figures\table4_usage_task_runtime.png
- results\chapter9\figures\table3_encoding_time_bar.png
- results\chapter9\figures\table3_qubits_bar.png
- results\chapter9\figures\table4_task_time_bar.png
- results\chapter9\figures\amplitude_probabilities.png
- results\chapter9\figures\combined_magnitude_phase.png
- results\chapter9\figures\synthetic_encoding_time.png
- results\chapter9\figures\synthetic_qubits.png
- results\chapter9\figures\synthetic_depth.png
- results\chapter9\figures\synthetic_total_time.png

## Generated Data Files

- results\chapter9\chapter9_raw_results.json
- results\chapter9\synthetic_raw_results.json
- results\chapter9\real_kg_raw_results.json
- results\chapter9\environment.json
- results\chapter9\RUN_SUMMARY.md

## Section 9.2 Synthetic Sizes

- 6
- 10
- 25
- 50
- 100
- 250
- 500

## Section 9.3 Real KG Files

- data\real_kgs\exampleV3.ttl
- data\real_kgs\productsSmall.rdf
- data\real_kgs\DecodedOntologies_V2.ttl
- data\real_kgs\Aristotle.xml

## Skipped Or Failed Experiments

- 6 (combined): skipped (3 rows); Combined synthetic scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- 10 (combined): skipped (3 rows); Combined synthetic scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- 25 (combined): skipped (3 rows); Combined synthetic scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- 50 (combined): skipped (3 rows); Combined synthetic scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- 100 (combined): skipped (3 rows); Combined synthetic scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- 250 (combined): skipped (3 rows); Combined synthetic scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- 500 (combined): skipped (3 rows); Combined synthetic scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- exampleV3 (combined): skipped; Combined real-KG scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- productsSmall (combined): skipped; Combined real-KG scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- DecodedOntologies_V2 (combined): skipped; Combined real-KG scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim
- Aristotle (combined): skipped; Combined real-KG scaling is not supported by the existing scalability runner; skipped. software-level observation; no quantum-advantage claim

## Environment

- Timestamp UTC: `2026-05-23T21:12:27+00:00`
- Hostname: `DESKTOP-8ICK71U`
- Python: `3.13.5`
- Qiskit: `2.4.1`
- Qiskit Aer: `0.17.2`
- NumPy: `1.26.4`
- rdflib: `7.6.0`
- OS: `Windows-11-10.0.26200-SP0`
- CPU: `Intel64 Family 6 Model 151 Stepping 5, GenuineIntel`
- RAM: `39.75 GiB`
- Git commit: `bdd5c6ba16bf4450bafb18797a11d0e6c16163a1`
- Random seed: `12345`
