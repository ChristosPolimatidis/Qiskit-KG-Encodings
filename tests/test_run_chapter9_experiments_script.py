from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def load_chapter9_runner(repo_root: Path):
    script = repo_root / "scripts" / "run_chapter9_experiments.py"
    spec = importlib.util.spec_from_file_location("chapter9_runner_for_test", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_chapter9_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_reported_outputs_under(
    test_case: unittest.TestCase,
    output_dir: Path,
    raw_payload: dict,
) -> None:
    resolved_output_dir = output_dir.resolve()
    output_files = raw_payload["output_files"]
    reported_paths = [
        value
        for value in output_files.values()
        if value not in (None, "") and not isinstance(value, list)
    ]
    reported_paths.extend(output_files.get("figures", []))

    for value in reported_paths:
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path
        test_case.assertTrue(
            path.resolve().is_relative_to(resolved_output_dir),
            f"{path} is not under {resolved_output_dir}",
        )


class RunChapter9ExperimentsScriptTests(unittest.TestCase):
    def test_script_writes_chapter9_outputs_only(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_chapter9_experiments.py"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "chapter9"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--shots",
                    "64",
                    "--repetitions",
                    "1",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Chapter 9 experiments complete", completed.stdout)
            self.assertIn("Machine:", completed.stdout)
            self.assertIn("old scalability experiments were not run", completed.stdout)
            self.assertTrue((output_dir / "table3_encoding_process.csv").exists())
            self.assertTrue((output_dir / "table4_usage_tasks.csv").exists())
            self.assertTrue((output_dir / "table6_circuit_statistics.csv").exists())
            self.assertFalse((output_dir / "table7_synthetic_results.csv").exists())
            self.assertFalse((output_dir / "table8_real_kg_results.csv").exists())
            self.assertTrue((output_dir / "table3_encoding_process.tex").exists())
            self.assertTrue((output_dir / "table4_usage_tasks.tex").exists())
            self.assertTrue((output_dir / "table6_circuit_statistics.tex").exists())
            self.assertFalse((output_dir / "table7_synthetic_results.tex").exists())
            self.assertFalse((output_dir / "table8_real_kg_results.tex").exists())
            self.assertTrue((output_dir / "chapter9_raw_results.json").exists())
            self.assertFalse((output_dir / "synthetic_raw_results.json").exists())
            self.assertFalse((output_dir / "real_kg_raw_results.json").exists())
            self.assertTrue((output_dir / "environment.json").exists())
            self.assertTrue((output_dir / "RUN_SUMMARY.md").exists())
            if importlib.util.find_spec("matplotlib") is not None:
                for figure_name in (
                    "table3_encoding_time_bar.png",
                    "table3_qubits_bar.png",
                    "table4_task_time_bar.png",
                    "amplitude_probabilities.png",
                    "combined_magnitude_phase.png",
                ):
                    self.assertTrue((output_dir / "figures" / figure_name).exists())

            table3_header = (
                (output_dir / "table3_encoding_process.csv")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            table4_header = (
                (output_dir / "table4_usage_tasks.csv")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            table6_text = (output_dir / "table6_circuit_statistics.csv").read_text(
                encoding="utf-8"
            )
            table6_lines = table6_text.splitlines()
            table4_csv = (output_dir / "table4_usage_tasks.csv").read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                table3_header,
                (
                    "Encoding,Variant,Index Mode,Qubits,Dimension,"
                    "Time to Create (ms),Circuit Depth,Gate Count,"
                    "Transpiled Depth,Transpiled Gate Count,Notes"
                ),
            )
            self.assertEqual(
                table4_header,
                "KG Task,Encoding,Quantum Method,Main Result,Time",
            )
            self.assertEqual(
                table6_lines[0],
                (
                    "Task,Encoding,Method,Qubits,Circuit Depth,Gate Count,"
                    "Transpiled Depth,Transpiled Gate Count,Shots,Repetitions"
                ),
            )
            for expected_row in (
                "Search,Basis,Grover",
                "Entity Matching,Amplitude,Swap Test",
                "Link Prediction,Amplitude,Distance Estimation",
                "Multi-hop Reasoning,Phase,Phase Kickback",
                "Schema Matching,Phase,QFT",
            ):
                self.assertIn(expected_row, table6_text)
            self.assertIn(",0,1", table6_text)
            self.assertIn("--", table6_text)
            table3_tex = (output_dir / "table3_encoding_process.tex").read_text(
                encoding="utf-8"
            )
            table4_tex = (output_dir / "table4_usage_tasks.tex").read_text(
                encoding="utf-8"
            )
            table6_tex = (output_dir / "table6_circuit_statistics.tex").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "\\begin{table}",
                table3_tex,
            )
            self.assertIn(
                (
                    "Encoding & Variant & Index Mode & Qubits & Time to Create (ms) "
                    "& Circuit Depth & Notes"
                ),
                table3_tex,
            )
            self.assertNotIn("Dimension", table3_tex)
            self.assertNotIn("Transpiled Depth", table3_tex)
            self.assertNotIn("Transpiled Gate Count", table3_tex)
            self.assertIn(
                "KG Task & Encoding & Quantum Method & Main Result & Time",
                table4_tex,
            )
            self.assertIn("sequential-only validation task", table4_csv)
            self.assertIn("sequential-only validation task", table4_tex)
            for expected_row in (
                "Search,Basis,Grover lookup",
                "Entity Matching,Amplitude,Swap Test",
                "Link Prediction,Amplitude,Distance Estimation",
                "Multi-hop Reasoning,Phase,Phase Kickback",
                "Schema Matching,Phase,QFT",
            ):
                self.assertIn(expected_row, table4_csv)
            self.assertNotIn(
                "chapter9_raw_results",
                table4_tex,
            )
            self.assertIn(
                (
                    "Task & Encoding & Method & Qubits & Circuit Depth & Gate Count "
                    "& Transpiled Depth & Transpiled Gate Count & Shots & Repetitions"
                ),
                table6_tex,
            )

            payload = json.loads(
                (output_dir / "chapter9_raw_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["running_example_triple_count"], 6)
            self.assertEqual(len(payload["encoding_process_rows"]), 6)
            self.assertEqual(len(payload["usage_task_rows"]), 5)
            self.assertEqual(len(payload["additional_validation_rows"]), 1)
            self.assertEqual(len(payload["table6_circuit_statistics"]), 5)
            self.assertEqual(payload["table7_synthetic_results"], [])
            self.assertEqual(payload["table8_real_kg_results"], [])
            self.assertIsNone(payload["output_files"]["table7_synthetic_results"])
            self.assertIsNone(payload["output_files"]["table8_real_kg_results"])
            self.assertIsNone(payload["output_files"]["synthetic_raw_results"])
            self.assertIsNone(payload["output_files"]["real_kg_raw_results"])
            assert_reported_outputs_under(self, output_dir, payload)
            self.assertNotIn(
                "combined_amplitude_phase_demo",
                [row["task"] for row in payload["usage_task_rows"]],
            )

            environment = json.loads(
                (output_dir / "environment.json").read_text(encoding="utf-8")
            )
            for field in (
                "python_version",
                "operating_system",
                "cpu",
                "total_ram_bytes",
                "qiskit_version",
                "qiskit_aer_version",
                "numpy_version",
                "rdflib_version",
                "command_line_arguments",
                "timestamp_utc",
                "random_seed",
                "git_commit_hash",
                "hostname",
                "exact_command_line",
                "generated_tables",
                "generated_plots",
                "generated_data_files",
                "synthetic_sizes_used",
                "real_kg_files_used",
                "skipped_or_failed_experiments",
                "software_level_observation_note",
                "run_summary",
            ):
                self.assertIn(field, environment)
            self.assertEqual(environment["random_seed"], 12345)
            self.assertEqual(environment["command_line_arguments"]["shots"], 64)
            self.assertIn("logical_cpus", environment["cpu"])
            self.assertIn("table3_encoding_process.csv", str(environment["generated_tables"]))
            self.assertIn("table6_circuit_statistics.csv", str(environment["generated_tables"]))
            self.assertIn("amplitude_probabilities.png", str(environment["generated_plots"]))
            self.assertIn("chapter9_raw_results.json", str(environment["generated_data_files"]))
            self.assertIn("RUN_SUMMARY.md", str(environment["generated_data_files"]))
            self.assertEqual(environment["synthetic_sizes_used"], [])
            self.assertEqual(environment["real_kg_files_used"], [])
            self.assertIn("simulator-based software-level observations", environment["software_level_observation_note"])

            run_summary = (output_dir / "RUN_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("# Chapter 9 Run Summary", run_summary)
            self.assertIn("## Command-Line Arguments", run_summary)
            self.assertIn("table3_encoding_process.csv", run_summary)
            self.assertIn("amplitude_probabilities.png", run_summary)
            self.assertIn("Not run", run_summary)
            self.assertIn("do not show quantum advantage", run_summary)
            self.assertIn("Qiskit", run_summary)

    def test_usage_tasks_make_index_mode_behavior_explicit(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        runner = load_chapter9_runner(repo_root)
        args = argparse.Namespace(
            shots=32,
            repetitions=1,
            index_mode="paper",
            backend="aer_simulator",
            seed=12345,
            include_combined=True,
        )

        rows = runner.run_usage_tasks(args)
        rows_by_task = {row["task"]: row for row in rows}

        basis = rows_by_task["search_grover_lookup"]
        self.assertEqual(basis["index_mode"], "sequential")
        self.assertIn("sequential-only validation task", basis["notes"])
        self.assertIn("sequential-only validation task", basis["result"].claim_note)

        amplitude = rows_by_task["entity_matching_swap_test"]
        self.assertEqual(amplitude["index_mode"], "feature_vector")
        self.assertIn("Index-mode independent", amplitude["notes"])

        link = rows_by_task["link_prediction_distance_estimation"]
        self.assertEqual(link["index_mode"], "feature_vector")
        self.assertIn("not HHL", link["notes"])

        multihop = rows_by_task["multihop_phase_kickback"]
        self.assertEqual(multihop["index_mode"], "path_phase")
        self.assertIn("not a full RDFS reasoner", multihop["notes"])

        schema = rows_by_task["schema_matching_qft"]
        self.assertEqual(schema["index_mode"], "phase_pattern")
        self.assertIn("not full schema matching", schema["notes"])

        additional = runner.run_additional_validations(args)
        self.assertEqual(additional[0]["task"], "combined_amplitude_phase_demo")
        self.assertEqual(additional[0]["index_mode"], "paper")
        self.assertEqual(additional[0]["result"].index_mode, "paper")
        self.assertEqual(additional[0]["result"].num_qubits, 8)

    def test_include_synthetic_writes_section_9_2_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_chapter9_experiments.py"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "chapter9"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--shots",
                    "32",
                    "--repetitions",
                    "1",
                    "--include-synthetic",
                    "--synthetic-sizes",
                    "6",
                    "--synthetic-repetitions",
                    "1",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Synthetic software-level observation rows: 4", completed.stdout)
            table7_path = output_dir / "table7_synthetic_results.csv"
            table7_tex_path = output_dir / "table7_synthetic_results.tex"
            synthetic_json_path = output_dir / "synthetic_raw_results.json"
            self.assertTrue(table7_path.exists())
            self.assertTrue(table7_tex_path.exists())
            self.assertTrue(synthetic_json_path.exists())
            for figure_name in (
                "synthetic_encoding_time.png",
                "synthetic_qubits.png",
                "synthetic_depth.png",
                "synthetic_total_time.png",
            ):
                self.assertTrue((output_dir / "figures" / figure_name).exists())

            table7_text = table7_path.read_text(encoding="utf-8")
            self.assertEqual(
                table7_text.splitlines()[0],
                (
                    "triple_count,entity_count,predicate_count,encoding,index_mode,"
                    "qubits,dimension,preprocessing_time_ms,encoding_time_ms,"
                    "circuit_construction_time_ms,transpilation_time_ms,"
                    "simulation_time_ms,total_time_ms,circuit_depth,gate_count,"
                    "transpiled_depth,transpiled_gate_count,status,notes"
                ),
            )
            for encoding in ("basis", "amplitude", "phase", "combined"):
                self.assertIn(f",{encoding},", table7_text)
            self.assertIn("combined,unsupported", table7_text)
            self.assertIn("skipped", table7_text)
            self.assertIn("software-level observation", table7_text)
            self.assertIn("no quantum-advantage claim", table7_text)

            table7_tex = table7_tex_path.read_text(encoding="utf-8")
            self.assertIn(
                "Triples & Encoding & Qubits & Enc. Time & Depth & Sim. Time & Status",
                table7_tex,
            )
            self.assertIn("software-level observations", table7_tex)

            synthetic_payload = json.loads(synthetic_json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(synthetic_payload["rows"]), 4)
            self.assertEqual(len(synthetic_payload["raw_scaling_rows"]), 3)
            self.assertEqual(synthetic_payload["settings"]["sizes"], [6])

            raw_payload = json.loads(
                (output_dir / "chapter9_raw_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(raw_payload["table7_synthetic_results"]), 4)
            self.assertEqual(len(raw_payload["synthetic_observations"]["rows"]), 4)
            self.assertIsNotNone(raw_payload["output_files"]["table7_synthetic_results"])
            self.assertIsNone(raw_payload["output_files"]["table8_real_kg_results"])
            self.assertIsNotNone(raw_payload["output_files"]["synthetic_raw_results"])
            self.assertIsNone(raw_payload["output_files"]["real_kg_raw_results"])
            assert_reported_outputs_under(self, output_dir, raw_payload)
            for figure_name in (
                "synthetic_encoding_time.png",
                "synthetic_qubits.png",
                "synthetic_depth.png",
                "synthetic_total_time.png",
            ):
                self.assertIn(figure_name, str(raw_payload["output_files"]["figures"]))

            environment = json.loads(
                (output_dir / "environment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(environment["synthetic_sizes_used"], [6])
            self.assertEqual(environment["real_kg_files_used"], [])
            self.assertIn(
                "table7_synthetic_results.csv",
                str(environment["generated_tables"]),
            )
            for figure_name in (
                "synthetic_encoding_time.png",
                "synthetic_qubits.png",
                "synthetic_depth.png",
                "synthetic_total_time.png",
            ):
                self.assertIn(figure_name, str(environment["generated_plots"]))
            self.assertTrue(
                any(
                    item["section"] == "synthetic_observations"
                    and item["encoding"] == "combined"
                    and item["status"] == "skipped"
                    for item in environment["skipped_or_failed_experiments"]
                )
            )

            run_summary = (output_dir / "RUN_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("--include-synthetic", run_summary)
            self.assertIn("## Section 9.2 Synthetic Sizes", run_summary)
            self.assertIn("- 6", run_summary)
            self.assertIn("table7_synthetic_results.csv", run_summary)
            self.assertIn("synthetic_encoding_time.png", run_summary)
            self.assertIn("combined", run_summary)
            self.assertIn("not supported", run_summary)
            self.assertIn("do not show quantum advantage", run_summary)

    def test_include_real_writes_real_kg_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_chapter9_experiments.py"
        real_file = repo_root / "data" / "real_kgs" / "exampleV3.ttl"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "chapter9"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--shots",
                    "32",
                    "--repetitions",
                    "1",
                    "--include-real",
                    "--real-kg-files",
                    str(real_file),
                    "--max-real-triples",
                    "10",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Real KG software-level observation rows: 4", completed.stdout)
            table8_path = output_dir / "table8_real_kg_results.csv"
            table8_tex_path = output_dir / "table8_real_kg_results.tex"
            real_json_path = output_dir / "real_kg_raw_results.json"
            self.assertTrue(table8_path.exists())
            self.assertTrue(table8_tex_path.exists())
            self.assertTrue(real_json_path.exists())

            table8_text = table8_path.read_text(encoding="utf-8")
            self.assertEqual(
                table8_text.splitlines()[0],
                (
                    "dataset_name,triple_count,entity_count,predicate_count,encoding,"
                    "qubits,dimension,preprocessing_time_ms,encoding_time_ms,"
                    "circuit_construction_time_ms,transpilation_time_ms,"
                    "simulation_time_ms,total_time_ms,circuit_depth,gate_count,"
                    "transpiled_depth,transpiled_gate_count,status,notes"
                ),
            )
            for encoding in ("basis", "amplitude", "phase", "combined"):
                self.assertIn(f",{encoding},", table8_text)
            self.assertIn("exampleV3,10", table8_text)
            self.assertIn("combined,--,--", table8_text)
            self.assertIn("skipped", table8_text)
            self.assertIn("software-level observation", table8_text)
            self.assertIn("no quantum-advantage claim", table8_text)
            self.assertIn("deterministically truncated", table8_text)

            table8_tex = table8_tex_path.read_text(encoding="utf-8")
            self.assertIn(
                (
                    "Dataset & Triples & Entities & Predicates & Encoding & "
                    "Qubits & Enc. Time & Status"
                ),
                table8_tex,
            )
            self.assertIn("software-level observations", table8_tex)

            real_payload = json.loads(real_json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(real_payload["rows"]), 4)
            self.assertEqual(len(real_payload["raw_scaling_rows"]), 3)
            self.assertEqual(real_payload["settings"]["max_real_triples"], 10)

            raw_payload = json.loads(
                (output_dir / "chapter9_raw_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(raw_payload["table8_real_kg_results"]), 4)
            self.assertEqual(len(raw_payload["real_kg_observations"]["rows"]), 4)
            self.assertIsNone(raw_payload["output_files"]["table7_synthetic_results"])
            self.assertIsNotNone(raw_payload["output_files"]["table8_real_kg_results"])
            self.assertIsNone(raw_payload["output_files"]["synthetic_raw_results"])
            self.assertIsNotNone(raw_payload["output_files"]["real_kg_raw_results"])
            assert_reported_outputs_under(self, output_dir, raw_payload)

            environment = json.loads(
                (output_dir / "environment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(environment["synthetic_sizes_used"], [])
            self.assertEqual(environment["real_kg_files_used"], [str(real_file)])
            self.assertIn(
                "table8_real_kg_results.csv",
                str(environment["generated_tables"]),
            )
            self.assertTrue(
                any(
                    item["section"] == "real_kg_observations"
                    and item["encoding"] == "combined"
                    and item["status"] == "skipped"
                    for item in environment["skipped_or_failed_experiments"]
                )
            )

            run_summary = (output_dir / "RUN_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("--include-real", run_summary)
            self.assertIn("## Section 9.3 Real KG Files", run_summary)
            self.assertIn("exampleV3.ttl", run_summary)
            self.assertIn("table8_real_kg_results.csv", run_summary)
            self.assertIn("real_kg_raw_results.json", run_summary)
            self.assertIn("Combined real-KG scaling is not supported", run_summary)
            self.assertIn("do not show quantum advantage", run_summary)

    def test_missing_real_kg_file_is_skipped_with_clear_notes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_chapter9_experiments.py"
        missing_file = "data/real/does_not_exist.ttl"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "chapter9"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--shots",
                    "32",
                    "--repetitions",
                    "1",
                    "--include-real",
                    "--real-kg-files",
                    missing_file,
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            table8_text = (output_dir / "table8_real_kg_results.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("does_not_exist", table8_text)
            self.assertIn("skipped", table8_text)
            self.assertIn("Real KG file not found", table8_text)
            self.assertIn("no quantum-advantage claim", table8_text)

            real_payload = json.loads(
                (output_dir / "real_kg_raw_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(real_payload["rows"]), 4)
            self.assertEqual(real_payload["raw_scaling_rows"], [])
            self.assertTrue(
                all(row["status"] == "skipped" for row in real_payload["rows"])
            )

            run_summary = (output_dir / "RUN_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("does_not_exist", run_summary)
            self.assertIn("Real KG file not found", run_summary)


if __name__ == "__main__":
    unittest.main()
