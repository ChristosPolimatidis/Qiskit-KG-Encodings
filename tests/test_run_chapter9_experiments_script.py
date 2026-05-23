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
            self.assertTrue((output_dir / "table3_encoding_process.tex").exists())
            self.assertTrue((output_dir / "table4_usage_tasks.tex").exists())
            self.assertTrue((output_dir / "chapter9_raw_results.json").exists())
            self.assertTrue((output_dir / "environment.json").exists())
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
            table3_tex = (output_dir / "table3_encoding_process.tex").read_text(
                encoding="utf-8"
            )
            table4_tex = (output_dir / "table4_usage_tasks.tex").read_text(
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

            payload = json.loads(
                (output_dir / "chapter9_raw_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["running_example_triple_count"], 6)
            self.assertEqual(len(payload["encoding_process_rows"]), 6)
            self.assertEqual(len(payload["usage_task_rows"]), 5)
            self.assertEqual(len(payload["additional_validation_rows"]), 1)
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
            ):
                self.assertIn(field, environment)
            self.assertEqual(environment["random_seed"], 12345)
            self.assertEqual(environment["command_line_arguments"]["shots"], 64)
            self.assertIn("logical_cpus", environment["cpu"])

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


if __name__ == "__main__":
    unittest.main()
