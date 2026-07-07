from __future__ import annotations

from pathlib import Path
import unittest


class WindowsRunnerTests(unittest.TestCase):
    def test_runner_profiles_and_labels_are_normal_experiment_profiles(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_experiments_windows.ps1"
        text = script.read_text(encoding="utf-8")

        self.assertIn('[ValidateSet("light", "medium", "hard")]', text)
        self.assertIn("[switch]$InstallDeps", text)
        self.assertIn("function Run-Step", text)
        self.assertIn("run_config.json", text)
        self.assertIn("run_manifest.json", text)
        self.assertIn("command_log.jsonl", text)
        self.assertIn("run_experiment_suite.py", text)
        self.assertIn("collect_run_outputs.py", text)
        self.assertIn("Compress-Archive", text)
        self.assertNotIn("reviewer", text.lower())
        self.assertNotIn("paper", text.lower())

    def test_runner_creates_requested_output_folder_names(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_experiments_windows.ps1"
        text = script.read_text(encoding="utf-8")

        for folder_name in (
            "logs",
            "tables",
            "plots",
            "json",
            "raw",
            "summary",
            "circuits",
            "histograms",
        ):
            self.assertIn(f'"{folder_name}"', text)


if __name__ == "__main__":
    unittest.main()
