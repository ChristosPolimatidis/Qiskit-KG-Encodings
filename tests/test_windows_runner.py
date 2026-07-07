from __future__ import annotations

from pathlib import Path
import unittest


class WindowsRunnerTests(unittest.TestCase):
    def test_runner_profiles_and_labels_are_normal_experiment_profiles(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_experiments_windows.ps1"
        text = script.read_text(encoding="utf-8")

        self.assertIn('[ValidateSet("light", "medium", "hard")]', text)
        self.assertIn("run_manifest.json", text)
        self.assertIn("command_log.jsonl", text)
        self.assertIn("build_validation_report.py", text)
        self.assertNotIn("reviewer", text.lower())
        self.assertNotIn("vldb", text.lower())


if __name__ == "__main__":
    unittest.main()
