from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_validation_report import main


class ValidationReportBuilderTests(unittest.TestCase):
    def test_builder_indexes_artifacts_and_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            reports_dir = run_dir / "reports"
            figures_dir = run_dir / "figures"
            reports_dir.mkdir(parents=True)
            figures_dir.mkdir(parents=True)
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"profile": "light", "status": "SUCCESS"}),
                encoding="utf-8",
            )
            (run_dir / "command_log.jsonl").write_text(
                json.dumps({"event": "finish", "name": "pytest", "exit_code": 0}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "chapter9_validation_thresholds.csv").write_text(
                (
                    "task,repetition,expected_value,estimated_value,absolute_error,"
                    "relative_error,pass_threshold,pass_fail,threshold_reason\n"
                    "search_grover_lookup,1,1,0.95,0.05,0.05,0.75,pass,ok\n"
                ),
                encoding="utf-8",
            )
            (figures_dir / "hist_basis_lookup.png").write_bytes(b"png")

            payload = main(["--run-dir", str(run_dir)])

            self.assertEqual(payload["run_profile"], "light")
            self.assertEqual(payload["threshold_pass_count"], 1)
            self.assertTrue((reports_dir / "validation_summary.md").exists())
            self.assertTrue((reports_dir / "validation_summary.json").exists())
            self.assertTrue((reports_dir / "artifact_index.csv").exists())
            summary = (reports_dir / "validation_summary.md").read_text(encoding="utf-8")
            self.assertIn("not a quantum-advantage claim", summary)
            self.assertIn("hist_basis_lookup.png", (reports_dir / "artifact_index.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
