from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.run_classical_baselines import main
from src.classical_baselines import run_all_classical_baselines


class ClassicalBaselineTests(unittest.TestCase):
    def test_classical_baselines_produce_expected_tasks(self) -> None:
        results = run_all_classical_baselines()
        tasks = {result.task for result in results}
        self.assertIn("search_grover_lookup", tasks)
        self.assertIn("entity_matching_swap_test", tasks)
        self.assertIn("link_prediction_distance_estimation", tasks)
        self.assertIn("keyword_search_swap_test", tasks)
        for result in results:
            self.assertIn("not establish quantum advantage", result.claim_scope)

    def test_script_writes_baseline_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "baselines"
            payload = main(["--output-dir", str(output_dir)])

            self.assertTrue((output_dir / "classical_baselines_raw.csv").exists())
            self.assertTrue((output_dir / "classical_baselines_summary.csv").exists())
            self.assertTrue((output_dir / "classical_baselines.json").exists())
            self.assertEqual(len(payload["raw_rows"]), 4)
            self.assertIn("not establish quantum advantage", payload["claim_scope"])


if __name__ == "__main__":
    unittest.main()
