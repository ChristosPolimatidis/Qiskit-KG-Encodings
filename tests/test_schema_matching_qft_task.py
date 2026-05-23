from __future__ import annotations

import unittest

from src.tasks.schema_matching_qft import run_schema_matching_qft_task


class SchemaMatchingQFTTaskTests(unittest.TestCase):
    def test_qft_schema_pattern_validation_reports_similarity(self) -> None:
        result = run_schema_matching_qft_task()

        self.assertEqual(result.task_name, "schema_matching_qft")
        self.assertEqual(len(result.phase_pattern_a), 4)
        self.assertEqual(len(result.phase_pattern_b), 4)
        self.assertEqual(len(result.fourier_magnitudes_a), 4)
        self.assertEqual(len(result.fourier_magnitudes_b), 4)
        self.assertGreaterEqual(result.fourier_pattern_similarity, 0.0)
        self.assertLessEqual(result.fourier_pattern_similarity, 1.0)
        self.assertEqual(result.num_qubits, 4)
        self.assertGreater(result.circuit_depth, 0)
        self.assertIn("not full schema matching", result.claim_note)


if __name__ == "__main__":
    unittest.main()
