from __future__ import annotations

import unittest

from src.tasks.schema_matching_qft import run_schema_matching_qft_task


class SchemaMatchingQFTTaskTests(unittest.TestCase):
    def test_qft_schema_pattern_validation_reports_similarity(self) -> None:
        result = run_schema_matching_qft_task(shots=1024, seed_simulator=12345)

        self.assertEqual(result.task_name, "schema_matching_qft")
        self.assertEqual(result.phase_assignment_strategy, "synonym_dictionary")
        self.assertEqual(len(result.phase_pattern_a), 4)
        self.assertEqual(len(result.phase_pattern_b), 4)
        self.assertEqual(len(result.fourier_magnitudes_a), 4)
        self.assertEqual(len(result.fourier_magnitudes_b), 4)
        self.assertEqual(sum(result.counts_pattern_a.values()), 1024)
        self.assertEqual(sum(result.counts_pattern_b.values()), 1024)
        self.assertIn("exact_uri", result.strategy_results)
        self.assertIn("synonym_dictionary", result.strategy_results)
        self.assertIn("negative_control", result.strategy_results)
        self.assertGreaterEqual(result.fourier_pattern_similarity, 0.0)
        self.assertLessEqual(result.fourier_pattern_similarity, 1.0)
        self.assertLess(
            result.negative_control_similarity,
            result.measured_distribution_similarity,
        )
        self.assertEqual(result.pass_fail, "pass")
        self.assertEqual(result.num_qubits, 4)
        self.assertGreater(result.circuit_depth, 0)
        self.assertGreater(result.transpiled_depth, 0)
        self.assertIn("not full schema matching", result.claim_note)


if __name__ == "__main__":
    unittest.main()
