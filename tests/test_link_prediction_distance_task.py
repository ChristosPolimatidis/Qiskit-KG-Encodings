from __future__ import annotations

import unittest

from src.tasks.link_prediction_distance import run_link_prediction_distance_task


class LinkPredictionDistanceTaskTests(unittest.TestCase):
    def test_transe_style_distance_estimation_reports_expected_metrics(self) -> None:
        result = run_link_prediction_distance_task(
            shots=512,
            seed_simulator=11223,
        )

        self.assertEqual(result.task_name, "link_prediction_distance_estimation")
        self.assertGreaterEqual(result.classical_distance, 0.0)
        self.assertGreaterEqual(result.estimated_similarity, 0.0)
        self.assertLessEqual(result.estimated_similarity, 1.0)
        self.assertGreaterEqual(result.absolute_error, 0.0)
        self.assertEqual(result.num_qubits, 3)
        self.assertGreater(result.circuit_depth, 0)
        self.assertGreater(result.gate_count, 0)
        self.assertIn("not an HHL implementation", result.claim_note)


if __name__ == "__main__":
    unittest.main()
