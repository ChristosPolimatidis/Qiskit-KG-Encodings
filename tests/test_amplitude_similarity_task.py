from __future__ import annotations

import unittest

from src.tasks.amplitude_similarity import run_amplitude_similarity_task


class AmplitudeSimilarityTaskTests(unittest.TestCase):
    def test_swap_test_estimate_is_close_to_classical_similarity(self) -> None:
        result = run_amplitude_similarity_task(
            shots=4096,
            seed_simulator=24680,
        )

        self.assertLessEqual(result.absolute_error, 0.08)
        self.assertAlmostEqual(
            result.estimated_similarity,
            result.classical_similarity,
            delta=0.08,
        )
        self.assertGreaterEqual(result.p0, 0.0)
        self.assertLessEqual(result.p0, 1.0)

    def test_task_records_expected_swap_test_metrics(self) -> None:
        result = run_amplitude_similarity_task(
            shots=1024,
            seed_simulator=24680,
        )

        self.assertEqual(result.num_qubits, 5)
        self.assertEqual(result.shots, 1024)
        self.assertGreater(result.circuit_depth, 0)
        self.assertGreater(result.gate_count, 0)
        self.assertGreater(result.transpiled_depth, 0)
        self.assertGreater(result.transpiled_gate_count, 0)


if __name__ == "__main__":
    unittest.main()
