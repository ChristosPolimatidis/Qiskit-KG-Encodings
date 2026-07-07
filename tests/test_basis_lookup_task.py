from __future__ import annotations

import unittest

from src.running_example import get_running_example_triples
from src.tasks.basis_lookup import (
    recommended_grover_iterations,
    run_basis_lookup_task,
)


class BasisLookupTaskTests(unittest.TestCase):
    def test_decoded_result_is_target_triple(self) -> None:
        target_triple = get_running_example_triples()[-1]

        result = run_basis_lookup_task(
            target_triple=target_triple,
            shots=512,
            seed_simulator=12345,
        )

        self.assertEqual(result.decoded_result, target_triple.to_dict())
        self.assertEqual(result.target_triple, target_triple.to_dict())
        self.assertGreater(result.success_probability, 0.8)
        self.assertEqual(result.num_qubits, 3)
        self.assertEqual(result.search_space_size, 8)
        self.assertEqual(result.marked_count, 1)
        self.assertEqual(result.recommended_grover_iterations, 2)
        self.assertEqual(result.grover_iterations, 2)
        self.assertEqual(result.pass_fail, "pass")
        self.assertIn(result.target_bitstring, result.probabilities)

    def test_decoded_result_is_target_index(self) -> None:
        target_triple = get_running_example_triples()[2]

        result = run_basis_lookup_task(
            target_index=2,
            shots=512,
            seed_simulator=12345,
        )

        self.assertEqual(result.decoded_result, target_triple.to_dict())
        self.assertEqual(result.decoded_index, 2)

    def test_recommended_grover_iterations_are_computed(self) -> None:
        self.assertEqual(recommended_grover_iterations(8, 1), 2)
        self.assertEqual(recommended_grover_iterations(16, 1), 3)
        self.assertEqual(recommended_grover_iterations(8, 8), 0)
        with self.assertRaises(ValueError):
            recommended_grover_iterations(0, 1)
        with self.assertRaises(ValueError):
            recommended_grover_iterations(8, 0)
        with self.assertRaises(ValueError):
            recommended_grover_iterations(4, 5)


if __name__ == "__main__":
    unittest.main()
