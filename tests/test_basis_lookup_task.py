from __future__ import annotations

import unittest

from src.running_example import get_running_example_triples
from src.tasks.basis_lookup import run_basis_lookup_task


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

    def test_decoded_result_is_target_index(self) -> None:
        target_triple = get_running_example_triples()[2]

        result = run_basis_lookup_task(
            target_index=2,
            shots=512,
            seed_simulator=12345,
        )

        self.assertEqual(result.decoded_result, target_triple.to_dict())
        self.assertEqual(result.decoded_index, 2)


if __name__ == "__main__":
    unittest.main()
