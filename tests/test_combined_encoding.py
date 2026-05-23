from __future__ import annotations

import math
import unittest

import numpy as np

from src.combined_encoding import combined_amplitude_phase_encoding
from src.running_example import (
    RDF_TYPE,
    RDFS_SUBCLASS_OF,
    get_running_example_indices,
    get_running_example_triples,
)


class CombinedEncodingTests(unittest.TestCase):
    def test_state_norm_is_one(self) -> None:
        result = combined_amplitude_phase_encoding()

        self.assertAlmostEqual(float(np.linalg.norm(result.statevector)), 1.0)

    def test_only_existing_triple_indices_have_nonzero_amplitude(self) -> None:
        result = combined_amplitude_phase_encoding(index_mode="paper")
        expected_indices = set(get_running_example_indices(mode="paper").values())
        actual_indices = set(np.nonzero(np.abs(result.statevector) > 1e-12)[0])

        self.assertEqual(actual_indices, expected_indices)
        self.assertEqual(set(result.nonzero_indices), expected_indices)

    def test_same_predicate_triples_have_same_phase(self) -> None:
        result = combined_amplitude_phase_encoding(index_mode="sequential")
        indices = get_running_example_indices(mode="sequential")
        triples = get_running_example_triples()

        for predicate in (RDF_TYPE, RDFS_SUBCLASS_OF):
            predicate_indices = [
                indices[triple] for triple in triples if triple.predicate == predicate
            ]
            self.assertEqual(len(predicate_indices), 2)
            self.assertAlmostEqual(
                result.index_phase_map[predicate_indices[0]],
                result.index_phase_map[predicate_indices[1]],
            )

    def test_complex_coefficients_include_amplitudes_and_phases(self) -> None:
        weights = [2, 1, 3, 1, 1, 2]
        result = combined_amplitude_phase_encoding(weights=weights)
        norm = math.sqrt(sum(weight * weight for weight in weights))

        for index, expected_weight in enumerate(weights):
            coefficient = result.statevector[index]
            self.assertAlmostEqual(abs(coefficient), expected_weight / norm)
            self.assertAlmostEqual(abs(coefficient), result.amplitude_map[index])
            self.assertAlmostEqual(np.angle(coefficient), result.index_phase_map[index])

        self.assertTrue(
            any(
                abs(result.statevector[index].real) > 1e-12
                and abs(result.statevector[index].imag) > 1e-12
                for index in result.nonzero_indices
            )
        )

    def test_paper_mode_uses_eight_qubits(self) -> None:
        result = combined_amplitude_phase_encoding(index_mode="paper")

        self.assertEqual(result.num_qubits, 8)
        self.assertEqual(result.dimension, 256)


if __name__ == "__main__":
    unittest.main()
