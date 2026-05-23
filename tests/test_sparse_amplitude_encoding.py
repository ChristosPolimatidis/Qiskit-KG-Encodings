from __future__ import annotations

import math
import unittest

import numpy as np

from src.amplitude_encoding import (
    build_paper_amplitude_encoding,
    build_sparse_index_amplitude_encoding,
)
from src.running_example import get_running_example_indices, get_running_example_triples


class SparseIndexAmplitudeEncodingTests(unittest.TestCase):
    def test_sequential_sparse_amplitude_encoding(self) -> None:
        triples = get_running_example_triples()
        result = build_sparse_index_amplitude_encoding(
            triples,
            index_mode="sequential",
        )

        self.assertEqual(result.num_qubits, 3)
        self.assertEqual(result.dimension, 8)
        self.assertEqual(result.nonzero_indices, [0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(float(np.linalg.norm(result.statevector)), 1.0)
        self.assertEqual(set(result.measurement_probabilities), {
            "000",
            "001",
            "010",
            "011",
            "100",
            "101",
        })

    def test_paper_sparse_amplitude_encoding(self) -> None:
        result = build_paper_amplitude_encoding(get_running_example_triples())

        self.assertEqual(result.num_qubits, 8)
        self.assertEqual(result.dimension, 256)
        self.assertEqual(
            result.nonzero_indices,
            [1, 17, 27, 37, 88, 124],
        )
        self.assertEqual(
            set(result.nonzero_indices),
            set(get_running_example_indices(mode="paper").values()),
        )

    def test_weighted_sparse_amplitudes_are_normalized(self) -> None:
        weights = [2, 1, 3, 1, 1, 2]
        result = build_sparse_index_amplitude_encoding(
            get_running_example_triples(),
            index_mode="sequential",
            weights=weights,
        )
        norm = math.sqrt(sum(weight * weight for weight in weights))

        self.assertAlmostEqual(float(np.linalg.norm(result.statevector)), 1.0)
        for index, weight in enumerate(weights):
            self.assertAlmostEqual(result.statevector[index].real, weight / norm)
            self.assertAlmostEqual(result.normalized_amplitudes[index], weight / norm)


if __name__ == "__main__":
    unittest.main()
