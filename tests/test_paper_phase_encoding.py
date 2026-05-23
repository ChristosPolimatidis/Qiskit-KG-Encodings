from __future__ import annotations

import math
import unittest

import numpy as np

from src.paper_phase_encoding import predicate_phase, presence_phase
from src.running_example import (
    RDF_TYPE,
    RDFS_SUBCLASS_OF,
    get_running_example_indices,
    get_running_example_triples,
)


class PaperPhaseEncodingTests(unittest.TestCase):
    def test_presence_phase_sign_flips_kg_triple_indices(self) -> None:
        result = presence_phase(index_mode="paper", shots=256)
        expected_magnitude = 1 / math.sqrt(256)

        for index in get_running_example_indices(mode="paper").values():
            self.assertAlmostEqual(result.statevector[index].real, -expected_magnitude)
            self.assertAlmostEqual(result.statevector[index].imag, 0.0)

    def test_presence_phase_keeps_non_triple_indices_positive_in_paper_mode(self) -> None:
        result = presence_phase(index_mode="paper", shots=256)
        triple_indices = set(get_running_example_indices(mode="paper").values())
        expected_magnitude = 1 / math.sqrt(256)

        checked = 0
        for index, amplitude in enumerate(result.statevector):
            if index in triple_indices:
                continue
            self.assertAlmostEqual(amplitude.real, expected_magnitude)
            self.assertAlmostEqual(amplitude.imag, 0.0)
            checked += 1
            if checked == 12:
                break

        self.assertEqual(checked, 12)

    def test_predicate_phase_matches_equal_predicates(self) -> None:
        result = predicate_phase(index_mode="sequential", support_mode="triples")
        indices = get_running_example_indices(mode="sequential")
        triples = get_running_example_triples()

        rdf_type_indices = [
            indices[triple] for triple in triples if triple.predicate == RDF_TYPE
        ]
        subclass_indices = [
            indices[triple]
            for triple in triples
            if triple.predicate == RDFS_SUBCLASS_OF
        ]

        self.assertEqual(len(rdf_type_indices), 2)
        self.assertEqual(len(subclass_indices), 2)
        self.assertAlmostEqual(
            result.index_phase_map[rdf_type_indices[0]],
            result.index_phase_map[rdf_type_indices[1]],
        )
        self.assertAlmostEqual(
            result.index_phase_map[subclass_indices[0]],
            result.index_phase_map[subclass_indices[1]],
        )
        self.assertAlmostEqual(
            np.angle(result.statevector[rdf_type_indices[0]]),
            np.angle(result.statevector[rdf_type_indices[1]]),
        )
        self.assertAlmostEqual(
            np.angle(result.statevector[subclass_indices[0]]),
            np.angle(result.statevector[subclass_indices[1]]),
        )

    def test_phase_encoding_preserves_probabilities_before_interference(self) -> None:
        results = [
            presence_phase(index_mode="paper", shots=256),
            predicate_phase(index_mode="paper", support_mode="triples", shots=256),
            predicate_phase(index_mode="paper", support_mode="full", shots=256),
        ]

        for result in results:
            with self.subTest(phase_mode=result.phase_mode, support_mode=result.support_mode):
                np.testing.assert_allclose(
                    np.abs(result.statevector) ** 2,
                    np.abs(result.initial_statevector) ** 2,
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
