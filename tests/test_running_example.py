from __future__ import annotations

import math
import unittest

from src.running_example import (
    EX,
    PAPER_TRIPLE_INDICES,
    RDF_TYPE,
    RDFS_SUBCLASS_OF,
    get_predicate_phase_map,
    get_running_example_graph,
    get_running_example_indices,
    get_running_example_triples,
)


class RunningExampleTests(unittest.TestCase):
    def test_running_example_has_exactly_six_triples(self) -> None:
        triples = get_running_example_triples()
        graph = get_running_example_graph()

        self.assertEqual(len(triples), 6)
        self.assertEqual(len(graph), 6)

    def test_sequential_indices_are_deterministic(self) -> None:
        first_indices = get_running_example_indices(mode="sequential")
        second_indices = get_running_example_indices(mode="sequential")

        self.assertEqual(first_indices, second_indices)
        self.assertEqual(list(first_indices.values()), [0, 1, 2, 3, 4, 5])

    def test_paper_indices_match_paper_set(self) -> None:
        indices = get_running_example_indices(mode="paper")

        self.assertEqual(tuple(indices.values()), PAPER_TRIPLE_INDICES)
        self.assertEqual(set(indices.values()), {1, 17, 27, 37, 88, 124})

    def test_predicate_phase_assignments_are_deterministic(self) -> None:
        phase_map = get_predicate_phase_map()
        repeated_phase_map = get_predicate_phase_map()
        expected = {
            RDF_TYPE: math.pi / 4,
            RDFS_SUBCLASS_OF: math.pi / 2,
            f"{EX}livesAt": 3 * math.pi / 4,
            f"{EX}teaches": math.pi,
        }

        self.assertEqual(set(phase_map), set(expected))
        self.assertEqual(phase_map, repeated_phase_map)
        for predicate, expected_phase in expected.items():
            self.assertAlmostEqual(phase_map[predicate], expected_phase)


if __name__ == "__main__":
    unittest.main()
