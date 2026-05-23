from __future__ import annotations

import unittest

from src.running_example import EX, RDF_TYPE, get_running_example_triples
from src.tasks.phase_filtering import (
    decode_top_results,
    run_phase_filtering_task,
    select_marked_indices,
)


class PhaseFilteringTaskTests(unittest.TestCase):
    def test_predicate_marking_logic_selects_expected_indices(self) -> None:
        self.assertEqual(select_marked_indices(predicate_uri=f"{EX}teaches"), [5])
        self.assertEqual(select_marked_indices(predicate_uri=RDF_TYPE), [0, 3])
        self.assertEqual(
            select_marked_indices(
                predicate_uri=f"{EX}teaches",
                index_mode="paper",
            ),
            [124],
        )
        self.assertEqual(
            select_marked_indices(predicate_uri=RDF_TYPE, index_mode="paper"),
            [1, 37],
        )

    def test_rule_marking_logic_selects_expected_indices(self) -> None:
        marked_indices = select_marked_indices(
            predicate_uri=None,
            rule_fn=lambda triple: triple.subject == f"{EX}Aristotle",
        )

        self.assertEqual(marked_indices, [0, 2, 5])

    def test_decode_top_results_maps_indices_to_triples(self) -> None:
        triples = get_running_example_triples()
        decoded = decode_top_results(
            {"101": 90, "111": 10},
            marked_indices=[5],
            shots=100,
            limit=2,
        )

        self.assertEqual(decoded[0]["index"], 5)
        self.assertEqual(decoded[0]["triple"], triples[5].to_dict())
        self.assertTrue(decoded[0]["marked"])
        self.assertIsNone(decoded[1]["triple"])

    def test_decode_top_results_supports_paper_indices(self) -> None:
        triples = get_running_example_triples()
        decoded = decode_top_results(
            {"01111100": 90, "00000101": 10},
            marked_indices=[124],
            shots=100,
            limit=2,
            index_mode="paper",
        )

        self.assertEqual(decoded[0]["index"], 124)
        self.assertEqual(decoded[0]["triple"], triples[5].to_dict())
        self.assertTrue(decoded[0]["marked"])
        self.assertIsNone(decoded[1]["triple"])

    def test_phase_filtering_top_result_is_marked_triple(self) -> None:
        result = run_phase_filtering_task(
            predicate_uri=f"{EX}teaches",
            shots=512,
            seed_simulator=13579,
        )

        self.assertEqual(result.decoded_top_results[0]["index"], 5)
        self.assertTrue(result.decoded_top_results[0]["marked"])
        self.assertGreater(
            result.marked_probability_after,
            result.marked_probability_before,
        )

    def test_phase_filtering_can_use_paper_index_mode(self) -> None:
        result = run_phase_filtering_task(
            predicate_uri=f"{EX}teaches",
            index_mode="paper",
            shots=128,
            seed_simulator=13579,
        )

        self.assertEqual(result.index_mode, "paper")
        self.assertEqual(result.num_qubits, 8)
        self.assertEqual(result.marked_indices, [124])
        self.assertAlmostEqual(result.marked_probability_before, 1 / 256)


if __name__ == "__main__":
    unittest.main()
