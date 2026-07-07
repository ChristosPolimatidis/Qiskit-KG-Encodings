from __future__ import annotations

import unittest

import numpy as np

from src.tasks.keyword_search import (
    ENTITY_FEATURE_VECTORS,
    QUERY_VECTOR,
    classical_keyword_scores,
    normalize_feature_vector,
    run_keyword_search_task,
)


class KeywordSearchTaskTests(unittest.TestCase):
    def test_vectors_are_normalized(self) -> None:
        normalized_query = normalize_feature_vector(QUERY_VECTOR)
        self.assertTrue(np.isclose(np.linalg.norm(normalized_query), 1.0))

        for vector in ENTITY_FEATURE_VECTORS.values():
            normalized_entity = normalize_feature_vector(vector)
            self.assertTrue(np.isclose(np.linalg.norm(normalized_entity), 1.0))

    def test_classical_scores_match_paper_example(self) -> None:
        scores = classical_keyword_scores()
        self.assertAlmostEqual(scores["Aristotle"], 0.5)
        self.assertAlmostEqual(scores["Athens"], 0.25)

    def test_keyword_search_ranks_aristotle_above_athens(self) -> None:
        result = run_keyword_search_task(shots=128, seed_simulator=12345)
        self.assertEqual(result.top_result, "Aristotle")
        self.assertEqual(result.ranking[0], "Aristotle")
        self.assertLess(
            result.ranking.index("Aristotle"),
            result.ranking.index("Athens"),
        )

    def test_task_returns_required_metadata_fields(self) -> None:
        result = run_keyword_search_task(shots=128, seed_simulator=12345)

        self.assertEqual(result.task, "Keyword Search")
        self.assertEqual(result.encoding, "Amplitude")
        self.assertEqual(result.method, "Swap Test")
        self.assertEqual(
            result.feature_space,
            ["Aristotle", "Athens", "Person", "City", "Philosophy"],
        )
        self.assertEqual(result.query_terms, ["Aristotle", "Athens"])
        self.assertEqual(result.candidate_entities, ["Aristotle", "Athens"])
        self.assertIn("Aristotle", result.classical_scores)
        self.assertIn("Athens", result.estimated_scores)
        self.assertIn("Aristotle", result.absolute_errors)
        self.assertEqual(result.shots, 128)
        self.assertEqual(result.repetitions, 1)
        self.assertGreater(result.num_qubits, 0)
        self.assertGreaterEqual(result.circuit_depth, 0)
        self.assertGreaterEqual(result.gate_count, 0)
        self.assertGreaterEqual(result.transpiled_depth, 0)
        self.assertGreaterEqual(result.transpiled_gate_count, 0)
        self.assertGreater(result.task_time_seconds, 0.0)
        self.assertIn("not a full RDF keyword search engine", result.claim_note)


if __name__ == "__main__":
    unittest.main()
