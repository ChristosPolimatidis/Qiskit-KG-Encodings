from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from src.running_example import RDF_TYPE, RDFS_SUBCLASS_OF
from src.tasks.combined_demo import (
    DEFAULT_CONFIDENCE_SCORES,
    run_combined_demo_task,
    run_confidence_weighted_combined_demo,
)


class CombinedDemoTaskTests(unittest.TestCase):
    def test_uniform_demo_records_per_triple_quantities(self) -> None:
        result = run_combined_demo_task(output_path=None)

        self.assertAlmostEqual(result.state_norm, 1.0)
        self.assertEqual(result.num_qubits, 3)
        self.assertEqual(result.nonzero_indices, [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(result.triple_encoding_details), 6)
        self.assertEqual(len(result.decoded_probabilities), 6)
        for detail in result.triple_encoding_details:
            self.assertIn("amplitude_magnitude", detail)
            self.assertIn("phase_angle", detail)
            self.assertIn("measurement_probability", detail)
            self.assertIn("triple", detail)

    def test_confidence_weighted_demo_changes_amplitudes(self) -> None:
        result = run_confidence_weighted_combined_demo(output_path=None)
        magnitudes = [
            detail["amplitude_magnitude"]
            for detail in result.triple_encoding_details
        ]

        self.assertEqual(result.amplitude_mode, "confidence")
        self.assertNotEqual(len(set(magnitudes)), 1)
        self.assertAlmostEqual(result.state_norm, 1.0)
        self.assertEqual(len(magnitudes), len(DEFAULT_CONFIDENCE_SCORES))

    def test_phase_composition_example_uses_two_hop_path(self) -> None:
        result = run_combined_demo_task(output_path=None)
        example = result.composed_phase_examples[0]

        self.assertEqual(len(example["path"]), 2)
        self.assertAlmostEqual(
            example["composed_phase"],
            (math.pi / 4) + (math.pi / 2),
        )
        self.assertIn(RDF_TYPE, example["phase_terms"])
        self.assertIn(RDFS_SUBCLASS_OF, example["phase_terms"])
        self.assertIn("not a complete implementation", example["interpretation"])

    def test_demo_saves_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "combined_demo.json"
            result = run_combined_demo_task(output_path=output_path)

            self.assertEqual(result.output_path, str(output_path))
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_name"], "combined_amplitude_phase_demo")
            self.assertEqual(len(payload["decoded_probabilities"]), 6)
            self.assertEqual(payload["nonzero_indices"], [0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
