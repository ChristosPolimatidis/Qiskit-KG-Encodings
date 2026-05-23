from __future__ import annotations

import math
import unittest

from src.tasks.multihop_phase_kickback import run_multihop_phase_kickback_task


class MultiHopPhaseKickbackTaskTests(unittest.TestCase):
    def test_phase_kickback_accumulates_two_hop_phase(self) -> None:
        result = run_multihop_phase_kickback_task()

        self.assertEqual(result.task_name, "multihop_phase_kickback")
        self.assertEqual(len(result.path), 2)
        self.assertAlmostEqual(
            result.expected_composed_phase,
            (math.pi / 4) + (math.pi / 2),
        )
        self.assertAlmostEqual(result.observed_composed_phase, result.expected_composed_phase)
        self.assertAlmostEqual(result.phase_error, 0.0)
        self.assertEqual(result.num_qubits, 2)
        self.assertGreater(result.circuit_depth, 0)
        self.assertIn("not a full RDFS reasoner", result.claim_note)


if __name__ == "__main__":
    unittest.main()
