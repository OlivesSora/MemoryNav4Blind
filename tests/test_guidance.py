import unittest

from memory_nav.recording.anchor_collector import AnchorCandidate
from memory_nav.replay.guidance import GuidanceController


class GuidanceTests(unittest.TestCase):
    def test_anchor_advance_and_arrival_once(self):
        anchor = AnchorCandidate("turn", 10, 10.0, "turn", "前方左转", True)
        guidance = GuidanceController([anchor], advance_distance_m=8, arrival_distance_m=2, cooldown_s=0)
        first = guidance.update(3, "normal", 0, False, 1)
        self.assertEqual([prompt.key for prompt in first.prompts], ["anchor:turn:advance"])
        second = guidance.update(8.5, "normal", 0, False, 2)
        self.assertEqual([prompt.key for prompt in second.prompts], ["anchor:turn:arrival"])
        third = guidance.update(10, "normal", 0, False, 3)
        self.assertEqual(third.prompts, ())

    def test_unconfirmed_anchor_is_not_announced(self):
        anchor = AnchorCandidate("draft", 1, 1, "turn", "左转", False)
        guidance = GuidanceController([anchor])
        self.assertEqual(guidance.update(0, "normal", 0, False, 1).prompts, ())

    def test_deviation_pauses_anchor_and_recovery_announces(self):
        anchor = AnchorCandidate("a", 1, 5, "turn", "左转", True)
        guidance = GuidanceController([anchor], cooldown_s=0)
        deviated = guidance.update(1, "deviated", 4, True, 1)
        self.assertEqual([prompt.key for prompt in deviated.prompts], ["state:deviated"])
        recovered = guidance.update(1, "recovering", 1, False, 2)
        self.assertEqual([prompt.key for prompt in recovered.prompts], ["state:recovered", "anchor:a:advance"])


if __name__ == "__main__":
    unittest.main()
