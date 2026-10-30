import unittest
from itertools import product

from feedback_task import Streams, distance, group_split, next_state, optimal_actions, state_id
from training_task import _paths


class FeedbackTaskTests(unittest.TestCase):
    def test_all_optimal_actions_match_independent_forward_paths(self):
        for modulus, add, subtract in product((7, 11), (1, 2), (1, 2)):
            for start, target in product(range(modulus), repeat=2):
                if start == target: continue
                p = {"modulus": modulus, "add": add, "subtract": subtract, "target": target}
                length = distance(p, start)
                valid = optimal_actions(p, start)
                self.assertTrue(valid)
                for action in range(3):
                    following = next_state(p, start, action)
                    expected = len(_paths(modulus, add, subtract, following)[target]) + 1 == length
                    self.assertEqual(action in valid, expected)

    def test_split_covers_all_states_and_terminal_action_fails(self):
        p = {"modulus": 11, "add": 2, "subtract": 3, "target": 7, "start": 1}
        for state in range(11):
            self.assertEqual(group_split(p), group_split({**p, "start": state}))
            self.assertEqual(state_id(p, state), state_id({**p, "start": state}, state))
        with self.assertRaises(AssertionError): optimal_actions(p, 7)

    def test_streams_reset_and_are_reproducible(self):
        p = {"modulus": 7, "add": 1, "subtract": 1, "target": 2, "start": 1}
        streams = Streams([p], 13, 3, 2)
        streams.advance([0, 1, 2])
        self.assertEqual(streams.episodes_succeeded, 2)
        self.assertEqual(streams.states, [1, 0, 1])
        streams.advance([1, 1, 1])
        self.assertEqual(streams.states, [0, 1, 0])
        self.assertEqual(streams.episodes_started, 6)


if __name__ == "__main__": unittest.main()
