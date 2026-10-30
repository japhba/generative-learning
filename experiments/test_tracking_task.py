from itertools import product
import unittest

from tracking_task import ACTIONS, HORIZON, Streams, grade, make_evaluation, target_action, transition


class TrackingTests(unittest.TestCase):
    def test_oracle_is_unique_optimal_and_never_overshoots(self):
        for u, e in product(range(-2, 3), range(-30, 31)):
            errors = [abs(transition(u, e, action)) for action in range(len(ACTIONS))]
            self.assertEqual(errors.index(min(errors)), target_action(u, e))
            self.assertLessEqual(abs(transition(u, e, target_action(u, e))), abs(e))

    def test_expert_stream_has_zero_error_and_exploration_excites_it(self):
        expert, exploratory = Streams(13, 8), Streams(13, 8)
        for _ in range(32):
            rows = expert.current()
            self.assertTrue(all(row['e'] == 0 for row in rows))
            expert.advance([row['target'] for row in rows])
        exploratory.advance([0] * 8)
        self.assertTrue(any(row['e'] != 0 for row in exploratory.current()))

    def test_fresh_episodes_and_independent_grading(self):
        dataset = make_evaluation(94110, 10)
        self.assertEqual(dataset, make_evaluation(94110, 10))
        for episodes in dataset.values():
            for episode in episodes:
                actions, error = [], episode['initial_error']
                for command in episode['commands']:
                    action = target_action(command, error)
                    actions.append(action)
                    error = transition(command, error, action)
                result = grade(episode, actions)
                self.assertTrue(result['stable_tracking'])
                self.assertEqual(result['final_error'], 0)
                self.assertTrue(all(result['optimal']))
        with self.assertRaises(AssertionError): grade(dataset['nominal'][0], [0])


if __name__ == '__main__': unittest.main()
