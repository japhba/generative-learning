import unittest

from tracking_exact import OFFSETS, exact_metrics, oracle_policy


class ExactTrackingTests(unittest.TestCase):
    def test_oracle_matches_independently_derived_recovery_cost(self):
        result=exact_metrics(oracle_policy(),OFFSETS['recovery'])
        # Error1 corrects immediately; error2 has total expected error .2;
        # error3 has .6 on the first step plus .2*.2 subsequently.
        self.assertAlmostEqual(result['mean_absolute_error'],(0+.2+.64)/3/16)
        self.assertAlmostEqual(result['stable_tracking'],1.)
        self.assertAlmostEqual(result['action_accuracy'],1.)

    def test_feedforward_controller_preserves_initial_error(self):
        policy={(u,e):u for u in range(-2,3) for e in range(-88,89)}
        result=exact_metrics(policy,OFFSETS['recovery'])
        self.assertAlmostEqual(result['mean_absolute_error'],2.)
        self.assertAlmostEqual(result['stable_tracking'],0.)
        self.assertAlmostEqual(exact_metrics(policy,[0])['mean_absolute_error'],0.)

    def test_zero_action_expected_final_squared_effect_is_symmetric(self):
        policy={(u,e):0 for u in range(-2,3) for e in range(-88,89)}
        positive=exact_metrics(policy,[3])
        negative=exact_metrics(policy,[-3])
        for field in positive: self.assertAlmostEqual(positive[field],negative[field])


if __name__=='__main__': unittest.main()
