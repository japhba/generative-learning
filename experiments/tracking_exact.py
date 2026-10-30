"""Exact finite-horizon evaluation of a deterministic tracking-policy lookup table."""

from collections import defaultdict
import numpy as np

OFFSETS = {'nominal':[0], 'recovery':[-3,-2,-1,1,2,3], 'large_shift':[-8,-7,-6,-5,-4,4,5,6,7,8]}


def exact_metrics(policy, initial_errors, horizon=16):
    distribution = {(error,True):1/len(initial_errors) for error in initial_errors}
    error_sum,optimal_sum = 0.,0.
    for step in range(horizon):
        following = defaultdict(float)
        for (error,stable),mass in distribution.items():
            for command in range(-2,3):
                action = policy[command,error]
                assert -3 <= action <= 3
                correct = max(-3,min(3,command-error))
                next_error = error+action-command
                remains = stable and (step < horizon-8 or next_error == 0)
                following[next_error,remains] += mass/5
                error_sum += mass/5*abs(next_error)
                optimal_sum += mass/5*(action==correct)
        distribution = following
        assert np.isclose(sum(distribution.values()),1.)
    return {'mean_absolute_error':error_sum/horizon,'action_accuracy':optimal_sum/horizon,
            'stable_tracking':sum(mass for (error,stable),mass in distribution.items() if stable),
            'final_error':sum(abs(error)*mass for (error,stable),mass in distribution.items())}


def oracle_policy():
    return {(u,e):max(-3,min(3,u-e)) for u in range(-2,3) for e in range(-88,89)}
