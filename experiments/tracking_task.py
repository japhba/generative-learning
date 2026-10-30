"""A tracking task whose expert demonstrations stay on a lower-dimensional support."""

from random import Random

ACTIONS = tuple(range(-3, 4))
LETTERS = "ABCDEFG"
STRATA = ("nominal", "recovery", "large_shift")
HORIZON = 16


def target_action(command, error):
    return max(-3, min(3, command - error)) + 3


def transition(command, error, action):
    assert 0 <= action < 7
    return error + ACTIONS[action] - command


def prompt(row):
    return ("Control a tracker. Requested motion is u and current tracking error is e.\n"
            "After action a, the error becomes e + a - u. Choose a to make this error as close to zero as possible.\n"
            "Actions: A=-3, B=-2, C=-1, D=0, E=1, F=2, G=3.\n"
            f"u={row['u']}; e={row['e']}.\nChoose one action letter:\nAction:")


def make_evaluation(seed, per_stratum):
    rng = Random(seed)
    offsets = {"nominal": [0], "recovery": [-3, -2, -1, 1, 2, 3], "large_shift": [-8, -7, -6, -5, -4, 4, 5, 6, 7, 8]}
    return {stratum: [{"id": f"{seed}-{stratum}-{i}", "initial_error": rng.choice(offsets[stratum]),
                       "commands": [rng.randrange(-2, 3) for _ in range(HORIZON)]} for i in range(per_stratum)] for stratum in STRATA}


class Streams:
    def __init__(self, seed, lanes=128):
        self.rng, self.lanes = Random(seed), lanes
        self.errors, self.ages = [0] * lanes, [0] * lanes
        self.commands = [self.rng.randrange(-2, 3) for _ in range(lanes)]

    def current(self):
        return [{"u": command, "e": error, "target": target_action(command, error), "age": age}
                for command, error, age in zip(self.commands, self.errors, self.ages, strict=True)]

    def advance(self, actions):
        assert len(actions) == self.lanes
        for index, action in enumerate(actions):
            error = transition(self.commands[index], self.errors[index], action)
            self.ages[index] += 1
            if self.ages[index] == HORIZON:
                error = 0
                self.ages[index] = 0
            self.errors[index] = error
            self.commands[index] = self.rng.randrange(-2, 3)


def grade(episode, actions):
    assert len(actions) == HORIZON
    error, errors, correct = episode["initial_error"], [], []
    for command, action in zip(episode["commands"], actions, strict=True):
        correct.append(action == target_action(command, error))
        error = transition(command, error, action)
        errors.append(error)
    return {"errors": errors, "optimal": correct, "mean_absolute_error": sum(abs(e) for e in errors) / HORIZON,
            "stable_tracking": all(e == 0 for e in errors[-8:]), "final_error": abs(errors[-1])}
