"""Stepwise modular planning with whole goal-context train/test separation."""

from hashlib import blake2b
from itertools import product
from random import Random

from training_task import OPERATIONS, _paths, transition

VERSION = "modular-feedback-goal-split-v1"
STRATA = {"easy": (1, 2), "in_distribution": (3, 4), "long_horizon": (5, 6)}


def group_id(problem):
    return f"m{problem['modulus']}-a{problem['add']}-b{problem['subtract']}-g{problem['target']}"


def group_split(problem):
    value = int.from_bytes(blake2b(f"{VERSION}:{group_id(problem)}".encode(), digest_size=8).digest(), "big")
    return "train" if value % 10 < 8 else ("validation" if value % 10 == 8 else "test")


def state_id(problem, state):
    return f"{group_id(problem)}-s{state}"


def next_state(problem, state, action):
    return transition(problem["modulus"], problem["add"], problem["subtract"], state, OPERATIONS[action])


def distance(problem, state):
    return len(_paths(problem["modulus"], problem["add"], problem["subtract"], state)[problem["target"]])


def optimal_actions(problem, state):
    depth = distance(problem, state)
    assert depth > 0, "Terminal states must be reset before requesting an action."
    return [action for action in range(3) if distance(problem, next_state(problem, state, action)) == depth - 1]


def prompt(problem, state):
    return (f"Reach the goal using the fewest moves. All states and arithmetic are modulo {problem['modulus']}.\n"
            f"A adds {problem['add']}; B subtracts {problem['subtract']}; C doubles the state.\n"
            f"Current state: {state}. Goal: {problem['target']}.\nChoose the next move: A, B, or C.\nMove:")


def make_dataset(seed, eval_seed=94100, eval_per_stratum=256, evaluation_split="test"):
    assert evaluation_split in ("validation", "test")
    pools = {"train": [], **{name: [] for name in STRATA}}
    for modulus, add, subtract in product((29, 31), range(1, 8), range(1, 8)):
        for target in range(modulus):
            context = {"modulus": modulus, "add": add, "subtract": subtract, "target": target}
            split = group_split(context)
            for start in range(modulus):
                depth = distance(context, start)
                problem = {**context, "start": start, "distance": depth, "id": state_id(context, start)}
                if split == "train" and depth in (3, 4): pools["train"].append(problem)
                if split == evaluation_split:
                    for name, depths in STRATA.items():
                        if depth in depths: pools[name].append(problem)
    rng = Random(eval_seed)
    return {"seed": seed, "eval_seed": eval_seed, "evaluation_split": evaluation_split, "version": VERSION, "train": pools["train"],
            "eval": {name: rng.sample(pools[name], eval_per_stratum) for name in STRATA},
            "candidate_counts": {name: len(pool) for name, pool in pools.items()},
            "split": "All current states with the same modulus, increments and goal share one split; no training trajectory can enter an evaluation goal context."}


class Streams:
    """Parallel trajectories reset on success or after a fixed horizon."""
    def __init__(self, problems, seed, lanes, horizon):
        assert lanes > 0 and horizon > 0
        self.pool, self.rng, self.horizon = problems, Random(seed), horizon
        self.problems = [self.rng.choice(problems) for _ in range(lanes)]
        self.states = [p["start"] for p in self.problems]
        self.ages = [0] * lanes
        self.episodes_started, self.episodes_succeeded = lanes, 0

    def advance(self, actions):
        assert len(actions) == len(self.states)
        for index, action in enumerate(actions):
            problem = self.problems[index]
            state = next_state(problem, self.states[index], action)
            self.states[index] = state
            self.ages[index] += 1
            success = state == problem["target"]
            if success or self.ages[index] == self.horizon:
                self.episodes_succeeded += int(success)
                self.episodes_started += 1
                self.problems[index] = self.rng.choice(self.pool)
                self.states[index] = self.problems[index]["start"]
                self.ages[index] = 0
