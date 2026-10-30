"""Finite modular planning with exact supervision and multiple valid solutions.

The task distribution is uniform over the requested finite parameter grid,
conditioned on shortest distance and a fixed identity-based train/test split.
The canonical teacher chooses a shortest path with A/B/C lexical tie-breaking.
Rollout training can instead retain any verified trajectory up to ``max_steps``;
the verifier never replaces generated text with the canonical teacher.
"""

from collections import deque
from functools import lru_cache
from hashlib import blake2b
from itertools import product
from random import Random
import re


OPERATIONS = ("A", "B", "C")
DEFAULT_EVAL_STRATA = {"easy": (1, 2), "in_distribution": (3, 4), "long_horizon": (5, 6)}
DEFAULT_CHECKPOINTS = (0, 64, 128, 256, 512, 1024, 2048, 4096)
SPLIT_VERSION = "modular-planning-v1"
_TRACE = re.compile(r"[ABC][ \t]+(?:0|[1-9][0-9]*)(?:[ \t]*;[ \t]*[ABC][ \t]+(?:0|[1-9][0-9]*))*")


def transition(modulus: int, add: int, subtract: int, state: int, operation: str) -> int:
    """Interpret one legal operation; unknown operation identifiers fail loudly."""
    return {"A": (state + add) % modulus, "B": (state - subtract) % modulus, "C": (2 * state) % modulus}[operation]


@lru_cache(maxsize=None)
def _paths(modulus: int, add: int, subtract: int, start: int) -> dict:
    paths, queue = {start: ()}, deque([start])
    while queue:
        state = queue.popleft()
        for operation in OPERATIONS:
            following = transition(modulus, add, subtract, state, operation)
            if following not in paths:
                paths[following] = paths[state] + ((operation, following),)
                queue.append(following)
    return paths


def identity(modulus: int, add: int, subtract: int, start: int, target: int) -> str:
    return f"m{modulus}-a{add}-b{subtract}-s{start}-t{target}"


def split_for(problem_id: str) -> str:
    """An 80/20 split fixed across dataset seeds, horizons, and distance strata."""
    digest = blake2b(f"{SPLIT_VERSION}:{problem_id}".encode(), digest_size=8).digest()
    return "train" if int.from_bytes(digest, "big") % 10 < 8 else "test"


def make_problem(modulus: int, add: int, subtract: int, start: int, target: int, max_steps: int = 8) -> dict:
    if not (modulus >= 3 and 0 < add < modulus and 0 < subtract < modulus):
        raise ValueError("Use modulus >= 3 and increments strictly between zero and the modulus.")
    if not (0 <= start < modulus and 0 <= target < modulus and start != target):
        raise ValueError("Distinct start and target must be valid states.")
    distance = len(_paths(modulus, add, subtract, start)[target])
    if not distance <= max_steps:
        raise ValueError("The target cannot be reached within the allowed horizon.")
    problem_id = identity(modulus, add, subtract, start, target)
    return {"id": problem_id, "split": split_for(problem_id), "modulus": modulus, "add": add, "subtract": subtract,
            "start": start, "target": target, "max_steps": max_steps, "shortest_distance": distance}


def shortest_plan(problem: dict) -> tuple:
    return _paths(problem["modulus"], problem["add"], problem["subtract"], problem["start"])[problem["target"]]


def teacher_solution(problem: dict) -> str:
    return "; ".join(f"{operation} {state}" for operation, state in shortest_plan(problem))


def prompt(problem: dict) -> str:
    """Expose the whole transition law, but neither a solution nor its distance."""
    modulus = problem["modulus"]
    return (
        "Find a sequence of moves that reaches the target.\n"
        f"All arithmetic is modulo {modulus}; states must be integers from 0 to {modulus - 1}.\n"
        f"A: add {problem['add']}. B: subtract {problem['subtract']}. C: double.\n"
        f"Start: {problem['start']}. Target: {problem['target']}. Use 1 to {problem['max_steps']} moves.\n"
        "For every move, write its letter, a space, and the resulting state. Separate moves with '; '.\n"
        "Write only the moves and states, with no explanation.\nSolution:"
    )


def verify_solution(problem: dict, completion: str) -> dict:
    """Grade the entire decoded completion after the caller removes EOS tokens.

    Invalid model outputs are expected observations, so they return explicit
    rejection reasons. Invalid experiment configuration raises in make_problem.
    Whitespace around the whole answer and around separators is immaterial;
    trailing text, code fences, arbitrary state claims, and extra moves are not.
    """
    text = completion.strip()
    if _TRACE.fullmatch(text) is None:
        return {"correct": False, "reason": "format", "steps": None, "plan": []}
    plan = [(parts[0], int(parts[1])) for move in text.split(";") for parts in [move.split()]]
    if len(plan) > problem["max_steps"]:
        return {"correct": False, "reason": "too_many_steps", "steps": len(plan), "plan": plan}
    state = problem["start"]
    for operation, claimed_state in plan:
        state = transition(problem["modulus"], problem["add"], problem["subtract"], state, operation)
        if claimed_state != state:
            return {"correct": False, "reason": "wrong_transition", "steps": len(plan), "plan": plan}
    correct = state == problem["target"]
    return {"correct": correct, "reason": "accepted" if correct else "wrong_target", "steps": len(plan), "plan": plan}


def is_correct(problem: dict, completion: str) -> bool:
    return verify_solution(problem, completion)["correct"]


def make_dataset(seed: int, train_size: int = 4352, eval_per_stratum: int = 128, moduli: tuple = (29, 31),
                 increments: tuple = tuple(range(1, 8)), train_distances: tuple = (2, 3, 4),
                 eval_strata: dict | None = None, max_steps: int = 8) -> dict:
    """Sample without replacement from an enumerated, auditable finite universe.

    Training examples are a shuffled prefix, permitting nested sample budgets.
    Test strata sample independent held-out identities. Reusing a seed gives
    identical lists; changing the seed never moves a problem across the split.
    """
    if train_size < 1 or eval_per_stratum < 1:
        raise ValueError("Dataset sizes must be positive.")
    if len(set(moduli)) != len(moduli) or len(set(increments)) != len(increments):
        raise ValueError("The parameter grid cannot contain duplicate entries.")
    if not (moduli and increments and min(moduli) >= 3 and 0 < min(increments) <= max(increments) < min(moduli)):
        raise ValueError("Invalid modulus/increment grid.")
    strata = DEFAULT_EVAL_STRATA if eval_strata is None else eval_strata
    distances = [distance for values in strata.values() for distance in values]
    if len(distances) != len(set(distances)):
        raise ValueError("Evaluation distance strata must be disjoint.")
    if min((*train_distances, *distances)) < 1 or max((*train_distances, *distances)) > max_steps:
        raise ValueError("All requested distances must fit the horizon.")
    candidates = {"train": [], **{name: [] for name in strata}}
    for modulus, add, subtract in product(moduli, increments, increments):
        for start in range(modulus):
            for target, path in _paths(modulus, add, subtract, start).items():
                distance = len(path)
                if not 1 <= distance <= max_steps:
                    continue
                problem_id = identity(modulus, add, subtract, start, target)
                split = split_for(problem_id)
                if split == "train" and distance in train_distances:
                    group = "train"
                elif split == "test" and distance in distances:
                    group = next(name for name, values in strata.items() if distance in values)
                else:
                    continue
                candidates[group].append({"id": problem_id, "split": split, "modulus": modulus, "add": add,
                                          "subtract": subtract, "start": start, "target": target,
                                          "max_steps": max_steps, "shortest_distance": distance})
    rng = Random(seed)
    requested = {"train": train_size, **dict.fromkeys(strata, eval_per_stratum)}
    selected = {name: rng.sample(problems, requested[name]) for name, problems in candidates.items()}
    return {"train": selected["train"], "eval": {name: selected[name] for name in strata},
            "metadata": {"task": SPLIT_VERSION, "seed": seed, "moduli": list(moduli), "increments": list(increments),
                         "train_distances": list(train_distances), "eval_strata": strata, "max_steps": max_steps,
                         "candidate_counts": {name: len(problems) for name, problems in candidates.items()},
                         "teacher": "Lexically tie-broken shortest BFS path; generated paths need only be valid.",
                         "split": "Identity hash modulo 10: buckets 0..7 train, 8..9 test; fixed across seeds."}}
