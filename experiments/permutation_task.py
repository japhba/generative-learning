"""Fresh permutation worlds with exact labels and a Bayesian reference learner."""

from itertools import permutations, product
from random import Random

STATES = "ABCD"
OPERATORS = ("f", "g")
PERMUTATIONS = tuple(permutations(STATES))


def _answer(mappings: dict, start: str, ops: list[str]) -> str:
    for op in ops: start = mappings[op][start]
    return start


def _query(mappings: dict, start: str, ops: list[str]) -> dict:
    return {"start": start, "ops": ops, "answer": _answer(mappings, start, ops)}


def make_world(seed: int, tests_per_length: int = 8) -> dict:
    """Sample independent uniform operators and unique, balanced test programs.

    Each practice prefix alternates operators. Six examples identify both
    permutations because knowing three images determines the fourth. Test
    lengths are 2, 3, and 4, with start-state counts differing by at most one
    within each length. At most 16 unique length-two programs are available.
    """
    assert 1 <= tests_per_length <= 16
    rng = Random(seed)
    mappings = {op: dict(zip(STATES, rng.choice(PERMUTATIONS), strict=True)) for op in OPERATORS}
    starts = {op: rng.sample(list(STATES), len(STATES)) for op in OPERATORS}
    practice = [_query(mappings, starts[op][i], [op]) for i in range(len(STATES)) for op in OPERATORS]

    tests = []
    for length in (2, 3, 4):
        programs = list(product(OPERATORS, repeat=length))
        available = {start: rng.sample(programs, len(programs)) for start in STATES}
        start_order = rng.sample(list(STATES), len(STATES))
        for i in range(tests_per_length):
            start = start_order[i % len(STATES)]
            tests.append(_query(mappings, start, list(available[start].pop())))
    rng.shuffle(tests)
    return {"seed": seed, "permutations": mappings, "practice": practice, "tests": tests}


def question_text(query: dict) -> str:
    """Render only the input, with operators applied from left to right."""
    return " ".join([query["start"], *query["ops"]])


def bayes_reference(world: dict, budget: int) -> dict:
    """Condition the uniform prior over 576 worlds on a practice prefix.

    Accuracy scores deterministic MAP predictions against this actual world;
    expected_accuracy averages their posterior probability of being correct.
    Equal-probability predictions choose the first state in STATES.
    """
    assert budget in range(9)
    observations = world["practice"][:budget]
    hypotheses = []
    for pair in product(PERMUTATIONS, repeat=len(OPERATORS)):
        mappings = {op: dict(zip(STATES, perm, strict=True)) for op, perm in zip(OPERATORS, pair, strict=True)}
        if all(_answer(mappings, query["start"], query["ops"]) == query["answer"] for query in observations):
            hypotheses.append(mappings)
    assert world["permutations"] in hypotheses

    answers, probabilities = [], []
    for query in world["tests"]:
        counts = dict.fromkeys(STATES, 0)
        for mappings in hypotheses: counts[_answer(mappings, query["start"], query["ops"])] += 1
        prediction = max(STATES, key=counts.__getitem__)
        answers.append(prediction)
        probabilities.append(counts[prediction] / len(hypotheses))
    return {
        "answers": answers,
        "accuracy": sum(prediction == query["answer"] for prediction, query in zip(answers, world["tests"], strict=True)) / len(answers),
        "expected_accuracy": sum(probabilities) / len(probabilities),
        "posterior_size": len(hypotheses),
    }
