"""Fixed, fully specified tasks for studying access to existing information.

Learner-visible records contain no answer. Exact grading is separate from the
answer parser and consensus selector, which never consult the task simulator.
"""

from collections import Counter
from hashlib import sha256
import json
from random import Random
import re


FAMILIES = ("arithmetic", "permutation")
DEPTHS = (4, 6, 8)
SYSTEMS = {
    "direct": "Solve the task. Give only the final integer answer in \\boxed{}. Do not write an explanation.",
    "reason": "Solve the task by working through the operations in order. Keep the reasoning concise, use numbers rather than code, and put the final integer answer in \\boxed{}.",
}
BOX = re.compile(r"\\boxed\s*\{\s*([+-]?(?:\d+|\d{1,3}(?:,\d{3})+))\s*\}")


def identifier(problem):
    return sha256(json.dumps(problem, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def make_problems(seed, per_cell, families=FAMILIES):
    rng, rows = Random(seed), []
    for family in families:
        for depth in DEPTHS:
            for index in range(per_cell):
                if family == "arithmetic":
                    operations, multiplications = [], 0
                    for step in range(depth):
                        operation = rng.choice(("add", "subtract", "multiply") if multiplications < 3 else ("add", "subtract"))
                        value = rng.randint(2, 5) if operation == "multiply" else rng.randint(2, 29)
                        operations.append([operation, value])
                        multiplications += operation == "multiply"
                    problem = {"family": family, "depth": depth, "start": rng.randint(2, 39), "operations": operations}
                else:
                    transitions = {name: rng.sample(range(6), 6) for name in ("A", "B")}
                    problem = {"family": family, "depth": depth, "start": rng.randrange(6), "transitions": transitions,
                               "operations": [rng.choice(("A", "B")) for _ in range(depth)]}
                rows.append({**problem, "id": identifier(problem), "cell_index": index})
    assert len({row["id"] for row in rows}) == len(rows)
    return rows


def states(problem):
    values = [problem["start"]]
    for operation in problem["operations"]:
        if problem["family"] == "arithmetic":
            name, amount = operation
            values.append({"add": lambda: values[-1] + amount, "subtract": lambda: values[-1] - amount,
                           "multiply": lambda: values[-1] * amount}[name]())
        else:
            values.append(problem["transitions"][operation][values[-1]])
    return values


def question(problem):
    if problem["family"] == "arithmetic":
        operations = "\n".join(f"{i + 1}. {name.capitalize()}{' by' if name == 'multiply' else ''} {value}."
                               for i, (name, value) in enumerate(problem["operations"]))
        return f"Start with the integer {problem['start']}. Apply these operations to the current value in the listed order:\n{operations}\nWhat is the final integer?"
    rules = "\n".join(f"{name}: " + ", ".join(f"{i}->{value}" for i, value in enumerate(values))
                      for name, values in problem["transitions"].items())
    operations = " ".join(problem["operations"])
    return ("A machine has states 0, 1, 2, 3, 4, 5. Each rule maps the current state to the state on the right.\n"
            f"{rules}\nStart in state {problem['start']}. Apply this rule sequence from left to right: {operations}.\nWhat is the final state?")


def parse_answer(text):
    matches = BOX.findall(text)
    if matches: return int(matches[-1].replace(",", ""))
    if re.fullmatch(r"\s*[+-]?\d+\s*", text): return int(text.strip())
    return None


def consensus(rollouts):
    """Plurality over completed parsed answers; ties use first occurrence.

    No confidence filtering, answer verification, or teacher replacement. An
    entirely unparseable group yields no target and remains in the denominator.
    """
    answers = [row["answer"] for row in rollouts if row["ended"] and row["answer"] is not None]
    if not answers: return {"answer": None, "votes": 0, "valid": 0, "support": 0., "selected_index": None}
    counts = Counter(answers)
    winner = max(counts, key=counts.__getitem__)
    selected = next(i for i, row in enumerate(rollouts) if row["ended"] and row["answer"] == winner)
    return {"answer": winner, "votes": counts[winner], "valid": len(answers),
            "support": counts[winner] / len(rollouts), "selected_index": selected}


def messages(problem, mode, demonstrations=()):
    result = [{"role": "system", "content": SYSTEMS[mode]}]
    for example, response in demonstrations:
        result.extend([{"role": "user", "content": question(example)}, {"role": "assistant", "content": response}])
    return [*result, {"role": "user", "content": question(problem)}]
