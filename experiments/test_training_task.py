"""Meaningful checks of supervision, alternate plans, and test-set isolation."""

from itertools import product
import unittest

from training_task import OPERATIONS, is_correct, make_dataset, make_problem, shortest_plan, split_for, teacher_solution, transition, verify_solution


class PlanningTaskTests(unittest.TestCase):
    def test_bfs_matches_independent_exhaustive_search(self):
        problem = make_problem(11, 2, 3, 0, 7, max_steps=6)
        teacher = shortest_plan(problem)
        self.assertTrue(is_correct(problem, teacher_solution(problem)))
        for length in range(len(teacher)):
            for operations in product(OPERATIONS, repeat=length):
                state = problem["start"]
                for operation in operations:
                    state = {"A": state + 2, "B": state - 3, "C": state * 2}[operation] % 11
                self.assertNotEqual(state, problem["target"])

    def test_distinct_generated_solutions_are_accepted(self):
        problem = make_problem(11, 2, 2, 0, 4, max_steps=4)
        self.assertTrue(is_correct(problem, "A 2; A 4"))
        self.assertTrue(is_correct(problem, "A 2; C 4"))
        self.assertTrue(is_correct(problem, "A 2; B 0; A 2; C 4"))
        self.assertTrue(is_correct(problem, "  A 2 ;\tC 4 \n"))

    def test_rejects_reward_hacks_and_incorrect_traces(self):
        problem = make_problem(11, 2, 2, 0, 4, max_steps=4)
        invalid = ["4", "A 4", "A 2; B 4", "A 2; C 15", "A 2; C 4;", "A 2; C 4\nActually, done.",
                   "```A 2; C 4```", "A 2; C 4<|endoftext|>", "A 2; D 4", "A 02; C 4", "", "A 2; C -7"]
        for text in invalid:
            with self.subTest(text=text):
                self.assertFalse(is_correct(problem, text))
        self.assertEqual(verify_solution(problem, "A 2")["reason"], "wrong_target")
        self.assertEqual(verify_solution(problem, "A 2; B 0; A 2; B 0; A 2; C 4")["reason"], "too_many_steps")

    def test_wraparound_is_exact(self):
        self.assertEqual(transition(11, 3, 4, 10, "A"), 2)
        self.assertEqual(transition(11, 3, 4, 1, "B"), 8)
        self.assertEqual(transition(11, 3, 4, 7, "C"), 3)

    def test_sampling_and_split_are_deterministic_and_separate(self):
        options = {"train_size": 32, "eval_per_stratum": 8, "moduli": (17,), "increments": (1, 2, 3),
                   "train_distances": (1, 2, 3), "eval_strata": {"short": (1, 2), "long": (3, 4)}, "max_steps": 6}
        first, repeat, second = (make_dataset(seed, **options) for seed in (7, 7, 8))
        self.assertEqual(first, repeat)
        train = {problem["id"] for data in (first, second) for problem in data["train"]}
        test = {problem["id"] for data in (first, second) for group in data["eval"].values() for problem in group}
        self.assertFalse(train & test)
        self.assertTrue(all(split_for(problem_id) == "train" for problem_id in train))
        for group in [first["train"], *first["eval"].values()]:
            self.assertEqual(len(group), len({problem["id"] for problem in group}))
            self.assertTrue(all(is_correct(problem, teacher_solution(problem)) for problem in group))

    def test_horizon_does_not_change_identity_or_split(self):
        short, long = (make_problem(11, 2, 2, 0, 4, max_steps=horizon) for horizon in (2, 8))
        self.assertEqual(short["id"], long["id"])
        self.assertEqual(short["split"], long["split"])
        with self.assertRaises(ValueError):
            make_problem(11, 2, 2, 0, 4, max_steps=1)


if __name__ == "__main__": unittest.main()
