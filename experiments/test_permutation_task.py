"""Offline checks of composition, information exposure, and exact inference."""

from collections import Counter
import unittest

from experiments.permutation_task import STATES, _answer, bayes_reference, make_world, question_text


class PermutationTaskTests(unittest.TestCase):
    def test_composition_is_left_to_right(self):
        mappings = {"f": dict(zip(STATES, "BCDA", strict=True)), "g": dict(zip(STATES, "BADC", strict=True))}
        self.assertEqual(_answer(mappings, "B", ["f", "g", "f"]), "A")
        query = {"start": "B", "ops": ["f", "g", "f"], "answer": "SECRET"}
        self.assertEqual(question_text(query), "B f g f")

    def test_worlds_are_deterministic_and_balanced(self):
        world = make_world(17)
        self.assertEqual(world, make_world(17))
        self.assertNotEqual(world["permutations"], make_world(18)["permutations"])
        self.assertEqual(len(world["practice"]), 8)
        self.assertEqual([query["ops"][0] for query in world["practice"]], ["f", "g"] * 4)
        self.assertEqual(len({question_text(query) for query in world["practice"]}), 8)
        self.assertEqual(len({question_text(query) for query in world["tests"]}), 24)
        self.assertEqual(Counter((len(query["ops"]), query["start"]) for query in world["tests"]), Counter({(length, start): 2 for length in (2, 3, 4) for start in STATES}))
        for query in world["practice"] + world["tests"]:
            state = query["start"]
            for op in query["ops"]: state = world["permutations"][op][state]
            self.assertEqual(query["answer"], state)

    def test_six_examples_identify_the_world(self):
        for seed in range(12):
            world = make_world(seed)
            for budget in (6, 8):
                with self.subTest(seed=seed, budget=budget):
                    result = bayes_reference(world, budget)
                    self.assertEqual(result["posterior_size"], 1)
                    self.assertEqual(result["answers"], [query["answer"] for query in world["tests"]])
                    self.assertEqual(result["accuracy"], 1.0)
                    self.assertEqual(result["expected_accuracy"], 1.0)

    def test_posterior_sizes_and_probability_ranges(self):
        world = make_world(31)
        for budget, size in enumerate((576, 144, 36, 12, 4, 2, 1, 1, 1)):
            with self.subTest(budget=budget):
                result = bayes_reference(world, budget)
                self.assertEqual(result["posterior_size"], size)
                self.assertGreaterEqual(result["expected_accuracy"], 0.25)
                self.assertLessEqual(result["expected_accuracy"], 1.0)
                self.assertGreaterEqual(result["accuracy"], 0.0)
                self.assertLessEqual(result["accuracy"], 1.0)

    def test_ties_choose_first_state(self):
        world = make_world(4)
        world["tests"] = [world["practice"][0]]
        result = bayes_reference(world, 0)
        self.assertEqual(result["answers"], ["A"])
        self.assertEqual(result["expected_accuracy"], 0.25)


if __name__ == "__main__": unittest.main()
