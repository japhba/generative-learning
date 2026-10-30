"""Scientific integrity checks for exact grading and reward-free selection."""

import unittest

from elicitation_task import consensus, make_problems, parse_answer, question, states


class ElicitationTaskTests(unittest.TestCase):
    def test_arithmetic_order(self):
        problem = {"family": "arithmetic", "start": 3, "operations": [["add", 4], ["multiply", 2], ["subtract", 20]]}
        self.assertEqual(states(problem), [3, 7, 14, -6])

    def test_permutation_order(self):
        problem = {"family": "permutation", "start": 0, "transitions": {"A": [1, 0, 2], "B": [0, 2, 1]}, "operations": ["A", "B", "A"]}
        self.assertEqual(states(problem), [0, 1, 2, 2])

    def test_answer_parser_does_not_guess_from_working(self):
        self.assertEqual(parse_answer("work 99; final \\boxed{ -1,234 }"), -1234)
        self.assertEqual(parse_answer("42"), 42)
        self.assertIsNone(parse_answer("I calculated 42 but have not finished."))
        self.assertIsNone(parse_answer("\\boxed{2+3}"))
        self.assertIsNone(parse_answer("\\boxed{1,2}"))

    def test_consensus_ignores_truth_and_charges_invalid(self):
        rows = [{"answer": value, "ended": ended} for value, ended in [(8, True), (7, True), (8, True), (7, False)]]
        self.assertEqual(consensus(rows), {"answer": 8, "votes": 2, "valid": 3, "support": .5, "selected_index": 0})
        self.assertEqual(consensus(rows[:2])["answer"], 8)
        self.assertIsNone(consensus([rows[-1]])["answer"])

    def test_disjoint_reproducible_data(self):
        first, second = make_problems(61000, 16), make_problems(62000, 16)
        self.assertEqual(first, make_problems(61000, 16))
        self.assertFalse({row["id"] for row in first} & {row["id"] for row in second})
        for row in first:
            self.assertNotIn("answer", row)
            self.assertIn("final", question(row))
            self.assertEqual(len(states(row)), row["depth"] + 1)


if __name__ == "__main__": unittest.main()
