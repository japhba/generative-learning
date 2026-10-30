import unittest

from elicitation_answer import parse_answer


class FinalAnswerTests(unittest.TestCase):
    def test_explicit_unboxed_final(self):
        self.assertEqual(parse_answer("work 12+7\nThe final integer is 375."), 375)
        self.assertEqual(parse_answer("work 3\nFinal integer: \\(-17\\)"), -17)
        self.assertEqual(parse_answer("Answer: **1,234**"), 1234)

    def test_reject_ambiguous_or_unfinished_work(self):
        self.assertIsNone(parse_answer("work 3+4=7"))
        self.assertIsNone(parse_answer("Final answer is 2 or 3"))
        self.assertIsNone(parse_answer("Final answer is 3.5"))
        self.assertIsNone(parse_answer("Final answer is 1,2"))

    def test_boxed_priority(self):
        self.assertEqual(parse_answer("Answer: 4\nFinal: \\boxed{6}"), 6)


if __name__ == "__main__": unittest.main()
