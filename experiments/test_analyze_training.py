"""Behavioral checks for arithmetic, stopping, paired inference, and archive provenance."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from analyze_training import candidate_mechanisms, clocks, constant_action_successes, estimate, evaluation_diagnostics, full_pool_constant_action_baselines, paired_contrasts, trajectory_diagnostic, weight_cosines, write_json
from training_task import make_dataset, make_problem, split_for, verify_solution


def prediction(problem, text, ended=True):
    verdict = verify_solution(problem, text)
    if not ended: verdict = {**verdict, "correct": False, "reason": "truncated"}
    return {"problem_id": problem["id"], "text": text, "ended": ended, "verdict": verdict}


class AnalysisTests(unittest.TestCase):
    def test_goal_prefix_is_not_rescored_as_complete_success(self):
        problem = make_problem(11, 2, 2, 0, 4, max_steps=4)
        expectations = [("A 2; C 4", False, "reached_goal_then_missing_eos"),
                        ("A 2; C 4; A 6", True, "reached_goal_then_wrong_target"),
                        ("A 2; C 4; A 7", True, "reached_goal_then_wrong_transition"),
                        ("A 2; C 4; A", True, "reached_goal_then_format")]
        for text, ended, category in expectations:
            with self.subTest(text=text, ended=ended):
                result = trajectory_diagnostic(problem, prediction(problem, text, ended))
                self.assertFalse(result["primary_correct"])
                self.assertTrue(result["valid_prefix_reaches_goal_within_horizon"])
                self.assertEqual(result["category"], category)
        wrong_arithmetic = trajectory_diagnostic(problem, prediction(problem, "A 9; C 4"))
        self.assertFalse(wrong_arithmetic["valid_prefix_reaches_goal_within_horizon"])
        self.assertEqual(wrong_arithmetic["category"], "arithmetic_error_before_goal")
        late = trajectory_diagnostic(problem, prediction(problem, "A 2; B 0; A 2; B 0; A 2; C 4"))
        self.assertFalse(late["valid_prefix_reaches_goal_within_horizon"])
        self.assertTrue(late["valid_prefix_reaches_goal_only_after_horizon"])
        self.assertEqual(late["category"], "no_goal_in_correct_horizon")

    def test_malformed_denominators_and_duplicate_completions_are_explicit(self):
        problem = make_problem(11, 2, 2, 0, 4)
        candidates = [prediction(problem, text, ended) for text, ended in
                      [("A 2; C 4", True), (" A 2; C 4 ", False), ("A 2; C 4; A", False), ("A 9; C 4", True), ("invalid", True)]]
        result = candidate_mechanisms(candidates, {problem["id"]: problem})
        self.assertEqual(result["mean_distinct_completions_per_prompt"], 4)
        self.assertEqual(result["denominators"]["parseable_first_moves"], 4)
        self.assertEqual(result["first_step_accuracy_all_candidates"], 0.6)
        self.assertEqual(result["first_step_accuracy_parseable_first_moves"], 0.75)
        self.assertEqual(result["text_valid_but_missing_eos_count"], 1)

    def test_saved_success_and_test_identity_corruption_fail_loudly(self):
        problem = make_problem(11, 2, 2, 0, 4)
        corrupted = prediction(problem, "A 9; C 4")
        corrupted["verdict"]["correct"] = True
        with self.assertRaisesRegex(ValueError, "independent text verification"):
            trajectory_diagnostic(problem, corrupted)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            row = {**prediction(problem, "A 2; C 4"), "stratum": "test"}
            write_json(root / "sft_n0_predictions.json", [row, row])
            run = {"directory": root, "dataset": {"eval": {"test": [problem]}}, "metrics": {"sft": {0: {}}}}
            with self.assertRaisesRegex(ValueError, "fixed held-out identities"):
                evaluation_diagnostics(run)

    def test_single_seed_has_no_interval_and_contrasts_keep_seed_pairing(self):
        self.assertIsNone(estimate([0.25])["ci95"])
        runs = []
        for seed, sft, rollout in [(1, 0.3, 0.6), (2, 0.9, 0.8)]:
            metrics = {arm: {64: {"metrics": {"test": {"accuracy": value}}}} for arm, value in [("sft", sft), ("rollout_distill", rollout)]}
            runs.append({"seed": seed, "metrics": metrics})
        result = paired_contrasts(runs, 2000)[0]
        self.assertEqual(result["seeds"], [1, 2])
        self.assertAlmostEqual(result["accuracy_difference"]["mean"], -0.1)
        np.testing.assert_allclose(result["accuracy_difference"]["ci95"], [-0.3, 0.1])

    def test_cost_clock_does_not_double_count_targets_or_evaluation(self):
        counters = {"problems_presented": 8, "training_examples": 4, "optimizer_updates": 1, "training_tokens": 200,
                    "supervised_tokens": 77, "rollout_prompt_tokens": 300, "rollout_completion_tokens": 100,
                    "training_seconds": 2.0, "rollout_seconds": 3.0, "evaluation_tokens": 10000, "evaluation_seconds": 100.0}
        result = clocks(counters)
        self.assertEqual(result["reported_tokens"], 600)
        self.assertEqual(result["phase_seconds"], 5.0)

    def test_full_pool_oracle_matches_closed_form_constant_actions(self):
        dataset = make_dataset(7, train_size=8, eval_per_stratum=2, moduli=(11,), increments=(1, 2),
                               train_distances=(1, 2, 3), eval_strata={"test": (1, 2, 3, 4)}, max_steps=6)
        reference = {operation: 0 for operation in "ABC"}
        count = 0
        for add in (1, 2):
            for subtract in (1, 2):
                for start in range(11):
                    for target in range(11):
                        if start == target: continue
                        problem = make_problem(11, add, subtract, start, target, max_steps=6)
                        if split_for(problem["id"]) != "test" or problem["shortest_distance"] not in (1, 2, 3, 4): continue
                        count += 1
                        expected = {"A": any((start + k * add) % 11 == target for k in range(1, 7)),
                                    "B": any((start - k * subtract) % 11 == target for k in range(1, 7)),
                                    "C": any(start * 2**k % 11 == target for k in range(1, 7))}
                        self.assertEqual(constant_action_successes(problem), expected)
                        for operation in reference: reference[operation] += expected[operation]
        actual = full_pool_constant_action_baselines(dataset["metadata"])["strata"]["test"]
        self.assertEqual(actual["count"], count)
        self.assertEqual(actual["constant_action_success_counts"], reference)

    def test_weight_archive_override_has_no_hidden_fallback(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "shared" / "run" / "weight_probes"
            archive.mkdir(parents=True)
            originals = root / "originals"
            originals.mkdir()
            metrics = {}
            for arm, vector in [("sft", [1, 0]), ("rollout_distill", [0, 1])]:
                name = f"{arm}_n64_weight_deltas.npz"
                np.savez(originals / name, weight=[1, 0])
                np.savez(archive / name, weight=vector)
                metrics[arm] = {64: {"weight_probe_path": str(originals / name), "weight_probes": {"weight": {}}}}
            run = {"seed": 1, "directory": root / "run", "metrics": metrics}
            result = weight_cosines([run], root / "shared")
            self.assertEqual(result["results"][0]["cosine"], 0)
            (archive / "sft_n64_weight_deltas.npz").unlink()
            unavailable = weight_cosines([run], root / "shared")
            self.assertEqual(unavailable["results"], [])
            self.assertEqual(len(unavailable["unavailable"]), 1)


if __name__ == "__main__": unittest.main()
