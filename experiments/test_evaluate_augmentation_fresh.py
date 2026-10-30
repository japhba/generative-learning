"""Checks for unseen-identity sampling and strict frozen-test grading."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluate_augmentation_fresh import SOURCES, freeze_evaluation_source, prepare, sample_fresh, sha256, verified_metrics
from training_task import make_problem, split_for, teacher_solution, verify_solution


class FreshEvaluationTests(unittest.TestCase):
    def test_evaluator_archive_pins_live_code_and_detects_later_edits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            live, output = root / "live", root / "output"
            live.mkdir()
            output.mkdir()
            for name in SOURCES: (live / name).write_text(f"# original {name}\n")
            selection = {"selection_source_sha256": {name: sha256(live / name) for name in SOURCES}}
            (output / "selection_record.json").write_text(json.dumps(selection))
            manifest = freeze_evaluation_source(output, live)
            self.assertEqual(manifest, freeze_evaluation_source(output, live))
            evaluator = live / "evaluate_augmentation_fresh.py"
            original = evaluator.read_bytes()
            evaluator.write_text("# changed evaluator\n")
            with self.assertRaisesRegex(ValueError, "Live evaluator sources changed"):
                freeze_evaluation_source(output, live)
            evaluator.write_bytes(original)
            (output / "evaluation_source/evaluate_augmentation_fresh.py").write_text("# changed archive\n")
            with self.assertRaisesRegex(ValueError, "Archived evaluation sources"):
                freeze_evaluation_source(output, live)
            (live / "training_task.py").write_text("# changed task\n")
            with self.assertRaisesRegex(ValueError, "frozen selection version"):
                freeze_evaluation_source(output, live)

    def test_sampling_excludes_all_seen_identities_and_preserves_fixed_split(self):
        metadata = {"moduli": [11], "increments": [1, 2], "eval_strata": {"near": [1, 2], "far": [3, 4]}, "max_steps": 6}
        original, full_counts = sample_fresh(metadata, set(), 17, 12)
        excluded = {problem["id"] for group in original.values() for problem in group}
        selected, counts = sample_fresh(metadata, excluded, 93201, 8)
        self.assertEqual(selected, sample_fresh(metadata, excluded, 93201, 8)[0])
        selected_ids = [problem["id"] for group in selected.values() for problem in group]
        self.assertEqual(len(selected_ids), len(set(selected_ids)))
        self.assertFalse(set(selected_ids) & excluded)
        self.assertTrue(all(split_for(problem_id) == "test" for problem_id in selected_ids))
        for stratum, group in selected.items():
            self.assertEqual(counts[stratum]["previously_evaluated"], 12)
            self.assertEqual(counts[stratum]["full_heldout_pool"], full_counts[stratum]["full_heldout_pool"])
            self.assertTrue(all(problem["shortest_distance"] in metadata["eval_strata"][stratum] for problem in group))

    def test_insufficient_pool_and_resampling_existing_directory_fail(self):
        metadata = {"moduli": [11], "increments": [1], "eval_strata": {"near": [1]}, "max_steps": 6}
        with self.assertRaisesRegex(ValueError, "Insufficient unseen"):
            sample_fresh(metadata, set(), 93201, 999)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileExistsError, "never resample"):
                prepare(Path(directory))

    def test_exact_eos_grading_and_corrupted_identity_or_verdict_fail(self):
        problems = [make_problem(11, 2, 2, 0, 4), make_problem(11, 2, 2, 1, 5)]
        predictions = [{"problem_id": problem["id"], "stratum": "test", "text": teacher_solution(problem), "ended": index == 0,
                        "verdict": verify_solution(problem, teacher_solution(problem))} for index, problem in enumerate(problems)]
        predictions[1]["verdict"].update(correct=False, reason="truncated")
        dataset = {"eval": {"test": problems}}
        self.assertEqual(verified_metrics(dataset, json.loads(json.dumps(predictions)))["test"]["accuracy"], .5)
        corrupted = deepcopy(predictions)
        corrupted[1]["verdict"]["correct"] = True
        with self.assertRaisesRegex(ValueError, "independent verification plus EOS"):
            verified_metrics(dataset, corrupted)
        corrupted = deepcopy(predictions)
        corrupted[1]["problem_id"] = corrupted[0]["problem_id"]
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            verified_metrics(dataset, corrupted)
        corrupted = deepcopy(predictions)
        corrupted[1]["verdict"]["reason"] = "wrong_target"
        with self.assertRaisesRegex(ValueError, "independent verification plus EOS"):
            verified_metrics(dataset, corrupted)


if __name__ == "__main__": unittest.main()
