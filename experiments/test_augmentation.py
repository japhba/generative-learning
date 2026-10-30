"""Checks of teacher coverage, genuine generated targets, replay, and immutable lineage."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_augmentation as augmentation
from train_planning import completed_minibatches
from training_task import make_problem, teacher_solution, verify_solution


def candidate(problem, text, ended=True):
    verdict = verify_solution(problem, text)
    if not ended: verdict = {**verdict, "correct": False, "reason": "truncated"}
    return {"problem_id": problem["id"], "text": text, "ended": ended, "verdict": verdict}


class AugmentationTargetTests(unittest.TestCase):
    def setUp(self):
        self.problems = [make_problem(11, 2, 2, 0, target, max_steps=8) for target in (4, 6, 8, 10, 1)]

    def test_first_completed_success_preserves_a_longer_noncanonical_target(self):
        problem, unsolved = self.problems[:2]
        alternative = "  A 2; B 0; A 2; C 4\n"
        predictions = [candidate(problem, "A 4"), candidate(problem, teacher_solution(problem), ended=False),
                       candidate(problem, alternative), candidate(problem, teacher_solution(problem))]
        predictions += [candidate(unsolved, "not a plan") for _ in range(4)]
        selected = augmentation.select_successes([problem, unsolved], predictions, 4)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["problem_id"], problem["id"])
        self.assertEqual(selected[0]["completion"], alternative)
        self.assertEqual(selected[0]["steps"], 4)
        self.assertEqual(selected[0]["teacher_steps"], 2)

    def test_forged_success_or_missing_eos_cannot_become_a_target(self):
        problem = self.problems[0]
        for text, ended in [("A 4", True), (teacher_solution(problem), False)]:
            with self.subTest(text=text, ended=ended):
                forged = candidate(problem, text, ended)
                forged["verdict"]["correct"] = True
                with self.assertRaises(AssertionError): augmentation.select_successes([problem], [forged], 1)

    def test_candidate_groups_cannot_cross_prompt_boundaries(self):
        first, second = self.problems[:2]
        predictions = [candidate(second, teacher_solution(second)), candidate(first, teacher_solution(first))]
        with self.assertRaises(AssertionError): augmentation.select_successes([first, second], predictions, 1)
        with self.assertRaises(AssertionError): augmentation.select_successes([first, second], predictions, 2)

    def test_unsolved_prompts_still_receive_all_teacher_targets(self):
        predictions = [candidate(problem, "bad") for problem in self.problems for _ in range(4)]
        selected = augmentation.select_successes(self.problems, predictions, 4)
        self.assertEqual(selected, [])
        for arm in augmentation.ARMS:
            examples = augmentation.augment_examples(self.problems, selected, arm)
            self.assertEqual([row["problem_id"] for row in examples], [problem["id"] for problem in self.problems])
            self.assertTrue(all(row["source"] == "teacher" for row in examples))
            self.assertTrue(all(verify_solution(problem, row["completion"])["correct"] for problem, row in zip(self.problems, examples, strict=True)))

    def test_replay_replaces_only_supplemental_targets_and_matches_buffer_boundaries(self):
        targets = {problem["id"]: teacher_solution(problem) for problem in self.problems}
        generated = {self.problems[0]["id"]: "A 2; B 0; A 2; C 4", self.problems[2]["id"]: "A 2; C 4; C 8"}
        streams, boundaries = {}, {}
        for arm in augmentation.ARMS:
            pending, trained, sizes = [], [], []
            for start, stop in [(0, 2), (2, 4), (4, 5)]:
                problems = self.problems[start:stop]
                selected = [{"problem_id": problem["id"], "completion": generated[problem["id"]]} for problem in reversed(problems) if problem["id"] in generated]
                incoming = augmentation.augment_examples(problems, selected, arm)
                ready, pending = completed_minibatches(pending, incoming, 4, stop == len(self.problems))
                sizes.append(len(ready))
                if stop < len(self.problems): self.assertEqual(len(ready) % 4, 0)
                trained.extend(ready)
            self.assertEqual(pending, [])
            self.assertEqual([row["problem_id"] for row in trained if row["source"] == "teacher"], list(targets))
            self.assertEqual([row["problem_id"] for row in trained if row["source"] != "teacher"], list(generated))
            streams[arm], boundaries[arm] = trained, sizes
        self.assertEqual(boundaries["teacher_rollout"], boundaries["teacher_replay"])
        self.assertEqual(boundaries["teacher_rollout"], [0, 4, 3])
        for proposed, replayed in zip(streams["teacher_rollout"], streams["teacher_replay"], strict=True):
            self.assertEqual(proposed["problem_id"], replayed["problem_id"])
            identity = proposed["problem_id"]
            if proposed["source"] == "teacher":
                self.assertEqual(proposed, replayed)
            else:
                self.assertEqual(proposed["completion"], generated[identity])
                self.assertEqual(replayed["completion"], targets[identity])
                self.assertEqual(replayed["source"], "teacher_replay")

    def test_duplicate_or_unpresented_supplements_fail(self):
        problem, other = self.problems[:2]
        row = {"problem_id": problem["id"], "completion": teacher_solution(problem)}
        with self.assertRaises(AssertionError): augmentation.augment_examples([problem], [row, row], "teacher_rollout")
        with self.assertRaises(AssertionError): augmentation.augment_examples([other], [row], "teacher_replay")


class AugmentationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        source = self.root / "experiments"
        source.mkdir()
        for name in augmentation.SOURCES: (source / name).write_bytes((augmentation.ROOT / "experiments" / name).read_bytes())
        self.parent, self.output, self.warm = (self.root / name for name in ("parent", "new", "warm"))
        self.parent.mkdir()
        self.warm.mkdir()
        (self.warm / "model.safetensors").write_bytes(b"synthetic checkpoint identity; never loaded")
        parent_config = {"seed": 6200, "train_size": 4096, "cache_directory": str(self.root / "cache"),
                         "source_sha256": {name: augmentation.sha256_file(source / name) for name in ("train_planning.py", "training_task.py")}}
        (self.parent / "config.json").write_text(json.dumps(parent_config))
        (self.parent / "dataset.json").write_text('{"train": [], "eval": {}}\n')
        self.args = argparse.Namespace(parent=self.parent, output=self.output, warm_checkpoint=self.warm)
        self.addCleanup(patch.stopall)
        patch.object(augmentation, "ROOT", self.root).start()
        patch.object(augmentation, "configure_environment", return_value=self.root / "fast").start()

    def test_preparation_preserves_exact_dataset_source_and_checkpoint_lineage(self):
        output, config, dataset = augmentation.prepare(self.args)
        self.assertEqual((output / "dataset.json").read_bytes(), (self.parent / "dataset.json").read_bytes())
        self.assertEqual(config["dataset_sha256"], augmentation.sha256_file(output / "dataset.json"))
        self.assertEqual(config["warm_model_sha256"], augmentation.sha256_file(self.warm / "model.safetensors"))
        self.assertEqual(config["parent_config_sha256"], augmentation.sha256_file(self.parent / "config.json"))
        for name in augmentation.SOURCES:
            self.assertEqual((output / "source" / name).read_bytes(), (self.root / "experiments" / name).read_bytes())
            self.assertEqual(config["source_sha256"][name], augmentation.sha256_file(output / "source" / name))
        self.assertEqual(augmentation.prepare(self.args), (output, config, dataset))

    def test_parent_implementation_changes_fail_before_preparation(self):
        (self.root / "experiments" / "train_planning.py").write_text("changed training implementation\n")
        with self.assertRaises(AssertionError): augmentation.prepare(self.args)
        self.assertFalse(self.output.exists())

    def test_replaced_warm_weights_fail_before_reusing_outputs(self):
        augmentation.prepare(self.args)
        (self.warm / "model.safetensors").write_bytes(b"different checkpoint")
        with self.assertRaises(AssertionError): augmentation.prepare(self.args)

    def test_modified_dataset_copy_cannot_silently_change_training(self):
        augmentation.prepare(self.args)
        (self.output / "dataset.json").write_text('{"train": [{"id": "unrecorded identity"}], "eval": {}}\n')
        with self.assertRaises(AssertionError): augmentation.prepare(self.args)

    def test_modified_source_snapshot_cannot_silently_break_reproducibility(self):
        augmentation.prepare(self.args)
        (self.output / "source" / "train_augmentation.py").write_text("unrecorded source snapshot\n")
        with self.assertRaises(AssertionError): augmentation.prepare(self.args)


if __name__ == "__main__": unittest.main()
