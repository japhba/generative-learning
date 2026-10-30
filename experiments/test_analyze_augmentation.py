"""Independent audit checks for candidate provenance, full teacher coverage, and paired inference."""

from copy import deepcopy
import json
import unittest

from analyze_augmentation import ARMS, HISTORICAL, PAIR_COUNTERS, audit_rollout_rows, audit_training_rows, paired_contrasts, summarize_endpoints
from training_task import make_problem, teacher_solution, verify_solution


def candidate(problem, text, ended=True):
    verdict = verify_solution(problem, text)
    if not ended: verdict = {**verdict, "correct": False, "reason": "truncated"}
    return json.loads(json.dumps({"problem_id": problem["id"], "text": text, "ended": ended, "verdict": verdict, "generated_tokens": 8}))


def fixture():
    problems = [make_problem(11, 2, 2, start, start + 4, max_steps=8) for start in range(4)]
    run = {"config": {"train_size": 4, "query_batch_size": 2, "batch_size": 3, "rollouts": 2}, "dataset": {"train": problems}}
    selected = {2: {problem["id"]: teacher_solution(problem) for problem in problems[:2]}, 4: {problems[3]["id"]: teacher_solution(problems[3])}}
    def item(index, source): return {"problem_id": problems[index]["id"], "completion": teacher_solution(problems[index]), "source": source}
    incoming = [[item(0, "teacher"), item(0, "generated"), item(1, "teacher"), item(1, "generated")],
                [item(2, "teacher"), item(3, "teacher"), item(3, "generated")]]
    trained = [incoming[0][:3], incoming[0][3:] + incoming[1]]
    expected = [dict(zip(PAIR_COUNTERS, [2, 4, 1, 3, 1, 2, 2, 2, 1], strict=True)),
                dict(zip(PAIR_COUNTERS, [4, 7, 0, 7, 3, 4, 3, 4, 3], strict=True))]
    rows = [{"budget": budget, "teacher_examples": 2, "extra_examples": extras, "accepted_examples": len(arrivals),
             "trained_examples": len(ready), "pending_examples": pending, "incoming": arrivals, "trained": ready,
             "losses": [1.0] * updates, "counters": counters}
            for budget, extras, pending, arrivals, ready, updates, counters in zip((2, 4), (2, 1), (1, 0), incoming, trained, (1, 2), expected, strict=True)]
    return run, selected, rows


class AugmentationAnalysisTests(unittest.TestCase):
    def test_audit_preserves_teacher_once_and_matches_carry_to_final_partial_batch(self):
        run, selected, rows = fixture()
        boundaries, result = audit_training_rows(run, ARMS[0], rows, selected)
        self.assertEqual(result["teacher_training_examples"], 4)
        self.assertEqual(result["extra_training_examples"], 3)
        self.assertEqual(result["optimizer_updates"], 3)
        self.assertEqual(result["pending_examples"], 0)
        replay = deepcopy(rows)
        for row in replay:
            for item in row["incoming"] + row["trained"]:
                if item["source"] == "generated": item["source"] = "teacher_replay"
            row["counters"].update(rollout_candidates=0, rollout_prompt_tokens=0, rollout_completion_tokens=0, rollout_seconds=0)
        replay_boundaries, _ = audit_training_rows(run, ARMS[1], replay, selected)
        self.assertEqual(boundaries, replay_boundaries)

    def test_duplicate_teacher_missing_teacher_and_changed_extra_fail(self):
        run, selected, rows = fixture()
        for source in ("generated", "teacher_replay"):
            corrupted = deepcopy(rows)
            corrupted[0]["incoming"][0]["source"] = source
            with self.assertRaisesRegex(ValueError, "Teacher coverage"):
                audit_training_rows(run, ARMS[0], corrupted, selected)
        corrupted = deepcopy(rows)
        corrupted[0]["incoming"][1]["completion"] = "A 10"
        with self.assertRaisesRegex(ValueError, "target provenance"):
            audit_training_rows(run, ARMS[0], corrupted, selected)
        corrupted = deepcopy(rows)
        corrupted[1]["trained"] = corrupted[1]["trained"][1:] + corrupted[1]["trained"][:1]
        with self.assertRaisesRegex(ValueError, "full-minibatch buffering"):
            audit_training_rows(run, ARMS[0], corrupted, selected)

    def test_counter_and_optimizer_record_corruption_fails(self):
        run, selected, rows = fixture()
        corrupted = deepcopy(rows)
        corrupted[-1]["counters"]["teacher_training_examples"] = 3
        with self.assertRaisesRegex(ValueError, "Cumulative source"):
            audit_training_rows(run, ARMS[0], corrupted, selected)
        corrupted = deepcopy(rows)
        corrupted[-1]["losses"] = [1.0]
        with self.assertRaisesRegex(ValueError, "optimizer update"):
            audit_training_rows(run, ARMS[0], corrupted, selected)

    def test_first_success_eos_and_candidate_identity_are_independently_verified(self):
        problems = [make_problem(11, 2, 2, 0, 4, max_steps=8), make_problem(11, 2, 2, 1, 5, max_steps=8)]
        first, second = problems
        canonical, alternative = teacher_solution(first), "B 9; A 0; " + teacher_solution(first)
        rows = [{"end_budget": 2, "candidates": [candidate(first, canonical, False), candidate(first, alternative),
                                                candidate(second, "wrong"), candidate(second, teacher_solution(second))],
                 "selected": [{"problem_id": first["id"], "completion": alternative}, {"problem_id": second["id"], "completion": teacher_solution(second)}],
                 "usage": {"completion_tokens": 32}}]
        run = {"config": {"train_size": 2, "query_batch_size": 2, "rollouts": 2}, "dataset": {"train": problems}}
        selected, audit = audit_rollout_rows(run, rows)
        self.assertEqual(selected[2][first["id"]], alternative)
        self.assertEqual(audit["raw_candidates_verified"], 4)
        corrupted = deepcopy(rows)
        corrupted[0]["selected"][0]["completion"] = canonical
        with self.assertRaisesRegex(ValueError, "first valid completed"):
            audit_rollout_rows(run, corrupted)
        corrupted = deepcopy(rows)
        corrupted[0]["candidates"][0]["verdict"]["correct"] = True
        with self.assertRaisesRegex(ValueError, "independent verification plus EOS"):
            audit_rollout_rows(run, corrupted)
        corrupted = deepcopy(rows)
        corrupted[0]["candidates"][0]["problem_id"] = second["id"]
        with self.assertRaisesRegex(ValueError, "candidate identities"):
            audit_rollout_rows(run, corrupted)

    def test_new_arm_pairing_and_historical_context_remain_distinct(self):
        runs = []
        for seed, generated, replay, historical in ((1, .6, .3, .8), (2, .8, .9, .1)):
            metrics = {arm: {4: {"metrics": {"in_distribution": {"accuracy": value}}}} for arm, value in ((ARMS[0], generated), (ARMS[1], replay))}
            runs.append({"seed": seed, "metrics": metrics, "historical": {4: {"metrics": {"in_distribution": {"accuracy": historical}}}}})
        contrasts = paired_contrasts(runs, 1000)
        primary = next(row for row in contrasts if row["second"] == ARMS[1])
        self.assertEqual(primary["seeds"], [1, 2])
        self.assertAlmostEqual(primary["accuracy_difference"]["mean"], .1)
        self.assertEqual(primary["role"], "paired new-arm comparison")
        historical = [row for row in contrasts if row["second"] == HISTORICAL]
        self.assertEqual(len(historical), 2)
        self.assertTrue(all("not independent confirmation" in row["role"] for row in historical))

    def test_partial_endpoint_does_not_substitute_a_lower_checkpoint(self):
        counters = {"problems_presented": 4, "training_examples": 7, "optimizer_updates": 3, "training_tokens": 100,
                    "rollout_prompt_tokens": 20, "rollout_completion_tokens": 30, "training_seconds": 1, "rollout_seconds": 2}
        record = {"metrics": {"in_distribution": {"accuracy": .5}}, "counters": counters}
        run = {"seed": 1, "config": {"train_size": 4}, "metrics": {ARMS[0]: {4: record}, ARMS[1]: {2: record}}, "historical": {4: record}}
        result = summarize_endpoints([run], paired_contrasts([run], 1000), 1000)
        self.assertNotIn(ARMS[1], result["accuracy"])
        self.assertEqual(result["primary"], [])
        self.assertEqual(result["accuracy"][ARMS[0]]["in_distribution"]["n_seeds"], 1)
        self.assertIsNone(result["accuracy"][ARMS[0]]["in_distribution"]["ci95"])
        self.assertEqual(result["costs"][ARMS[0]]["reported_tokens"]["mean"], 150)


if __name__ == "__main__": unittest.main()
