"""Audit consolidation targets, causal schedules, and all held-out predictions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from elicitation_answer import parse_answer
from elicitation_task import DEPTHS, consensus, states
from run_elicitation_pilot import digest, save
from train_elicitation import ARMS, CHECKPOINTS, SEEDS, SOURCES, read_rows


LABELS = {"frozen_reasoning": "Frozen reasoning source", "on_policy_reasoning": "Current reasoning source",
          "direct_self_imitation": "Direct self-imitation"}


def grade(path, problems):
    rows = read_rows(path)
    assert [r["id"] for r in rows] == [p["id"] for p in problems]
    answers = [parse_answer(r["text"]) for r in rows]
    correct = np.array([int(r["ended"] and answer == states(p)[-1]) for r, answer, p in zip(rows, answers, problems, strict=True)])
    return {"accuracy": float(correct.mean()), "correct": correct.tolist(), "answers": answers,
            "completion": float(np.mean([r["ended"] for r in rows])),
            "parsed": float(np.mean([r["ended"] and a is not None for r, a in zip(rows, answers, strict=True)])),
            "mean_output_tokens": float(np.mean([r["completion_tokens"] for r in rows])),
            "by_depth": {str(d): float(correct[[p["depth"] == d for p in problems]].mean()) for d in DEPTHS},
            "source_sha256": digest(path)}


def analyze(output, reports):
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(Path(__file__).with_name(name)) for name in SOURCES}
    train = json.loads((output / "train.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert protocol["train_sha256"] == digest(output / "train.json")
    assert protocol["evaluation_sha256"] == digest(output / "evaluation.json")
    cases = []
    for seed in SEEDS:
        source = output / str(seed)
        order = json.loads((source / "environment.json").read_text())["order"]
        for epoch in range(4): assert sorted(order[96*epoch:96*(epoch+1)]) == list(range(96))
        sequence = [train[i] for i in order]
        first_frozen = read_rows(source / "frozen_candidates/round_0.jsonl")
        first_live = read_rows(source / "on_policy_reasoning/round_0.jsonl")
        assert [r["token_ids"] for r in first_frozen] == [r["token_ids"] for r in first_live], "Initial paired teacher draws must agree."
        for arm in ARMS:
            directory = source / arm
            complete = json.loads((directory / "complete.json").read_text())
            updates, teacher = read_rows(directory / "updates.jsonl"), []
            assert len(updates) == 24
            counted_valid, latest = 0, {}
            latest_by_budget = {0: {}}
            for step, update in enumerate(updates):
                assert update["step"] == step and update["budget"] == (step + 1)*16
                expected = sequence[16*step:16*(step+1)]
                assert [r["id"] for r in update["selected"]] == [p["id"] for p in expected]
                path = Path(update["candidate_path"])
                assert digest(path) == update["candidate_sha256"]
                candidates = [{**r, "answer": parse_answer(r["text"])} for r in read_rows(path)]
                size = 1 if arm == "direct_self_imitation" else 4
                assert len(candidates) == 16*size
                truths, voted, individual = [], [], []
                for i, (selected, problem) in enumerate(zip(update["selected"], expected, strict=True)):
                    group = candidates[size*i:size*(i+1)]
                    assert [r["id"] for r in group] == [problem["id"]]*size
                    vote = consensus(group)
                    assert vote == selected["vote"]
                    if arm == "direct_self_imitation": assert selected["target_ids"] == group[0]["token_ids"]
                    elif vote["answer"] is None: assert selected["target_ids"] == []
                    else: assert parse_answer(selected["target_text"]) == vote["answer"]
                    counted_valid += bool(selected["target_ids"])
                    latest[problem["id"]] = vote["answer"]
                    truth = states(problem)[-1]
                    truths.append(truth)
                    voted.append(int(vote["answer"] == truth))
                    individual.extend(int(r["ended"] and r["answer"] == truth) for r in group)
                count = update["counters"]
                assert count["practice_presentations"] == (step + 1)*16
                assert count["optimizer_updates"] == (step + 1)*4
                assert count["training_presentations"] == (step + 1)*64
                assert count["rollout_responses"] == (step + 1)*16*size
                assert count["valid_targets"] == counted_valid
                teacher.append({"budget": (step+1)*16, "plurality_accuracy": float(np.mean(voted)),
                                "raw_sample_accuracy": float(np.mean(individual)), "first_loss": update["losses"][0],
                                "last_loss": update["losses"][-1], "counters": count})
                if update["budget"] in CHECKPOINTS: latest_by_budget[update["budget"]] = dict(latest)
            assert complete["counters"] == updates[-1]["counters"]
            checkpoint_records = read_rows(directory / "checkpoints.jsonl")
            assert [r["budget"] for r in checkpoint_records] == list(CHECKPOINTS)
            checkpoints = []
            for record in checkpoint_records:
                n = record["budget"]
                metrics = {mode: grade(directory / f"eval_{mode}_n{n}.jsonl", evaluation) for mode in ("direct", "reason")}
                practice = grade(directory / f"practice_direct_n{n}.jsonl", train)
                seen = latest_by_budget[n]
                agreements = [answer == seen[p["id"]] for p, answer in zip(train, practice["answers"], strict=True) if p["id"] in seen and seen[p["id"]] is not None]
                practice["latest_self_target_agreement"] = float(np.mean(agreements)) if agreements else None
                checkpoints.append({"budget": n, "metrics": metrics, "practice": practice, "counters": record["counters"]})
            cases.append({"seed": seed, "arm": arm, "checkpoints": checkpoints, "teacher": teacher})
        baseline = next(c for c in cases if c["seed"] == seed and c["arm"] == ARMS[0])["checkpoints"][0]
        for case in [c for c in cases if c["seed"] == seed]:
            for mode in ("direct", "reason"):
                assert case["checkpoints"][0]["metrics"][mode]["answers"] == baseline["metrics"][mode]["answers"]
    rng = np.random.default_rng(701000)
    seed_draws = rng.integers(0, len(SEEDS), (3000, len(SEEDS)))
    strata = [np.array([i for i,p in enumerate(evaluation) if p["depth"] == d]) for d in DEPTHS]
    problem_draws = np.concatenate([rng.choice(indices, size=(3000, len(indices))) for indices in strata], axis=1)
    curves, bootstrap_means = {}, {}
    for arm in ARMS:
        chosen = [next(c for c in cases if c["arm"] == arm and c["seed"] == seed) for seed in SEEDS]
        curves[arm], bootstrap_means[arm] = {}, {}
        for mode in ("direct", "reason"):
            array = np.array([[cp["metrics"][mode]["correct"] for cp in case["checkpoints"]] for case in chosen])
            means, draws = [], []
            for j, budget in enumerate(CHECKPOINTS):
                values = array[:,j,:]
                boot = values[seed_draws[:,:,None], problem_draws[:,None,:]].mean(axis=(1,2))
                draws.append(boot)
                means.append({"budget": budget, "accuracy": float(values.mean()), "ci95": np.quantile(boot,[.025,.975]).tolist(),
                              "seed_accuracies": values.mean(axis=1).tolist(),
                              "mean_generation_tokens": float(np.mean([case["checkpoints"][j]["counters"]["rollout_prompt_tokens"] + case["checkpoints"][j]["counters"]["rollout_completion_tokens"] for case in chosen])),
                              "mean_training_tokens": float(np.mean([case["checkpoints"][j]["counters"]["training_input_tokens"] for case in chosen]))})
            curves[arm][mode], bootstrap_means[arm][mode] = means, draws
    contrasts = {}
    for mode in ("direct", "reason"):
        delta = bootstrap_means["on_policy_reasoning"][mode][-1] - bootstrap_means["frozen_reasoning"][mode][-1]
        contrasts[mode] = {"on_policy_minus_frozen": curves["on_policy_reasoning"][mode][-1]["accuracy"] - curves["frozen_reasoning"][mode][-1]["accuracy"],
                           "ci95": np.quantile(delta,[.025,.975]).tolist()}
    icl = []
    for budget in (4,16):
        for representation in ("answers", "traces"):
            icl.append({"budget": budget, "representation": representation,
                        "metrics": {mode: grade(output / "68000/icl" / f"{representation}_{mode}_n{budget}.jsonl", evaluation) for mode in ("direct", "reason")},
                        "usage": {mode: json.loads((output / "68000/icl" / f"{representation}_{mode}_n{budget}.usage.json").read_text()) for mode in ("direct", "reason")}})
    result = {"protocol": protocol, "cases": cases, "curves": curves, "contrasts": contrasts, "icl": icl,
              "audit": {"cases": len(cases), "checkpoints": sum(len(c["checkpoints"]) for c in cases),
                        "causal_target_and_counter_checks": True, "initial_teacher_draws_paired": True, "baseline_predictions_paired": True},
              "bootstrap": "3000 paired resamples of3optimization seeds and144questions, stratified by depth; task cohorts fixed. Three seeds limit precision."}
    save(output / "analysis.json", result)
    reports.mkdir(exist_ok=True)
    save(reports / "elicitation-consolidation-data.json", result)
    for mode in ("direct", "reason"):
        fig, ax = plt.subplots()
        for arm in ARMS:
            rows = curves[arm][mode]
            x = [r["budget"] for r in rows]
            y = [r["accuracy"] for r in rows]
            line, = ax.plot(x,y, marker="o", label=LABELS[arm])
            ax.fill_between(x, [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows], color=line.get_color(), alpha=.2)
        ax.set(xlabel="Practice presentations (96 fixed questions, up to 4 passes)", ylabel="Held-out accuracy",
               title="Short-answer consolidation" if mode == "direct" else "Reasoning after short-answer training", ylim=(0,1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports / f"elicitation-learning-{mode}.png")
        plt.close(fig)
    fig, ax = plt.subplots()
    for arm in ARMS:
        chosen = [c for c in cases if c["arm"] == arm]
        values = np.array([[r["plurality_accuracy"] for r in c["teacher"]] for c in chosen])
        ax.plot(np.arange(1,25)*16, values.mean(axis=0), label=LABELS[arm])
    ax.set(xlabel="Practice presentations", ylabel="Correctness of self-produced targets", ylim=(0,1), title="Does the model remain a useful teacher to itself?")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "elicitation-target-quality.png")
    plt.close(fig)
    print(json.dumps({"contrasts": contrasts, "endpoints": {arm:{mode:rows[-1] for mode,rows in value.items()} for arm,value in curves.items()},
                      "icl": [{"budget":r["budget"],"representation":r["representation"],"direct":r["metrics"]["direct"]["accuracy"],"reason":r["metrics"]["reason"]["accuracy"]} for r in icl]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("results/elicitation_consolidation"))
    parser.add_argument("--reports",type=Path,default=Path("reports"))
    args = parser.parse_args()
    analyze(args.output,args.reports)
