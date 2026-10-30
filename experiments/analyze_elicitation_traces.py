"""Audit the completed trace pilot; labels are used only in this offline analysis."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from analyze_elicitation_consolidation import grade
from elicitation_answer import parse_answer
from elicitation_task import DEPTHS, consensus, states
from run_elicitation_pilot import digest, save
from train_elicitation import counters, read_rows
from train_elicitation_traces import CHECKPOINTS


LABELS = {"frozen_reasoning": "Frozen reasoning source", "on_policy_reasoning": "Current reasoning source",
          "raw_reasoning_self": "Unfiltered reasoning self-imitation"}


def analyze(output, reports):
    protocol = json.loads((output / "protocol.json").read_text())
    arms, sources = tuple(protocol["arms"]), tuple(protocol["source_sha256"])
    # Require all arms to finish before opening any file for correctness scoring.
    complete = {arm: json.loads((output / arm / "complete.json").read_text()) for arm in arms}
    assert protocol["source_sha256"] == {name: digest(Path(__file__).with_name(name)) for name in sources}
    assert protocol["source_sha256"] == {name: digest(output / "source" / name) for name in sources}
    assert protocol["train_sha256"] == digest(output / "train.json")
    assert protocol["evaluation_sha256"] == digest(output / "evaluation.json")
    train = json.loads((output / "train.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())
    order = json.loads((output / "order.json").read_text())
    for epoch in range(4): assert sorted(order[96*epoch:96*(epoch+1)]) == list(range(96))
    sequence = [train[i] for i in order]
    frozen = read_rows(output / "frozen_candidates/round_0.jsonl")
    live = read_rows(output / "on_policy_reasoning/round_0.jsonl")
    assert [r["token_ids"] for r in frozen] == [r["token_ids"] for r in live]
    cases = []
    for arm in arms:
        directory = output / arm
        updates = read_rows(directory / "updates.jsonl")
        assert len(updates) == 24
        count, teacher = counters(), []
        for step, update in enumerate(updates):
            assert update["step"] == step and update["budget"] == (step+1)*16
            expected = sequence[step*16:(step+1)*16]
            assert [s["id"] for s in update["selected"]] == [p["id"] for p in expected]
            path = Path(update["candidate_path"])
            expected_path = output / ("frozen_candidates" if arm == "frozen_reasoning" else arm) / f"round_{step}.jsonl"
            assert path.resolve() == expected_path.resolve()
            assert digest(path) == update["candidate_sha256"]
            candidates = [{**r, "answer": parse_answer(r["text"])} for r in read_rows(path)]
            size = 1 if arm == "raw_reasoning_self" else 4
            assert len(candidates) == 16*size
            voted, individual, valid = [], [], []
            for i, (selected, problem) in enumerate(zip(update["selected"], expected, strict=True)):
                group = candidates[i*size:(i+1)*size]
                assert [r["id"] for r in group] == [problem["id"]]*size
                vote = consensus(group)
                assert vote == selected["vote"]
                chosen = 0 if arm == "raw_reasoning_self" else vote["selected_index"]
                assert selected["selected_index"] == chosen
                assert selected["target_ids"] == ([] if chosen is None else group[chosen]["token_ids"])
                assert selected["target_text"] == ("" if chosen is None else group[chosen]["text"])
                truth = states(problem)[-1]
                voted.append(int(vote["answer"] == truth))
                individual.extend(int(r["ended"] and r["answer"] == truth) for r in group)
                valid.append(bool(selected["target_ids"]))
                count["supervised_tokens"] += len(selected["target_ids"])
            usage = json.loads(path.with_suffix(".usage.json").read_text())
            assert usage["responses"] == len(candidates)
            assert usage["completion_tokens"] == sum(r["completion_tokens"] for r in candidates)
            count["practice_presentations"] += 16
            count["training_presentations"] += 16
            count["optimizer_updates"] += 1
            count["valid_targets"] += sum(valid)
            for key in ("responses", "prompt_tokens", "completion_tokens", "padded_prompt_tokens", "padded_completion_tokens"):
                count["rollout_"+key] += usage[key]
            for key in ("practice_presentations", "training_presentations", "optimizer_updates", "valid_targets", "supervised_tokens",
                        "rollout_responses", "rollout_prompt_tokens", "rollout_completion_tokens", "rollout_padded_prompt_tokens", "rollout_padded_completion_tokens"):
                assert count[key] == update["counters"][key], (arm, step, key)
            assert np.isfinite(update["loss"])
            teacher.append({"budget": (step+1)*16, "plurality_accuracy": float(np.mean(voted)),
                            "raw_sample_accuracy": float(np.mean(individual)), "valid_fraction": float(np.mean(valid)),
                            "loss": update["loss"], "counters": update["counters"]})
        assert complete[arm]["counters"] == updates[-1]["counters"]
        records = read_rows(directory / "checkpoints.jsonl")
        assert [r["budget"] for r in records] == list(CHECKPOINTS)
        checkpoints = []
        for record in records:
            n = record["budget"]
            assert record["counters"] == (counters() if n == 0 else updates[n//16-1]["counters"])
            checkpoints.append({"budget": n, "counters": record["counters"], "evaluation_usage": record["evaluation_usage"],
                                "metrics": {mode: grade(directory / f"eval_{mode}_n{n}.jsonl", evaluation) for mode in ("direct", "reason")},
                                "practice": grade(directory / f"practice_reason_n{n}.jsonl", train)})
        cases.append({"seed": protocol["seed"], "arm": arm, "checkpoints": checkpoints, "teacher": teacher})
    for mode in ("direct", "reason"):
        baseline_rows = read_rows(output / arms[0] / f"eval_{mode}_n0.jsonl")
        for arm in arms[1:]:
            other = read_rows(output / arm / f"eval_{mode}_n0.jsonl")
            assert [r["token_ids"] for r in baseline_rows] == [r["token_ids"] for r in other]
    rng = np.random.default_rng(702000)
    strata = [np.array([i for i,p in enumerate(evaluation) if p["depth"] == d]) for d in DEPTHS]
    draws = np.concatenate([rng.choice(s, size=(5000,len(s))) for s in strata], axis=1)
    curves, sampled, contrasts = {}, {}, {}
    for case in cases:
        arm = case["arm"]
        curves[arm], sampled[arm], contrasts[arm] = {}, {}, {}
        for mode in ("direct", "reason"):
            rows, bootstraps = [], []
            for cp in case["checkpoints"]:
                metrics, count = cp["metrics"][mode], cp["counters"]
                values = np.array(metrics["correct"])
                boot = values[draws].mean(axis=1)
                bootstraps.append(boot)
                rows.append({"budget": cp["budget"], "accuracy": metrics["accuracy"], "ci95": np.quantile(boot,[.025,.975]).tolist(),
                             "completion": metrics["completion"], "parsed": metrics["parsed"], "mean_output_tokens": metrics["mean_output_tokens"],
                             "generation_tokens": count["rollout_prompt_tokens"]+count["rollout_completion_tokens"],
                             "training_tokens": count["training_input_tokens"], "supervised_tokens": count["supervised_tokens"],
                             "practice_accuracy": cp["practice"]["accuracy"]})
            curves[arm][mode], sampled[arm][mode] = rows, bootstraps
            contrasts[arm][mode] = {"final_minus_baseline": rows[-1]["accuracy"]-rows[0]["accuracy"],
                                    "ci95": np.quantile(bootstraps[-1]-bootstraps[0],[.025,.975]).tolist()}
    comparison = {}
    for mode in ("direct", "reason"):
        delta = sampled["on_policy_reasoning"][mode][-1]-sampled["frozen_reasoning"][mode][-1]
        comparison[mode] = {"on_policy_minus_frozen": curves["on_policy_reasoning"][mode][-1]["accuracy"]-curves["frozen_reasoning"][mode][-1]["accuracy"],
                            "ci95": np.quantile(delta,[.025,.975]).tolist()}
    result = {"protocol": protocol, "cases": cases, "curves": curves, "baseline_contrasts": contrasts, "contrasts": comparison,
              "audit": {"cases": len(cases), "checkpoints": sum(len(c["checkpoints"]) for c in cases),
                        "all_targets_actual_sampled_tokens": True, "initial_teacher_draws_paired": True, "baseline_token_sequences_paired": True,
                        "causal_source_and_counter_checks": True, "source_and_data_hashes": True},
              "bootstrap": "5000 paired question resamples, stratified by depth. One optimization seed: intervals do not include training-seed or cohort uncertainty.",
              "scope": "Exploratory revised recipe. No test-dependent changes within this protocol. No exponent fitting or seed replication claim."}
    save(output / "analysis.json", result)
    if reports is None: return result
    save(reports / "elicitation-traces-data.json", result)
    for mode in ("direct", "reason"):
        fig, ax = plt.subplots()
        for arm in arms:
            rows = curves[arm][mode]
            x = [r["budget"] for r in rows]
            line, = ax.plot(x,[r["accuracy"] for r in rows],marker="o",label=LABELS[arm])
            ax.fill_between(x,[r["ci95"][0] for r in rows],[r["ci95"][1] for r in rows],color=line.get_color(),alpha=.2)
        ax.set(xlabel="Practice presentations (96 questions, up to 4 passes)", ylabel="Held-out accuracy", ylim=(0,1),
               title="Training on full self-generated reasoning" if mode == "reason" else "Short answers after full-trace training")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports / f"elicitation-traces-{mode}.png")
        plt.close(fig)
    fig, ax = plt.subplots()
    for arm in arms:
        rows = curves[arm]["reason"]
        ax.plot([r["generation_tokens"]+r["training_tokens"] for r in rows],[r["accuracy"] for r in rows],marker="o",label=LABELS[arm])
    ax.set(xlabel="Practice generation + training input tokens (not FLOPs)", ylabel="Held-out reasoning accuracy", ylim=(0,1),
           title="Learning against measured token work")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "elicitation-traces-tokens.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    for case in cases:
        ax.plot([r["budget"] for r in case["teacher"]],[r["plurality_accuracy"] for r in case["teacher"]],label=LABELS[case["arm"]])
    ax.set(xlabel="Practice presentations", ylabel="Correctness of self-produced targets", ylim=(0,1), title="Teacher quality with full-trace training")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "elicitation-traces-teacher.png")
    plt.close(fig)
    print(json.dumps({"contrasts": comparison, "baseline_contrasts": contrasts,
                      "endpoints": {arm:{mode:rows[-1] for mode,rows in modes.items()} for arm,modes in curves.items()}},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("results/elicitation_traces"))
    parser.add_argument("--reports",type=Path,default=Path("reports"))
    args = parser.parse_args()
    analyze(args.output,args.reports)
