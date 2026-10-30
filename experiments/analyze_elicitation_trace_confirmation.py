"""Audit all six completed confirmation runs before computing paired seed intervals."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from analyze_elicitation_traces import analyze as audit_seed
from elicitation_task import DEPTHS
from replicate_elicitation_traces import ARMS, SEEDS, CHECKPOINTS
from run_elicitation_pilot import save


LABELS = {"frozen_reasoning":"Frozen reasoning source", "on_policy_reasoning":"Current reasoning source"}


def analyze(output,reports):
    for seed in SEEDS:
        for arm in ARMS: assert json.loads((output/str(seed)/arm/"complete.json").read_text())["counters"]["practice_presentations"] == 384
    analyses = [audit_seed(output/str(seed),None) for seed in SEEDS]
    cases = [case for a in analyses for case in a["cases"]]
    evaluation = json.loads((output/"evaluation.json").read_text())
    protocol = json.loads((output/"protocol.json").read_text())
    baseline = cases[0]["checkpoints"][0]["metrics"]
    for case in cases:
        for mode in ("direct","reason"): assert case["checkpoints"][0]["metrics"][mode]["answers"] == baseline[mode]["answers"]
    rng = np.random.default_rng(713000)
    seed_draws = rng.integers(0,len(SEEDS),(5000,len(SEEDS)))
    strata = [np.array([i for i,p in enumerate(evaluation) if p["depth"]==d]) for d in DEPTHS]
    question_draws = np.concatenate([rng.choice(s,size=(5000,len(s))) for s in strata],axis=1)
    curves,bootstraps,baseline_contrasts = {},{},{}
    for arm in ARMS:
        chosen = [next(c for c in cases if c["arm"]==arm and c["seed"]==seed) for seed in SEEDS]
        curves[arm],bootstraps[arm],baseline_contrasts[arm] = {},{},{}
        for mode in ("direct","reason"):
            rows,samples = [],[]
            for j,n in enumerate(CHECKPOINTS):
                cps = [c["checkpoints"][j] for c in chosen]
                values = np.array([cp["metrics"][mode]["correct"] for cp in cps])
                boot = values[seed_draws[:,:,None],question_draws[:,None,:]].mean(axis=(1,2))
                samples.append(boot)
                rows.append({"budget":n,"accuracy":float(values.mean()),"ci95":np.quantile(boot,[.025,.975]).tolist(),
                             "seed_accuracies":values.mean(axis=1).tolist(),
                             "completion":float(np.mean([cp["metrics"][mode]["completion"] for cp in cps])),
                             "mean_output_tokens":float(np.mean([cp["metrics"][mode]["mean_output_tokens"] for cp in cps])),
                             "generation_tokens":float(np.mean([cp["counters"]["rollout_prompt_tokens"]+cp["counters"]["rollout_completion_tokens"] for cp in cps])),
                             "training_tokens":float(np.mean([cp["counters"]["training_input_tokens"] for cp in cps])),
                             "supervised_tokens":float(np.mean([cp["counters"]["supervised_tokens"] for cp in cps])),
                             "practice_accuracy":float(np.mean([cp["practice"]["accuracy"] for cp in cps]))})
            curves[arm][mode],bootstraps[arm][mode] = rows,samples
            baseline_contrasts[arm][mode] = {"final_minus_baseline":rows[-1]["accuracy"]-rows[0]["accuracy"],
                                            "ci95":np.quantile(samples[-1]-samples[0],[.025,.975]).tolist()}
    contrasts = {}
    for mode in ("direct","reason"):
        contrasts[mode] = {"on_policy_minus_frozen":curves["on_policy_reasoning"][mode][-1]["accuracy"]-curves["frozen_reasoning"][mode][-1]["accuracy"],
                           "ci95":np.quantile(bootstraps["on_policy_reasoning"][mode][-1]-bootstraps["frozen_reasoning"][mode][-1],[.025,.975]).tolist(),
                           "seed_differences":(np.array(curves["on_policy_reasoning"][mode][-1]["seed_accuracies"])-np.array(curves["frozen_reasoning"][mode][-1]["seed_accuracies"])).tolist()}
    result={"protocol":protocol,"cases":cases,"curves":curves,"contrasts":contrasts,"baseline_contrasts":baseline_contrasts,
            "audit":{"cases":len(cases),"checkpoints":sum(len(c["checkpoints"]) for c in cases),
                     "all_case_audits_passed":True,"baseline_answers_match_across_seeds":True},
            "bootstrap":"5000 paired resamples of three optimization seeds and 144 questions, stratified by program depth. Three seeds and one shared fresh cohort limit uncertainty estimates. Intervals are pointwise; perfect scores can give degenerate bootstrap intervals.",
            "scope":"Fresh-data confirmation of the unchanged two-arm full-trace recipe, same architecture. Neither an independent-model replication nor an exponent estimate."}
    save(output/"analysis.json",result)
    save(reports/"elicitation-confirmation-data.json",result)
    for mode in ("direct","reason"):
        fig,ax=plt.subplots()
        for arm in ARMS:
            rows=curves[arm][mode]
            x=[r["budget"] for r in rows]
            line,=ax.plot(x,[r["accuracy"] for r in rows],marker="o",label=LABELS[arm])
            ax.fill_between(x,[r["ci95"][0] for r in rows],[r["ci95"][1] for r in rows],color=line.get_color(),alpha=.2)
        ax.set(xlabel="Practice presentations (96 fresh questions, 4 passes)",ylabel="Held-out accuracy",ylim=(0,1),
               title="Full-trace training: three-seed confirmation" if mode=="reason" else "Short answers: three-seed confirmation")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports/f"elicitation-confirmation-{mode}.png")
        plt.close(fig)
    fig,ax=plt.subplots()
    for arm in ARMS:
        chosen=[c for c in cases if c["arm"]==arm]
        values=np.array([[r["plurality_accuracy"] for r in c["teacher"]] for c in chosen])
        ax.plot(np.arange(1,25)*16,values.mean(axis=0),label=LABELS[arm])
    ax.set(xlabel="Practice presentations",ylabel="Correctness of self-produced targets",ylim=(0,1),title="Teacher quality in the confirmation")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports/"elicitation-confirmation-teacher.png")
    plt.close(fig)
    fig,ax=plt.subplots()
    for arm in ARMS:
        rows=curves[arm]["reason"]
        ax.plot([r["generation_tokens"]+r["training_tokens"] for r in rows],[r["accuracy"] for r in rows],marker="o",label=LABELS[arm])
    ax.set(xlabel="Practice generation + training input tokens (not FLOPs)",ylabel="Held-out reasoning accuracy",ylim=(0,1),title="Confirmation against measured token work")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports/"elicitation-confirmation-tokens.png")
    plt.close(fig)
    print(json.dumps({"contrasts":contrasts,"endpoints":{arm:{mode:rows[-1] for mode,rows in modes.items()} for arm,modes in curves.items()}},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("results/elicitation_trace_confirmation"))
    parser.add_argument("--reports",type=Path,default=Path("reports"))
    args=parser.parse_args()
    analyze(args.output,args.reports)
