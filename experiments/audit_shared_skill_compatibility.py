"""Check an exact consequence of the one-skill toy using existing LLM monitors."""

import json

import numpy as np

from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def main():
    root = ROOT / "results/scaling_r5s/confirmation"
    protocol = json.loads((root / "protocol.json").read_text())
    cohort = json.loads((root / "monitor.json").read_text()); labels = np.array([r["label"] for r in cohort])
    rows, files = [], []
    for cell in protocol["cells"]:
        path = root / str(cell["seed"]) / cell["name"] / "complete.json"
        history = json.loads(path.read_text()); files.append({"path": str(path), "sha256": digest(path)})
        initial = history["evaluations"][0]["records"]
        assert [r["id"] for r in initial] == [r["id"] for r in cohort]
        lp0 = np.array([r["correct_logprob"] for r in initial])
        for step in (128, 512):
            records = next(r for r in history["evaluations"] if r["step"] == step)["records"]
            assert [r["id"] for r in records] == [r["id"] for r in initial]
            lp = np.array([r["correct_logprob"] for r in records])
            finite = (lp < 0) & (lp0 < 0)
            gain = lp[finite] - np.log(-np.expm1(lp[finite])) - lp0[finite] + np.log(-np.expm1(lp0[finite]))
            rows.append({"seed": cell["seed"], "source": cell["source"], "step": step, "fraction_improved": float(np.mean(lp > lp0)),
                         "initial_vs_trained_probability_rank_correlation": float(np.corrcoef(ranks(lp0), ranks(lp))[0, 1]),
                         "finite_logit_pairs": int(finite.sum()), "logit_improvement_q10_q50_q90": np.quantile(gain, [.1, .5, .9]).tolist(),
                         "within_answer_class_rank_correlations": {label: float(np.corrcoef(ranks(lp0[labels == label]), ranks(lp[labels == label]))[0, 1]) for label in ("0", "1", "2")}})
    result = {"created_at_uk": stamp(), "scope": "Post-hoc descriptive check on already used monitoring questions, with no new training, evaluations or test-cohort access.",
              "prediction": "A literal one-skill additive-logit model preserves every question's probability rank and has a common logit improvement. This prediction fails in the LLM; it does not rule out multiple shared skills or establish a causal mechanism.",
              "probability_saturation": "Exact probabilities of one are omitted only from logit-change summaries, with finite-pair counts reported. Rank correlations retain every question.",
              "task": protocol["cell"], "questions": len(cohort), "rows": rows, "source_files": files,
              "analysis_sha256": digest(ROOT / "experiments/audit_shared_skill_compatibility.py")}
    save(ROOT / "results/scaling_r5s/shared_skill_theory/llm_compatibility.json", result)
    print(json.dumps({"questions": len(cohort), "step_512_rank_correlations": {source: [r["initial_vs_trained_probability_rank_correlation"] for r in rows if r["source"] == source and r["step"] == 512] for source in ("current", "frozen")}}, indent=2))


if __name__ == "__main__": main()
