"""Inspect finite toy replicas and render the predicted amplitude improvement."""

import json

import matplotlib.pyplot as plt
import numpy as np

from run_elicitation_pilot import ROOT, save


def main():
    root = ROOT / "results/scaling_r5s/toy"
    data = json.loads((root / "results.json").read_text())
    initial = np.array(data["prediction"]["initial_success"]); q0 = initial * (2 - initial)
    rng = np.random.default_rng(197113); analysis = {}
    fig, ax = plt.subplots()
    for source in ("current", "frozen"):
        records = data["population"][source]
        t = np.array([r["time"] for r in records]); error = np.array([r["error"] for r in records]); keep = t >= 10
        line, = ax.semilogx(t[keep], t[keep] * error[keep], label=f"{source}: population")
        amplitude = data["prediction"][f"{source}_amplitude"]
        ax.axhline(amplitude, linestyle="--", color=line.get_color())
        finite = data["finite"][source]; rows = finite["records"]
        steps = np.array([r["step"] for r in rows]); errors = np.array([r["error"] for r in rows]); time = steps * finite["eta"]
        keep = time >= 10
        ax.semilogx(time[keep], time[keep] * errors[keep].mean(-1), "o", color=line.get_color(), label=f"{source}: finite batches")
        selected_loss = []
        for row in rows:
            p = 1 / (1 + np.exp(-np.array(row["weights"])))
            q = p * (2 - p) if source == "current" else np.broadcast_to(q0, p.shape)
            selected_loss.append(float(np.mean(np.sum(-np.log(p) * q, axis=-1) / q.sum(-1))))
        audit = []
        for index, step in enumerate(steps):
            if step < 64: continue
            indices = np.arange(index - 3, index + 1)
            slopes = [float(np.polyfit(np.log(steps[indices]), np.log(values[indices]), 1)[0]) for values in (errors.mean(-1), np.array(selected_loss))]
            audit.append({"step": int(step), "log_error_slope": slopes[0], "log_expected_accepted_nll_slope": slopes[1], "would_halt": any(abs(s) < .02 for s in slopes)})
        keep = steps >= finite["steps"] / 8
        samples = []
        for _ in range(2000):
            ids = rng.integers(finite["replicas"], size=finite["replicas"])
            values = errors[:, ids].mean(-1)
            samples.append(-np.polyfit(np.log(steps[keep]), np.log(values[keep]), 1)[0])
        analysis[source] = {"beta": float(-np.polyfit(np.log(steps[keep]), np.log(errors[keep].mean(-1)), 1)[0]),
                            "beta_ci95": np.quantile(samples, [.025, .975]).tolist(),
                            "endpoint_amplitude_ratio": float(errors[-1].mean() * time[-1] / amplitude),
                            "stationarity_audit": audit, "any_stationary_check": any(r["would_halt"] for r in audit)}
        assert not analysis[source]["any_stationary_check"]
    ax.set(xlabel="SGD time (learning rate × updates)", ylabel="Time × mean error", title="Same inverse-budget exponent, different error amplitudes")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-verified-toy-amplitudes.png"); plt.close(fig)
    analysis["audit_status"] = "The finite toy was a short fixed-horizon simulation; this retrospective audit verifies that none of its recorded eligible checkpoints had stationary mean error or expected accepted-target loss."
    save(root / "analysis.json", analysis)
    print(json.dumps({s: {k: v for k, v in analysis[s].items() if k != "stationarity_audit"} for s in ("current", "frozen")}, indent=2))


if __name__ == "__main__": main()
