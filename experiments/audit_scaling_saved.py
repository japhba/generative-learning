"""Retrospective shape/ceiling ambiguity in the completed b2k trajectories."""

import json

import matplotlib.pyplot as plt
import numpy as np

from run_elicitation_pilot import ROOT, save


def family_grid(x, y, family):
    tau = np.geomspace(.1, 1e6, 241)
    beta = np.geomspace(.05, 8, 121) if family == "power" else np.array([1.])
    b, t = np.meshgrid(beta, tau, indexing="ij")
    b, t = b.ravel(), t.ravel()
    q = (1 + x[None] / t[:, None])**(-b[:, None]) if family == "power" else np.exp(-x[None] / t[:, None])
    a = 1 - q
    floor = np.clip(np.sum(a * (y[None] - y[0] * q), axis=1) / np.sum(a**2, axis=1), 0, y[0])
    curves = floor[:, None] + (y[0] - floor[:, None]) * q
    mse = np.mean((curves - y[None])**2, axis=1)
    index = int(np.argmin(mse))
    near = np.sqrt(mse) <= np.sqrt(mse[index]) + .005
    return {"family": family, "beta": float(b[index]), "tau": float(t[index]), "floor": float(floor[index]),
            "rmse": float(np.sqrt(mse[index])), "near_optimal_beta_range": [float(b[near].min()), float(b[near].max())],
            "near_optimal_floor_range": [float(floor[near].min()), float(floor[near].max())]}, curves[index]


def main():
    output = ROOT / "results/scaling_s4g/saved_audit"
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    fig, ax = plt.subplots()
    for source in ("current", "frozen"):
        histories = [json.loads((ROOT / f"results/bestn_b2k/training/{seed}/{source}/complete.json").read_text()) for seed in (86001, 86002, 86003)]
        common = sorted(set.intersection(*({r["step"] for r in h["evaluations"]} for h in histories)))
        x = 32 * np.array(common, dtype=float)
        errors = np.array([[1 - next(r["accuracy"] for r in h["evaluations"] if r["step"] == step) for step in common] for h in histories])
        y = errors.mean(axis=0)
        line, = ax.plot(x, y, "o", label=f"{source}: saved mean error")
        fits = {}
        for family, style in (("power", "-"), ("exponential", "--")):
            fit, curve = family_grid(x, y, family)
            fits[family] = fit
            ax.plot(x, curve, style, color=line.get_color(), label=f"{source}: {family} + floor")
        records[source] = {"candidates": x.tolist(), "errors_by_seed": errors.tolist(), "fits": fits}
    ax.set(xlabel="Training candidate rollouts", ylabel="Held-out error", title="Existing arithmetic runs: power/floor ambiguity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports/scaling-saved-audit.png")
    plt.close(fig)
    save(output / "analysis.json", {"status": "Retrospective descriptive fits; no new scaling confirmation.",
         "near_optimal_definition": "Fits within0.5percentage-point RMSE of the best grid fit; a sensitivity set, not a confidence interval.",
         "curve": "error=floor+(initial_error-floor)*decay(time/tau); shifted power or exponential; initial error fixed to observed baseline.",
         "results": records})
    print(json.dumps({source: row["fits"] for source, row in records.items()}, indent=2))


if __name__ == "__main__": main()
