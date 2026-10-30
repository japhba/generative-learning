"""Unfitted population overlap laws for the independent Bernoulli modes."""

import json
import itertools

import matplotlib.pyplot as plt
import numpy as np

from run_elicitation_pilot import ROOT, save
from train_bestn import stamp


def phi(r): return r + 1.25 * np.log(r) - .5 / r - .25 * np.log(r + 2)


def misalignment(weights):
    mean = weights.mean(-1); variance = weights.var(-1); rms = np.sqrt(mean * mean + variance)
    return variance / (rms * (rms + mean))


def main():
    root = ROOT / "results/scaling_r5s/toy"; data = json.loads((root / "results.json").read_text())
    p0 = np.array(data["prediction"]["initial_success"]); q0 = p0 * (2 - p0); m = len(p0)
    initial_phi = phi(p0 / (1 - p0)); variance_phi = np.var(initial_phi)
    w0 = np.log(p0 / (1 - p0)); visible = np.array(list(itertools.product((0, 1), repeat=m)))
    enumerated_z = float(np.sum(1 + np.exp(visible @ w0))); formula_z = float(2 ** m + np.prod(1 + np.exp(w0)))
    assert abs(enumerated_z - formula_z) < 1e-10
    np.testing.assert_allclose(np.exp(w0) / (1 + np.exp(w0)), p0, atol=1e-14)
    result = {"checked_at_uk": stamp(), "status": "Analytic reduction checked retrospectively on the already stored population trajectories; no coefficient fit.",
              "rbm_partition_check": {"enumerated": enumerated_z, "formula": formula_z},
              "current_variance_amplitude": float(m * m * variance_phi), "frozen_variance_limit": float(np.var(np.log(q0))), "sources": {}}
    fig, ax = plt.subplots()
    for source in ("current", "frozen"):
        records = data["population"][source]
        t = np.array([r["time"] for r in records]); w = np.array([r["weights"] for r in records]); keep = t >= 10
        observed = misalignment(w)
        line, = ax.loglog(t[keep], observed[keep], label=f"{source}: population")
        late = t >= 1e3
        if source == "current":
            mean = np.log(t[late] / m); variance = m * m * variance_phi / t[late] ** 2
            expected = variance / (2 * mean * mean)
            differences = phi(np.exp(w)) - initial_phi
            invariant_error = float(np.max(np.ptp(differences, axis=-1)))
        else:
            mean = np.log(t[late] / q0.sum()) + np.mean(np.log(q0)); variance = np.var(np.log(q0))
            rms = np.sqrt(mean * mean + variance); expected = variance / (rms * (rms + mean))
            invariant_error = float(np.max(np.abs(np.exp(w) + w - (np.exp(w[0]) + w[0]) - t[:, None] * q0[None, :] / q0.sum())))
        ax.loglog(t[late], expected, "--", color=line.get_color(), label=f"{source}: unfitted asymptote")
        finite = data["finite"][source]; ft = finite["eta"] * np.array([r["step"] for r in finite["records"]]); fw = np.array([r["weights"] for r in finite["records"]])
        fk = ft >= 10; ax.loglog(ft[fk], misalignment(fw).mean(-1)[fk], "o", color=line.get_color(), label=f"{source}: finite batches")
        result["sources"][source] = {"maximum_integral_identity_error": invariant_error, "endpoint_misalignment": float(observed[-1]),
                                      "endpoint_predicted_misalignment": float(expected[-1]), "endpoint_over_prediction": float(observed[-1] / expected[-1]),
                                      "finite_endpoint_misalignment": float(misalignment(fw[-1]).mean())}
        assert invariant_error < 1e-3
        assert abs(observed[-1] / expected[-1] - 1) < .01
    ax.set(xlabel="Population / finite-batch SGD time", ylabel="1 − cosine with equal positive mode weights", title="Refreshing changes how mode weights become balanced")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-verified-toy-overlap.png"); plt.close(fig)
    save(root / "weight_laws.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
