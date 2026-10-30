"""Whole-replica uncertainty and unfitted asymptotic comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_elicitation_pilot import ROOT, save
from scaling_theory import asymptotic_constant, normal_cdf, phi


def analyze(stage):
    root = ROOT/"results/scaling_s4g"/stage
    cases, rng = [], np.random.default_rng(97199)
    fig, ax = plt.subplots()
    for path in sorted(root.glob("d*.json")):
        row = json.loads(path.read_text())
        if row["source"] != "current": continue
        records = row["records"]
        steps = np.array([r["step"] for r in records])
        times = steps*row["eta"]
        errors = np.array([r["errors"] for r in records])-normal_cdf(-1.5)
        assert np.all(np.isfinite(errors)) and np.all(errors>0)
        tail = steps >= steps[-1]/8
        x = np.log(times[tail]); centered=x-x.mean()
        mean = errors.mean(axis=1)
        beta = float(-centered@np.log(mean[tail])/(centered@centered))
        boot = []
        for _ in range(2000):
            indices = rng.integers(errors.shape[1],size=errors.shape[1])
            boot.append(float(-centered@np.log(errors[tail][:,indices].mean(axis=1))/(centered@centered)))
        dimension, batch, eta = [row[k] for k in ("dimension","batch","eta")]
        lam = (dimension-1)*eta*float(phi(1.5))/batch
        predicted_beta = .5 if dimension==1 else 1/3
        coefficient = np.sqrt(asymptotic_constant()/8) if dimension==1 else lam**(2/3)/(2*3**(1/3))
        predicted = coefficient*times[1:]**(-predicted_beta)
        cases.append({"path":str(path),"dimension":dimension,"batch":batch,"eta":eta,"last_step":int(steps[-1]),
                      "beta":beta,"beta_ci95":np.quantile(boot,[.025,.975]).tolist(),"asymptotic_beta":predicted_beta,
                      "unfitted_coefficient":float(coefficient),"last_error_over_asymptote":float(mean[-1]/predicted[-1]),
                      "last_excess_error":float(mean[-1]),"last_norm":records[-1]["norm_mean"],"last_overlap":records[-1]["overlap_mean"]})
        line,=ax.loglog(times[1:],mean[1:],label=f"{dimension} weights, batch {batch}, rate {eta}")
        ax.loglog(times[1:],predicted,"--",color=line.get_color())
    ax.set(xlabel="Learning rate × SGD updates",ylabel="Sampled error above the known Bayes floor")
    ax.legend()
    fig.tight_layout()
    filename="scaling-finite-confirmation.png" if stage=="finite_confirmation" else "scaling-finite-sgd.png"
    fig.savefig(ROOT/"reports"/filename)
    plt.close(fig)
    save(root/"analysis.json",{"fit_window":"Final factor8 in steps; descriptive finite-range exponent.",
         "interval":"2000 whole-replica bootstrap resamples, preserving time correlations.",
         "dashed_lines":"Predictions fixed before sampling; no fitted amplitude or time constant.","cases":cases})
    print(json.dumps(cases,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("finite_sgd","finite_confirmation"),default="finite_sgd")
    analyze(parser.parse_args().stage)
