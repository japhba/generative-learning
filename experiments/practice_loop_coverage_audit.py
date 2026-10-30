"""Independent finite-state checks of the Curie--Weiss coverage control."""

import json
import os
from pathlib import Path
import shutil

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import gammaln, logsumexp
from dotenv import load_dotenv


def model(d, a):
    k = np.arange(d + 1)
    m = 2 * k - d
    multiplicity = gammaln(d + 1) - gammaln(k + 1) - gammaln(d - k + 1)
    energy = a * a * m * m / (2 * d)
    logz = logsumexp(multiplicity + energy)
    return m, multiplicity, np.exp(multiplicity + energy - logz), logz


def main():
    load_dotenv(Path.home() / ".env")
    out = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_loop_coverage"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out / "audit_source.py")
    checks = []
    for d in [16, 64, 256, 1024]:
        m, mult, pstar, _ = model(d, 1.25)
        c = (m * m - d) / (d * (d - 1))
        observed = float(pstar @ c) + .001
        proposal_mean = np.sqrt(observed)
        plusp = (1 + proposal_mean) / 2
        k = (m + d) / 2
        logproposal = np.logaddexp(k * np.log(plusp) + (d-k) * np.log1p(-plusp), (d-k) * np.log(plusp) + k * np.log1p(-plusp)) - np.log(2)
        importance_errors = []
        gradient_errors = []
        stationarity_errors = []
        quadrature_errors = []
        for a in [.8, 1., 1.25, 1.6, 2.]:
            _, _, p, _ = model(d, a)
            target = p @ c
            for reference in [0., 1., None]:
                logq = mult + (reference * reference * m * m / (2*d) if reference is not None else logproposal)
                logq -= logsumexp(logq)
                logw = a * a * m * m / (2*d) - (reference * reference * m * m / (2*d) if reference is not None else logproposal)
                corrected = np.exp(logq + logw - logsumexp(logq + logw)) @ c
                importance_errors.append(abs(corrected-target))
            def nll(x):
                return model(d, x)[3] - .5*x*x*(1+(d-1)*observed)
            epsilon = 1e-5
            derivative = (nll(a+epsilon)-nll(a-epsilon))/(2*epsilon)
            analytic = (d-1)*a*(target-observed)
            gradient_errors.append(abs(derivative-analytic)/(1+abs(analytic)))
            moments = []
            for degree in [100, 200]:
                nodes, weights = hermgauss(degree)
                z = a*m[:, None]/np.sqrt(d) + np.sqrt(2)*nodes
                after_one_sweep = np.tanh(a*z/np.sqrt(d))**2 @ (weights/np.sqrt(np.pi))
                moments.append(float(p @ after_one_sweep))
            stationarity_errors.append(abs(moments[-1]-target))
            quadrature_errors.append(abs(moments[-1]-moments[0]))
        threshold = .8
        rate = .5*((1+threshold)*np.log(1+threshold)+(1-threshold)*np.log(1-threshold))
        tails = {}
        for a0 in [0., 1.]:
            exact = float(model(d,a0)[2][abs(m)/d >= threshold].sum())
            bound = min(1., (d+1)*np.exp(-d*(rate-a0*a0*threshold*threshold/2)))
            assert exact <= bound*(1+1e-12)
            tails[str(a0)] = {"exact_probability":exact,"type_count_upper_bound":float(bound)}
        checks.append({"dimension":d,"importance_expectation_max_error":max(importance_errors),"nll_gradient_relative_max_error":max(gradient_errors),"gibbs_invariant_correlation_max_error":max(stationarity_errors),"quadrature_100_200_max_error":max(quadrature_errors),"ordered_tail":tails})
    assert max(c["importance_expectation_max_error"] for c in checks) < 1e-12
    assert max(c["nll_gradient_relative_max_error"] for c in checks) < 2e-7
    assert max(c["gibbs_invariant_correlation_max_error"] for c in checks) < 1e-8
    result = {"checks":checks,"scope":"Population importance identities and fixed-parameter Gibbs moment invariance; no finite-batch unbiasedness or moving-chain mixing theorem claimed."}
    (out / "audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
