"""Per-level likelihood progress for the sigmoid belief net runs of practice_sbn_lead_info.py, the analogue of the RHM's
pi_h = (log N_(h-1) - NLL) / (log N_(h-1) - log N_h). Reference models that know the world up to a level and nothing above:
  l_1: the world's rendering x | h_1, with the h_1 units independent (their true marginals);
  l_2: the world's rendering and h_1 | h_2, with the h_2 units independent;  l_3 = H(x).
pi_2 = (l_1 - NLL) / (l_1 - l_2), pi_3 = (l_2 - NLL) / (l_2 - H), NLL = H + KL(world || learner) on the run's evaluation set.
Compared with reading as information (read_info / the ideal reader's, needs ref_info written by the analysis in that run dir).
Usage: python practice_sbn_lead_levels.py RUN_DIR"""

import json
from pathlib import Path
import sys

import numpy as np

from practice_sbn_lead import configs, logpdf, lsig, make_world


def nll_with_prior(G, X, lp_h1): return -np.mean(np.logaddexp.reduce(G.loglik(X, configs(G.K[0])) + lp_h1[None], axis=1))


def indep(Hs, p):  # log prior of configurations Hs under independent units with the marginals of distribution p over Hs
    m = p @ ((Hs + 1) / 2); return ((Hs + 1) / 2 * np.log(m) + (1 - Hs) / 2 * np.log(1 - m)).sum(1)


def main():
    d = json.load(open(Path(sys.argv[1]) / "results.json")); a = d["args"]; rows = {}
    for r in d["results"]:
        K, D, w = r["K"], r["D"], r["world"]; rng = np.random.default_rng([a["seed"], w, D, *K]); G = make_world(K, D, a["sigma"], a["beta"], rng)
        _, Xt = G.sample(a["n_eval"], rng); H = -np.mean(logpdf(G, Xt)); Hs, lp = G.tables()
        l1 = nll_with_prior(G, Xt, indep(Hs[0], np.exp(np.logaddexp.reduce(lp.reshape(-1, lp.shape[-1]), axis=0))))
        lc = lsig(2 * (Hs[1] @ G.V[0].T + G.a[0])[:, None, :] * Hs[0][None]).sum(2)
        l2 = nll_with_prior(G, Xt, np.logaddexp.reduce(indep(Hs[1], np.exp(np.logaddexp.reduce(lp, axis=(0, 2))))[:, None] + lc, axis=0))
        t = np.array([c["step"] for c in r["curve"]]); nll = np.array([c["kl"] for c in r["curve"]]) + H
        onset = lambda p: (lambda i: t[i[0]] if len(i) else np.nan)(np.flatnonzero(p >= 0.5))
        rd = lambda l: onset(np.array([c["read_info"][l] for c in r["curve"]]) / r["ref_info"][l])
        rows.setdefault(r["name"], []).append([l1 - l2, l2 - H, rd(0), onset(np.clip((l1 - nll) / (l1 - l2), 0, 1)), rd(1), onset(np.clip((l2 - nll) / (l2 - H), 0, 1))])
    for name, R in rows.items():
        R = np.array(R, float); med = np.nanmedian(R, 0); ok3 = np.isfinite(R[:, 5] / R[:, 4])
        print(f"{name}: level 2 worth {med[0]:.2f} nats, level 3 {med[1]:.2f} | reads layer 1 at {med[2]:.0f}, likelihood has level 2 at {med[3]:.0f} "
              f"(lag x{np.nanmedian(R[:, 3] / R[:, 2]):.1f}, {np.sum(R[:, 3] > R[:, 2])}/{len(R)} reading first) | reads layer 2 at {med[4]:.0f}, likelihood has level 3 at {med[5]:.0f} "
              f"(lag x{np.nanmedian(R[:, 5] / R[:, 4]):.1f}, {np.sum(R[ok3, 5] > R[ok3, 4])}/{ok3.sum()} reading first)")


if __name__ == "__main__": main()
