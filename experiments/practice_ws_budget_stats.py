"""Paired comparisons for practice_ws_budget.py (fixed before the main run was read): differences in population KL,
paired over worlds, 95% bootstrap intervals of the mean, and the number of worlds in which the first arm is lower.

  generation helps wake-sleep      ws(x)       - ws(x=1)        for x = 0.9, 0.5
  generation helps the baseline    ws_teach(x) - ws_teach(x=1)  for x = 0.9, 0.5
  baseline without generation      ws_teach(x=1) - ws(x=0.9)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from practice_ws_budget_plot import load

COMPARE = [('ws', 0.9, 'ws', 1.0), ('ws', 0.5, 'ws', 1.0), ('ws_teach', 0.9, 'ws_teach', 1.0), ('ws_teach', 0.5, 'ws_teach', 1.0), ('ws_teach', 1.0, 'ws', 0.9)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--Ps', default='8192,65536,524288,2097152'); a = ap.parse_args()
    d, xs, data = load(a.run); xs = np.array(xs); rng = np.random.default_rng(0); rows = []
    for prior in d['curves']:
        P = data[prior, 'ws']['P']
        for Pv in [int(p) for p in a.Ps.split(',')]:
            c = int(np.flatnonzero(P == Pv)[0])
            for l1, x1, l2, x2 in COMPARE:
                k1, k2 = (int(np.flatnonzero(np.isclose(xs, x))[0]) for x in (x1, x2))
                diff = data[prior, l1]['kl'][k1, :, c] - data[prior, l2]['kl'][k2, :, c]
                boot = rng.choice(diff, (10000, len(diff))).mean(1)
                rows.append(dict(prior=prior, P=Pv, a=f'{l1}@{x1:g}', b=f'{l2}@{x2:g}', kl_a=float(data[prior, l1]['kl'][k1, :, c].mean()), kl_b=float(data[prior, l2]['kl'][k2, :, c].mean()),
                                 diff=float(diff.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)), a_lower=int((diff < 0).sum()), n=len(diff)))
                r = rows[-1]; print(f"{prior:5s} P=2^{int(np.log2(Pv)):2d} {r['a']:13s} - {r['b']:13s}: {r['kl_a']:.3f} - {r['kl_b']:.3f} = {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  lower in {r['a_lower']}/{r['n']}")
    json.dump(rows, open(Path(a.run) / 'paired.json', 'w'), indent=1)


if __name__ == '__main__': main()
