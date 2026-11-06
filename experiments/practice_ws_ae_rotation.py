"""The rotation plot of toy model 2 (section 2, two causes in the plane) with the autoencoder baseline added.

Python port of toy_model_2_ws_sim.js: z uniform on {-1,+1}^2, x = z + sigma eps (A = I), 256 stored samples, the generator
starts turned by 30 degrees with its recognizer at the sleep fixed point. Per update every learner gets two batches of 32:
  ws      a wake batch (stored samples) and a sleep batch (own dreams)
  frozen  two wake batches, recognizer never changes
  ae      two autoencoder batches on stored samples: binary autoencoder with the expected squared error under the factorial
          code (practice_disentangle.ae_step 'bae'), recognizer rate scaled by s (0.05 in the report's baseline)
Angles are those of the nearest rotation to the generator's columns, folded to +-45 degrees (as on the page).
"""

import json
from pathlib import Path

import numpy as np

OUT = Path('/var/tmp/jbauer/generative_practice_disentangle/ae_floor/two_cause.json')
LEARNERS = {'ws': {}, 'frozen': {}, 'ae': {'s': 0.05}, 'ae_fast': {'s': 1.0}}


def rot(t): return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def angle(M):
    M = M.copy()
    if np.linalg.det(M) < 0: M[:, 1] *= -1
    t = np.degrees(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])); return ((t + 45) % 90 + 90) % 90 - 45


def simulate(learner, seed, theta0=30, sigma=0.5, T=3000, lr=0.02, sleep_scale=10, B=32, n=256, snaps=(0, 250, 500, 1000, 3000)):
    rw = np.random.default_rng(1000 + seed); Zw = np.where(rw.random((n, 2)) < 0.5, -1.0, 1.0); X = Zw + sigma * rw.standard_normal((n, 2))
    W = rot(np.radians(theta0)); c = np.zeros(2); R = W.T / sigma ** 2; b = np.zeros(2); r = np.random.default_rng(5000 + seed)
    s = LEARNERS[learner].get('s'); tr, sn = [], {}
    def wake():
        nonlocal W, c
        x = X[r.integers(0, n, B)]; z = np.where(r.random((B, 2)) < 1 / (1 + np.exp(-2 * (x @ R.T + b))), 1.0, -1.0)
        E = x - z @ W.T - c; W = W + lr * E.T @ z / B; c = c + lr * E.mean(0)
    def ae():
        nonlocal W, c, R, b
        x = X[r.integers(0, n, B)]; m = np.tanh(x @ R.T + b); var = 1 - m ** 2; E = x - m @ W.T - c
        gW = E.T @ m / B - W * var.mean(0); gm = (E @ W + m * (W ** 2).sum(0)) / sigma ** 2; ga = gm * var
        W = W + lr * gW; c = c + lr * E.mean(0); R = R + lr * s * ga.T @ x / B; b = b + lr * s * ga.mean(0)
    for t in range(T + 1):
        tr.append((angle(W), angle(R.T)))
        if t in snaps: sn[t] = dict(W=W.tolist(), c=c.tolist(), R=R.tolist())
        if t == T: break
        if learner == 'ae': ae(); ae()
        elif learner == 'ae_fast': ae(); ae()
        else:
            wake()
            if learner == 'frozen': wake()
            else:
                z = np.where(r.random((B, 2)) < 0.5, -1.0, 1.0); xd = z @ W.T + c + sigma * r.standard_normal((B, 2)); g = z - np.tanh(xd @ R.T + b)
                R = R + lr * sleep_scale * g.T @ xd / B; b = b + lr * sleep_scale * g.mean(0)
    return np.array(tr), sn, X


def main():
    seeds = range(16); out = {'learners': {}, 'X0': None}
    for l in LEARNERS:
        runs = [simulate(l, s) for s in seeds]
        out['learners'][l] = {'gen': np.array([tr[:, 0] for tr, _, _ in runs]).tolist(), 'rec': np.array([tr[:, 1] for tr, _, _ in runs]).tolist(), 'snaps0': runs[0][1]}
        g = np.abs(np.array(out['learners'][l]['gen']))
        print(f"{l:8s} |angle| at 0/250/500/1000/3000: " + ' '.join(f'{g[:, t].mean():.1f}' for t in [0, 250, 500, 1000, 3000]) + f"  (range at 3000: {g[:, -1].min():.1f}-{g[:, -1].max():.1f})")
    out['X0'] = runs[0][2].tolist(); OUT.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(OUT, 'w')); print(OUT)


if __name__ == '__main__': main()
