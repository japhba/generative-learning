"""One row of a writer's table (K entries, M of them valid, target uniform on the valid ones) that starts with random mass on invalid entries:
how fast do fresh teacher examples remove that mass, against own samples judged by a reader that knows which entries are valid?
Arms, each given the same number of samples after a short warm-up on teacher examples:
  logits (softmax):  fresh, cross-entropy step            z += eta (e_c - p), c a valid entry drawn uniformly
                     own, signed three-factor step         z += eta v (e_k - p), k ~ p, v = +1 valid / -1 invalid
                     own, failure only                     z_k -= eta if k is invalid
  counts (p = n / sum n, the rule of toy model 3):
                     fresh, count                          n_c += 1                      (no minus: teacher examples can only thin)
                     fresh, count with a minus             n += e_c - p, clipped at 0    (delta rule on the counts)
                     own, signed count                     n_k <- max(0, n_k + v)
Reported: mass on invalid entries and KL(target || row), mean over rows. Usage: python experiments/evict_own_vs_fresh_row.py  (seconds)"""

import numpy as np

K, M, ROWS, WARM, ETA = 64, 4, 4000, 8, 0.5; T = [0, 10, 30, 100, 300, 1000, 3000]; VALID = np.arange(K) < M; IDX = np.arange(ROWS)


def soft(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def draw(p, rng): return (p.cumsum(1) < rng.random((ROWS, 1)) * p.sum(1, keepdims=True)).sum(1).clip(max=K - 1)


def onehot(k): e = np.zeros((ROWS, K)); e[IDX, k] = 1; return e


def stats(p): return p[:, ~VALID].sum(1).mean(), (np.log(1 / M) - np.log(p[:, VALID].clip(1e-300))).mean()


def run(arm, rng):
    logits = arm.startswith("logit"); x = rng.normal(0, 2, (ROWS, K)) if logits else rng.exponential(1, (ROWS, K)); out = {}
    for t in range(-WARM, T[-1] + 1):
        p = soft(x) if logits else x / x.sum(1, keepdims=True)
        if t in T: out[t] = stats(p)
        if t < 0 or "fresh" in arm:
            e = onehot(rng.integers(0, M, ROWS))
            x = x + (ETA * (e - p) if logits else (e - p if "minus" in arm else e))
        else:
            k = draw(p, rng); v = np.where(VALID[k], 1.0, -1.0); e = onehot(k)
            if arm == "logit own signed": x = x + ETA * v[:, None] * (e - p)
            elif arm == "logit own failure only": x = x - ETA * e * (v < 0)[:, None]
            else: x = x + v[:, None] * e
        if not logits: x = x.clip(min=0)
    return out


if __name__ == "__main__":
    arms = ["logit fresh", "logit own signed", "logit own failure only", "count fresh", "count fresh minus", "count own signed"]
    res = {a: run(a, np.random.default_rng(0)) for a in arms}
    for name, j in (("mass on invalid entries", 0), ("KL(target || row), nats", 1)):
        print(f"\n{name}\n{'samples':>24s} " + " ".join(f"{t:>8d}" for t in T))
        for a in arms: print(f"{a:>24s} " + " ".join(f"{res[a][t][j]:8.4f}" for t in T))
