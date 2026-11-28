"""Sampled runs on the order-m world (one winner-take-all group, K = q^(m-1) states, m blocks valid iff they sum to 0 mod q): time to
KL(teacher || writer) < 0.1 for teacher strings only and for practice variants with half of the samples own, at a weak or a sharp coupled
start (reader and writer from the same table). Rule P <- P + eps (onehot - P) at the credited state, eps 0.003, batches of 64.
  teacher only        : teacher strings, credit = reader posterior; move reader R and writer G.
  sleep only          : own samples move R with the writing state (classic sleep).
  verdict + sleep     : own samples move R with the writing state; accepted (valid) ones also move G at the writing state.
  verdict signed      : as above, refused samples move G away from what they wrote (log domain).
  verdict, strict     : as 'verdict + sleep' with credit = the reader's posterior of the own string instead of the writing state.
  verdict, no sleep   : accepted own samples move G only.
  all own             : pure practice (verdict + sleep) from the start.

    python practice_identity_sampled.py --out /var/tmp/jbauer/practice_identity/sampled.json
"""
import argparse, itertools, json
from multiprocessing import Pool
import numpy as np

FLOOR = 1e-6
sm = lambda u: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(u - u.max(-1, keepdims=True)))


def hebb(P, states, symbols, eps, sign=None):
    K, q = P.shape[0], P.shape[-1]
    s1 = np.eye(K)[states]
    if sign is None:
        count = s1.sum(0)
        target = np.einsum('nk,niv->kiv', s1, np.eye(q)[symbols]) / np.maximum(count, 1)[:, None, None]
        keep = ((1 - eps) ** count)[:, None, None]
        return keep * P + (1 - keep) * target
    return sm(np.log(P + 1e-300) + eps * np.einsum('nk,niv,n->kiv', s1, np.eye(q)[symbols], sign))


def run(job):
    arm, q, m, a, own, mode, seed, t_max, eps = job
    rng, K = np.random.default_rng([seed, 2]), q ** (m - 1)
    G = sm(a * np.random.default_rng([seed, 1]).normal(size=(K, m, q))); R = G.copy()
    S = np.array(list(itertools.product(range(q), repeat=m))); valid = S.sum(1) % q == 0
    batch, n_own, blocks = 64, 0, np.arange(m)[None, :]
    draw = lambda logits: (logits + rng.gumbel(size=logits.shape)).argmax(-1)
    read = lambda s: draw(np.log(R + FLOOR)[:, blocks, s].sum(2).T)
    def kl():
        p = np.exp(np.log(G + FLOOR)[:, blocks, S].sum(2)).mean(0)
        return float(np.mean(np.log((1 / valid.sum()) / np.maximum(p[valid], 1e-12))))
    for i in range(int(t_max / batch) + 1):
        if i % 100 == 0 and kl() < .1:
            return dict(arm=arm, q=q, m=m, a=a, seed=seed, t=i * batch, kl=kl())
        if n_own < own * (i + 1):
            n_own += 1
            k = rng.integers(K, size=batch)
            s = draw(np.log(G + FLOOR)[k])
            V = s.sum(1) % q == 0
            cred = read(s) if mode == 'strict' else k
            if mode in ('verdict', 'strict', 'signed', 'nosleep') and V.any() and mode != 'signed':
                G = hebb(G, cred[V], s[V], eps)
            if mode == 'signed':
                G = hebb(G, cred, s, eps, sign=np.where(V, 1., -1.))
            if mode != 'nosleep':
                R = hebb(R, cred, s, eps)
        else:
            s = rng.integers(q, size=(batch, m)); s[:, -1] = (-s[:, :-1].sum(1)) % q
            k = read(s)
            R, G = hebb(R, k, s, eps), hebb(G, k, s, eps)
    return dict(arm=arm, q=q, m=m, a=a, seed=seed, t=None, kl=kl())


ARMS = {'teacher only': (0., 'teacher'), 'sleep only': (.5, 'sleep'), 'verdict + sleep': (.5, 'verdict'), 'verdict signed': (.5, 'signed'),
        'verdict, strict credit': (.5, 'strict'), 'verdict, no sleep': (.5, 'nosleep'), 'all own (verdict + sleep)': (1., 'verdict')}
CASES = ((2, 3), (2, 4), (2, 5), (3, 3))
STARTS = (.01, .3, 1., 4.)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='/var/tmp/jbauer/practice_identity/sampled.json')
    out = ap.parse_args().out
    jobs = [(arm, q, m, a, *ARMS[arm], s, 4e6, .003) for q, m in CASES for arm in ARMS for a in STARTS for s in range(8)]
    jobs.sort(key=lambda j: -(j[1] ** j[2]) / (1 + j[3]))
    with Pool(30) as pool:
        res = pool.map(run, jobs, chunksize=1)
    json.dump(res, open(out, 'w'))
    for q, m in CASES:
        for a in STARTS:
            print(f'q {q} m {m} start {a}')
            for arm in ARMS:
                rs = [r for r in res if (r['arm'], r['q'], r['m'], r['a']) == (arm, q, m, a)]
                d = [r['t'] for r in rs if r['t'] is not None]
                print(f"   {arm:28s} {np.median(d) / 1000 if d else float('nan'):7.0f}k ({len(d)}/8)")
