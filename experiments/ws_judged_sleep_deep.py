"""Sleep weighted by yesterday's recogniser in the two-level word model of ws_two_level_chimera.py (cause -> S symbols -> patches of L tokens;
reader bottom-up, patch -> symbol -> code, taught by own samples only). Weights with mean 1 per batch:
  R1 (patch -> symbol): w ~ q1_old(m~ | patch),   R2 (symbols -> code): w ~ q2_old(code~ | m~);   'balanced': mean 1 among the dreams of each symbol / code.
Starts: blank; dictionary right and top table blank (only the deep level is to be learned); both with PRESLEEP own batches before the first
world batch (the protocol of ws_two_level_chimera.py), and blank without them. Sizes (S, L) = (8, 8) and (2, 2).
Resumable: arms already in the saved file are not run again.
Usage: python ws_judged_sleep_deep.py [report]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm
import ws_two_level_chimera as C
from ws_two_level_chimera import K, M, V, B, LR, SEEDS, sm, draw

OUT = '/var/tmp/jbauer/generative_practice_reading/ws_judged_sleep_deep.npz'; LAG, STEPS, PRESLEEP = 25, 3000, 2000; T = np.unique(np.r_[0, np.round(np.geomspace(1, STEPS, 60))].astype(int))
SIZES, STARTS, KINDS, BATCHES = [(8, 8), (2, 2)], {'blank': (None, None, PRESLEEP), 'dictionary right, top blank': (None, 'L', PRESLEEP), 'blank, no own batches first': (None, None, 0)}, ['plain', 'judged', 'balanced'], 2

def size(S, L): C.S, C.L, C.SR, C.LR_ = S, L, np.arange(S), np.arange(L)

def weight(sc, lab, n_lab, kind):
    """sc, lab [n, ...]: yesterday's probability of the label that was written, and that label"""
    flat, fl = sc.reshape(SEEDS, -1), lab.reshape(SEEDS, -1)
    if kind == 'balanced': oh = np.eye(n_lab)[fl]; w = flat / np.take_along_axis((oh * flat[..., None]).sum(1) / oh.sum(1).clip(1), fl, 1)
    else: w = flat / flat.mean(1, keepdims=True)
    return w.reshape(sc.shape)

def sleep(rng, A, D, R1, b1, R2, b2, old, kind):
    c, m, x = C.dream(rng, A, D, B); w1, w2 = np.ones(m.shape), np.ones(c.shape)
    if kind != 'plain':
        w1 = weight(np.take_along_axis(sm(C.read1(old[0], old[1], x)), m[..., None], 3)[..., 0], m, M, kind); w2 = weight(np.take_along_axis(sm(C.read2(old[2], old[3], m)), c[..., None], 2)[..., 0], c, K, kind)
    g1 = w1[..., None] * (np.eye(M)[m] - sm(C.read1(R1, b1, x))); R1 += LR * np.einsum("njbm,njlbv->nlmv", g1, np.eye(V)[x]) / (C.S * B); b1 += LR * g1.mean((1, 2))
    g2 = w2[..., None] * (np.eye(K)[c] - sm(C.read2(R2, b2, m))); R2 += LR * np.einsum("nbk,njbm->njkm", g2, np.eye(M)[m]) / B; b2 += LR * g2.mean(1)

def run(a):
    (S, L), start, kind, seed = a; size(S, L); top, bottom, pre = STARTS[start]; bottom = L if bottom == 'L' else bottom; rng = np.random.default_rng([seed, S, L])
    tau, sigma = np.argsort(rng.random((SEEDS, S, K)), -1), np.argsort(rng.random((SEEDS, L, M)), -1); A, D, ta, td = C.start(rng, tau, sigma, top, bottom); A0, D0 = sm(A), sm(D)
    R1, b1, R2, b2, log = np.zeros((SEEDS, L, M, V)), np.zeros((SEEDS, M)), np.zeros((SEEDS, S, K, M)), np.zeros((SEEDS, K)), []; old = [R1.copy(), b1.copy(), R2.copy(), b2.copy()]
    for t in range(-pre, STEPS + 1):
        if t % LAG == 0: old = [R1.copy(), b1.copy(), R2.copy(), b2.copy()]
        if t >= 0 and t in T: log.append(C.score(rng, tau, sigma, A, D, R1, b1, R2, b2, A0, D0))
        if t >= 0: C.wake(rng, tau, sigma, A, D, R1, b1, R2, b2, True, "bottom-up")
        sleep(rng, A, D, R1, b1, R2, b2, old, kind)
    return {k: np.array([row[k] for row in log]) for k in C.KEYS}

def main():
    res = dict(np.load(OUT)) if os.path.exists(OUT) else {}; jobs = [(sz, st, kind, s) for sz in SIZES for st in STARTS for kind in KINDS for s in range(BATCHES) if f'{sz[0]}x{sz[1]}/{st}/{kind}/kl' not in res]
    if not jobs: return
    with Pool(len(jobs)) as pool: out = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
    for a, o in zip(jobs, out):
        for k, v in o.items(): res[f'{a[0][0]}x{a[0][1]}/{a[1]}/{a[2]}/{k}'] = np.concatenate([res[f'{a[0][0]}x{a[0][1]}/{a[1]}/{a[2]}/{k}'], v], 1) if a[3] else v
    np.savez(OUT, **res)

def report():
    d = np.load(OUT); onset = lambda y: np.where((y >= 0.5).any(0), T[(y >= 0.5).argmax(0)], np.nan); med = lambda v: np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan
    name = dict(plain='plain sleep', judged="weighted by yesterday's reading", balanced='the same, equal weight per label')
    for (S, L) in SIZES:
        for st in STARTS:
            print(f'\n{S} symbols x {L} tokens, start: {st}   (world batches until the median world is above 0.5; {BATCHES * SEEDS} worlds)')
            for kind in KINDS:
                r = {k: d[f'{S}x{L}/{st}/{kind}/{k}'] for k in C.KEYS}
                print(f'  {name[kind]:34s} patches valid {med(onset(r["surface"])):6.0f}  combinations valid {med(onset(r["deep"])):6.0f}  strings read {med(onset(r["read2"])):6.0f}   KL at 100 / 1000 / end: {np.median(r["kl"][np.searchsorted(T, 100)]):6.2f} {np.median(r["kl"][np.searchsorted(T, 1000)]):6.2f} {np.median(r["kl"][-1]):6.2f}   worlds with KL > 1 at the end: {(r["kl"][-1] > 1).sum()}/{r["kl"].shape[1]}   before any world batch: syllables read {np.median(r["read1"][0]):.2f}, strings read {np.median(r["read2"][0]):.2f}')

if __name__ == '__main__':
    if len(sys.argv) == 1: main()
    report()
