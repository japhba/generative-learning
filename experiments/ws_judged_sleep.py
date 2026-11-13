"""Sleep updates weighted by yesterday's recogniser. Same world and learner as ws_reading_gain.py (K causes, s views, one fixed
token per cause and view; generator and recogniser are softmax tables; wake: code sampled from the recogniser; sleep: the only
update of the recogniser).
  sleep, weighted:  own batch c~ ~ prior, x~ ~ pi(. | c~);  R_j[:, x~_j] += lr w (onehot(c~) - q(. | x~)),  weights with mean 1 per batch
  judged    w ~ q_old(c~ | x~): the probability that yesterday's recogniser (a copy made every LAG updates) gives to the code the
            sample was written from. At the optimum q_new(c | x) ~ p(c | x) q_old(c | x): the generator's posterior, sharpened day by day.
  balanced  judged, but the weights have mean 1 among the dreams of each code (no code gets more sleep than another)
  today     the same as judged with today's recogniser as the judge (no lag)
  shuffled  the judged weights given to other dreams of the batch (same spread of weights, no judgement)
  sure      w ~ max_c q_old(c | x~): how sure yesterday's recogniser is of ANY code (a score of the sample alone)
  misread   w ~ 1 - q_old(c~ | x~): the dreams yesterday's recogniser got wrong count more
  argmax    plain sleep, but the wake code is the recogniser's most probable code instead of a sample (sharp from the start)
  ramp      plain sleep, wake code sampled from q^beta with beta = 1 + t / LAG (sharpening by a schedule, no judge)
Part 1: all of it learns (16 worlds, s = 8 and s = 2). Part 2: generator frozen at per-view accuracy a, recogniser from zero by sleep alone.
Usage: python ws_judged_sleep.py [report]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm
import ws_reading_gain as W
from ws_reading_gain import K, V, sm, gather, draw, dream, score, rho, onset

OUT = '/var/tmp/jbauer/generative_practice_reading/ws_judged_sleep.npz'; FIG = '/nfs/nhome/live/jbauer/generative-learning/reports/ws-judged-sleep.png'
W.SEEDS = SEEDS = 32; B, LR, STEPS, LAG = W.B, W.LR, 5000, 25; T = np.unique(np.r_[0, np.round(np.geomspace(1, STEPS, 70))].astype(int))
ARMS = ['plain', 'judged', 'balanced', 'today', 'shuffled', 'sure', 'misread', 'argmax', 'ramp']; LAGS = [5, 25, 100, 400]; SS = [8, 2]
FROZEN_A, FROZEN_STEPS = [0.15, 0.2, 0.3], 4000; TF = np.unique(np.r_[0, np.round(np.geomspace(1, FROZEN_STEPS, 40))].astype(int)); KEYS = ["read_world", "read_own", "a", "valid", "kl"]

def sleep(rng, G, R, b, Ro, bo, arm):
    c, x = dream(rng, G, B); w = np.ones(c.shape)
    if arm in ('judged', 'balanced', 'today', 'shuffled', 'sure', 'misread'):
        qo = sm(bo[:, None] + gather(Ro, x).sum(1)); sc = qo.max(2) if arm == 'sure' else np.take_along_axis(qo, c[..., None], 2)[..., 0]; sc = 1 - sc if arm == 'misread' else sc
        w = sc / sc.mean(1, keepdims=True); w = rng.permuted(w, axis=1) if arm == 'shuffled' else w
        if arm == 'balanced': oh = np.eye(K)[c]; w = sc / np.take_along_axis((oh * sc[..., None]).sum(1) / oh.sum(1).clip(1), c, 1)
    g = w[..., None] * (np.eye(K)[c] - sm(b[:, None] + gather(R, x).sum(1))); R += LR * np.einsum("nbk,njbv->njkv", g, np.eye(V)[x]) / B; b += LR * g.mean(1)

def run(a):
    s, arm, lag, seed = a; rng = np.random.default_rng([seed, s]); sig = np.argsort(rng.random((SEEDS, s, K)), -1)
    G, R, b, log = 0.1 * rng.standard_normal((SEEDS, s, K, V)), np.zeros((SEEDS, s, K, V)), np.zeros((SEEDS, K)), []; Ro, bo = R.copy(), b.copy()
    for t in range(STEPS + 1):
        if t in T: log.append(score(rng, G, R, b, sig))
        if arm == 'today' or t % lag == 0: Ro, bo = R.copy(), b.copy()
        c = rng.integers(K, size=(SEEDS, B)); x = np.take_along_axis(sig, np.broadcast_to(c[:, None, :], (SEEDS, s, B)), 2); lq = b[:, None] + gather(R, x).sum(1)
        oc = np.eye(K)[lq.argmax(-1) if arm == 'argmax' else draw(rng, lq * (1 + t / lag if arm == 'ramp' else 1))]
        G += LR * (np.einsum("nbk,njbv->njkv", oc, np.eye(V)[x]) - oc.sum(1)[:, None, :, None] * sm(G)) / B; sleep(rng, G, R, b, Ro, bo, arm)
    return {k: np.array([row[k] for row in log]) for k in KEYS}

def frozen(a):
    acc, s, arm, lag, seed = a; rng = np.random.default_rng([seed, s, int(100 * acc)]); sig = np.argsort(rng.random((SEEDS, s, K)), -1); P = np.full((SEEDS, s, K, V), (1 - acc) / (V - 1))
    np.put_along_axis(P, sig[..., None], acc, 3); G, R, b, log = np.log(P), np.zeros((SEEDS, s, K, V)), np.zeros((SEEDS, K)), []; Ro, bo = R.copy(), b.copy()
    for t in range(FROZEN_STEPS + 1):
        if t in TF: log.append(score(rng, G, R, b, sig))
        if t % lag == 0: Ro, bo = R.copy(), b.copy()
        sleep(rng, G, R, b, Ro, bo, arm)
    return {k: np.array([row[k] for row in log]) for k in KEYS}

def job(a): return (frozen if a[0] == 'f' else run)(a[1:])

def main():
    j1 = [('r', s, arm, LAG, 0) for s in SS for arm in ARMS] + [('r', s, 'judged', lag, 0) for s in SS for lag in LAGS if lag != LAG]; j2 = [('f', acc, 8, arm, LAG, 0) for acc in FROZEN_A for arm in ('plain', 'judged', 'balanced', 'shuffled', 'sure')]
    with Pool(len(j1 + j2)) as pool: out = list(tqdm(pool.imap(job, j1 + j2), total=len(j1 + j2)))
    res = dict(t=T, tf=TF)
    for a, o in zip(j1 + j2, out):
        for k, v in o.items(): res[(f'f/{a[1]}/{a[3]}' if a[0] == 'f' else f'r/{a[1]}/{a[2]}/{a[3]}') + f'/{k}'] = v
    np.savez(OUT, **res)

def report():
    import matplotlib.pyplot as plt
    d = np.load(OUT); name = dict(plain='plain sleep', judged="weighted by yesterday's reading of the code", balanced='the same, equal weight per code', today="weighted by today's reading of the code", shuffled='the same weights, shuffled', sure='weighted by how sure the reader is', misread='misread dreams count more', argmax='plain sleep, wake code = most probable', ramp='plain sleep, wake code sharpened by a schedule')
    def line(key):
        r = {k: d[f'{key}/{k}'] for k in KEYS}; on = {k: onset(T, r[k]) for k in ('read_world', 'a', 'valid')}; bound = (r['a'][-1] > 0.9).sum()
        return '  '.join(f'{k} {np.nanmedian(v):6.0f}' for k, v in on.items()) + f'   end: reads world {np.median(r["read_world"][-1]):.3f}, valid {np.median(r["valid"][-1]):.3f}, KL {np.median(r["kl"][-1]):.3f}; worlds with every cause bound to its own code: {bound}/{SEEDS}'
    for s in SS:
        print(f'\ns = {s}: updates until the median world is above 0.5 (reads world strings, writes one view right, writes valid strings)')
        for arm in ARMS: print(f'  {name[arm]:46s} ' + line(f'r/{s}/{arm}/{LAG}'))
        for lag in LAGS: print(f'  judged, a day = {lag:4d} updates                    ' + line(f'r/{s}/judged/{lag}'))
    print(f'\nfrozen generator, s = 8, sleep alone, {FROZEN_STEPS} own batches: share of world strings read right (generator\'s exact posterior: rho(a))')
    for acc in FROZEN_A: print(f'  a = {acc}: rho {rho(acc, 8):.3f} | ' + ' | '.join(f'{name[arm]}: world {d[f"f/{acc}/{arm}/read_world"][-1].mean():.3f}, own {d[f"f/{acc}/{arm}/read_own"][-1].mean():.3f}' for arm in ('plain', 'judged', 'balanced', 'shuffled', 'sure')))
    w, h = plt.rcParams['figure.figsize']; fig, ax = plt.subplots(1, 3, figsize=(3.3 * w, 1.0 * h), constrained_layout=True); col = dict(plain='C0', judged='C1', balanced='C5', shuffled='C7', argmax='C3', ramp='C2', sure='C4')
    for i, acc in enumerate(FROZEN_A):
        for arm, ls in (('plain', '--'), ('judged', '-')): ax[0].plot(TF[1:], d[f'f/{acc}/{arm}/read_world'].mean(1)[1:], color=f'C{i}', ls=ls, label=f'writes a view right in {acc:.2f}: {name[arm] if arm == "plain" else "weighted by yesterday"}')
    ax[0].set(xscale='log', xlabel='own batches (generator frozen, no world strings)', ylabel='share of world strings read right', title='(a) reader trained by sleep alone')
    for arm in ('plain', 'judged', 'balanced', 'shuffled', 'argmax', 'ramp'):
        r = d[f'r/8/{arm}/{LAG}/valid']; ax[1].plot(T[1:], np.median(r, 1)[1:], color=col[arm], label=name[arm]); r = d[f'r/2/{arm}/{LAG}/valid']; ax[2].plot(T[1:], np.median(r, 1)[1:], color=col[arm], label=name[arm])
    for a, s in ((ax[1], 8), (ax[2], 2)): a.set(xscale='log', xlabel='updates (one world batch + one own batch)', ylabel='share of own samples that are valid (median world)', title=f'({"b" if s == 8 else "c"}) all of it learning, {s} views')
    ax[0].legend(loc='lower right'); ax[2].legend(loc='center left', bbox_to_anchor=(1.01, 0.5)); fig.savefig(FIG, dpi=150, facecolor='white'); print(FIG)

if __name__ == '__main__':
    if len(sys.argv) == 1: main()
    report()
