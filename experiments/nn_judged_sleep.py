"""Sleep weighted by yesterday's recogniser, with a neural network as the recogniser. RHM (v symbols, m productions, pairs, depth L).
Generator (frozen): the true rules, blurred: every node writes one of its symbol's productions with probability a, else a random
pair of symbols. Own sample = (root symbol, tokens). World strings: a = 1.
Recogniser: MLP on the one-hot tokens -> root symbol, trained by sleep alone (cross-entropy to the root the sample was written from;
fresh dreams every step, no world string is ever used for training).
  plain     all dreams count the same
  judged    weight ~ q_old(root | tokens), q_old = copy of the network made every LAG steps; mean 1 among the dreams of each root
  unbal     the same with mean 1 over the whole batch
  today     judged by the network of this step
  shuffled  the judged weights given to other dreams of the same root
Reference: the blurred generator's exact posterior on world strings (inside algorithm): share of the mass on the right root, and how
often the right root is its most probable one. Measured on world strings: is the network's most probable root right (accuracy), and
the probability it gives to the right root.
Needs torch (the project's 'training' extra). Usage: python nn_judged_sleep.py [train | report]   (train: no figure, so no matplotlib needed)"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm
from unlock_escape_rate_check import rules

OUT = '/var/tmp/jbauer/generative_practice_reading/nn_judged_sleep.npz'; FIG = '/nfs/nhome/live/jbauer/generative-learning/reports/nn-judged-sleep.png'
V, M, DEPTHS, BLUR, ARMS, SEEDS = 16, 4, [2, 3], [0.4, 0.6, 0.8], ['plain', 'judged', 'unbal', 'today', 'shuffled'], 4
STEPS, BATCH, LAG, WIDTH, LRATE, EVERY, NTEST = 20000, 256, 200, 512, 1e-3, 250, 4000

def dream(R, a, n, rng):
    sym = rng.integers(V, size=(n, 1)); root = sym[:, 0].copy()
    for h in range(len(R) - 1, -1, -1):
        pair = R[h][sym, rng.integers(M, size=sym.shape)]; noise = rng.random(sym.shape) >= a; sym = np.where(noise[..., None], rng.integers(V, size=sym.shape + (2,)), pair).reshape(n, -1)
    return sym, root

def posterior(R, a, x):
    """root posterior of the blurred generator, x [n, tokens]"""
    msg = np.eye(V)[x]
    for h in range(len(R)):
        l, r = msg[:, 0::2], msg[:, 1::2]; msg = a / M * (l[:, :, R[h][:, :, 0]] * r[:, :, R[h][:, :, 1]]).sum(-1) + (1 - a) / V ** 2 * (l.sum(-1) * r.sum(-1))[..., None]; msg /= msg.sum(-1, keepdims=True)
    return msg[:, 0]

def run(job):
    import torch; torch.set_num_threads(1)
    depth, a, arm, seed = job; rng = np.random.default_rng([seed, depth]); torch.manual_seed(seed * 1000 + depth); R = [rules(V, M, rng) for _ in range(depth)]; xw, zw = dream(R, 1.0, NTEST, rng); xd, zd = dream(R, a, NTEST, rng)
    rng = np.random.default_rng([seed, depth, int(100 * a), ARMS.index(arm)]); oh = lambda x: torch.nn.functional.one_hot(torch.as_tensor(x), V).float().flatten(1); Xw, Zw, Xd, Zd = oh(xw), torch.as_tensor(zw), oh(xd), torch.as_tensor(zd)
    net = lambda: torch.nn.Sequential(torch.nn.Linear(V * 2 ** depth, WIDTH), torch.nn.ReLU(), torch.nn.Linear(WIDTH, WIDTH), torch.nn.ReLU(), torch.nn.Linear(WIDTH, V))
    f, old = net(), net(); old.load_state_dict(f.state_dict()); opt = torch.optim.Adam(f.parameters(), lr=LRATE); log = []; pw = posterior(R, a, xw)
    for t in range(STEPS + 1):
        if t % EVERY == 0:
            with torch.no_grad(): qw, qd = f(Xw).softmax(1), f(Xd).softmax(1)
            log.append([(qw.argmax(1) == Zw).float().mean().item(), qw[torch.arange(NTEST), Zw].mean().item(), (qd.argmax(1) == Zd).float().mean().item(), qd[torch.arange(NTEST), Zd].mean().item()])
        if t % LAG == 0: old.load_state_dict(f.state_dict())
        x, z = dream(R, a, BATCH, rng); X, Z = oh(x), torch.as_tensor(z); logits = f(X); w = torch.ones(BATCH)
        if arm != 'plain':
            with torch.no_grad(): sc = (logits.detach() if arm == 'today' else old(X)).softmax(1)[torch.arange(BATCH), Z]
            if arm == 'unbal': w = sc / sc.mean()
            else:
                Zo = torch.nn.functional.one_hot(Z, V).float(); w = sc / ((Zo * sc[:, None]).sum(0) / Zo.sum(0).clamp(min=1))[Z]
                if arm == 'shuffled': order = torch.argsort(Z * BATCH + torch.randperm(BATCH)); srt = torch.argsort(Z, stable=True); w2 = w.clone(); w2[srt] = w[order]; w = w2
        loss = (w * torch.nn.functional.cross_entropy(logits, Z, reduction='none')).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return dict(curve=np.array(log), ref=np.array([pw[np.arange(NTEST), zw].mean(), (pw.argmax(1) == zw).mean(), (a ** (2 ** depth - 1))]))

def main():
    jobs = [(d, a, arm, s) for d in DEPTHS for a in BLUR for arm in ARMS for s in range(SEEDS)]
    with Pool(24) as pool: out = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
    res = {}
    for (d, a, arm, s), o in zip(jobs, out): res.setdefault(f'{d}/{a}/{arm}/curve', []).append(o['curve']); res.setdefault(f'{d}/{a}/ref', []).append(o['ref'])
    np.savez(OUT, **{k: np.array(v) for k, v in res.items()})

def report():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = np.load(OUT); t = np.arange(0, STEPS + 1, EVERY) * BATCH; name = dict(plain='plain sleep', judged="weighted by yesterday's network, equal per root", unbal="weighted by yesterday's network", today="weighted by today's network", shuffled='the same weights, shuffled')
    first = lambda y, lv: np.array([t[np.argmax(c >= lv)] if (c >= lv).any() else np.nan for c in y])
    for depth in DEPTHS:
        for a in BLUR:
            ref = d[f'{depth}/{a}/ref'].mean(0); print(f'\ndepth {depth}, a production is written right in {a} (fully valid dreams: {ref[2]:.3f}); exact posterior on world strings: mass on the right root {ref[0]:.3f}, right root is the most probable {ref[1]:.3f}')
            for arm in ARMS:
                c = d[f'{depth}/{a}/{arm}/curve']; print(f'  {name[arm]:48s} world strings: accuracy at 1e5 / 1e6 / end {c[:, np.searchsorted(t, 1e5), 0].mean():.3f} {c[:, np.searchsorted(t, 1e6), 0].mean():.3f} {c[:, -1, 0].mean():.3f}; dreams until accuracy 0.5 / 0.9: {np.nanmedian(first(c[:, :, 0], 0.5)):9.0f} {np.nanmedian(first(c[:, :, 0], 0.9)):9.0f}; probability of the right root at the end {c[:, -1, 1].mean():.3f} | own samples: accuracy {c[:, -1, 2].mean():.3f}')
    w, h = plt.rcParams['figure.figsize']; fig, ax = plt.subplots(2, 3, figsize=(3.3 * w, 1.9 * h), constrained_layout=True, sharex=True); depth = 3; col = dict(plain='C0', judged='C1', unbal='C3', today='C2', shuffled='C7')
    for j, a in enumerate(BLUR):
        ref = d[f'{depth}/{a}/ref'].mean(0)
        for arm in ARMS:
            c = d[f'{depth}/{a}/{arm}/curve']
            for i in (0, 1): ax[i, j].plot(t[1:], c[:, 1:, i].mean(0), color=col[arm], label=name[arm]); ax[i, j].fill_between(t[1:], c[:, 1:, i].min(0), c[:, 1:, i].max(0), color=col[arm], alpha=0.12, lw=0)
        ax[0, j].axhline(ref[1], color='k', ls=':', lw=0.8, label="generator's exact posterior"); ax[1, j].axhline(ref[0], color='k', ls=':', lw=0.8, label="generator's exact posterior")
        ax[0, j].set_title(f'a production is written right in {a}'); ax[1, j].set_xlabel('own samples seen'); ax[0, j].set_xscale('log')
    ax[0, 0].set_ylabel('world strings: most probable root is right'); ax[1, 0].set_ylabel('world strings: probability of the right root'); ax[0, 2].legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f'network recogniser trained by sleep alone, RHM of depth {depth} (v = {V}, m = {M}); bands: range over {SEEDS} worlds'); fig.savefig(FIG, dpi=150, facecolor='white'); print(FIG)

if __name__ == '__main__':
    if sys.argv[1:] != ['report']: main()
    if sys.argv[1:] != ['train']: report()
