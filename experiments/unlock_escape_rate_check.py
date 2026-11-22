"""How fast one diffuse level of the RHM leaves its symmetric start, and how much data it needs: check of the formulas.
One parent unit u (K states) ties two things together: a = a sibling tuple read in symbols (v*m states) and
y = the other sibling tuple read in symbols (k = 0), or ONE token k levels below the other sibling's symbol (k >= 1).
Model p(a, y) = sum_u pi_u A(a|u) B(y|u); sigma_i = canonical correlations of (a, y) (singular values of the whitened joint).
(1) EM from an almost diffuse start: amplitude x (1 + sigma_max) per pass.
(2) wake-sleep, reader = separate tables that track the generator at rate eta_s (expected updates): growth per step
    mu - 1, mu = 1 - (ew+es)/2 + sqrt(((ew-es)/2)^2 + ew es (1+sigma)), about sigma ew es / (ew + es).
(3) codes read right with prob. rho, else uniformly random: sigma -> rho^2 sigma.
(4) fixed set of T pairs, EM run long: share of the sibling information I(a; y) captured on the population, and how alike the
    synonyms' posteriors are, against T / T_c with T_c = sqrt(n_a n_y) / <sigma^2>.
Usage: python unlock_escape_rate_check.py [report]   (report = replot from the saved arrays)"""
import sys, numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm

OUT = '/var/tmp/jbauer/generative_practice_rhm/unlock_escape_rate.npz'; FIG = '/nfs/nhome/live/jbauer/generative-learning/reports/unlock-escape-rate.png'

def rules(v, m, rng):
    cells = rng.choice(v * v, size=v * m, replace=False); return np.stack([cells // v, cells % v], 1).reshape(v, m, 2)

def joint(v, m, k, rng):
    """p(a, y) and the symbol of every row: a = (symbol A, production i) of one sibling; y = the other sibling's tuple (k = 0)
    or a token k levels below it. Rows and columns that never occur are dropped."""
    r = rules(v, m, rng).reshape(-1, 2); PAB = np.zeros((v, v)); PAB[r[:, 0], r[:, 1]] = 1 / (v * m); cls = np.repeat(np.arange(v), m)
    if k == 0: P = np.repeat(np.repeat(PAB, m, 0), m, 1) / m ** 2
    else:
        D = np.eye(v)
        for _ in range(k):
            M = np.zeros((v, v)); np.add.at(M, (np.repeat(np.arange(v), m), rules(v, m, rng)[:, :, 0].ravel()), 1 / m); D = D @ M
        P = np.repeat(PAB @ D, m, 0) / m
    r1, r2 = P.sum(1) > 0, P.sum(0) > 0; return P[r1][:, r2], cls[r1]

def cancorr(P):
    p1, p2 = P.sum(1), P.sum(0); return np.linalg.svd(P / np.sqrt(np.outer(p1, p2)), compute_uv=False)[1:]

def em_pass(N, pi, A, B, smooth=0.0):
    W = N / ((A * pi[:, None]).T @ B); An = pi[:, None] * A * (B @ W.T); Bn = pi[:, None] * B * (A @ W); pin = An.sum(1)
    An = An + smooth * N.sum(1) / N.sum(); Bn = Bn + smooth * N.sum(0) / N.sum()
    return pin / pin.sum(), An / An.sum(1, keepdims=True), Bn / Bn.sum(1, keepdims=True)

def start(P, K, eps, rng):
    p1, p2 = P.sum(1) / P.sum(), P.sum(0) / P.sum(); A = p1 * (1 + eps * rng.standard_normal((K, len(p1)))); B = p2 * (1 + eps * rng.standard_normal((K, len(p2))))
    return np.full(K, 1 / K), A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True)

def amplitude(A, p1):
    d = A / p1 - 1; d = d - d.mean(0); return np.sqrt((d ** 2 * p1).sum(1).mean())

def slope(amp, lo=1e-7, hi=1e-3):
    i = np.where((amp > lo) & (amp < hi))[0]; i = i[len(i) // 2:]; return np.exp(np.polyfit(i, np.log(amp[i]), 1)[0])

def escape_em(v, m, k, seed, K=64, passes=4000):
    rng = np.random.default_rng(seed); P, _ = joint(v, m, k, rng); sig = cancorr(P); pi, A, B = start(P, K, 1e-9, rng); amp = []
    for _ in range(passes):
        amp.append(amplitude(A, P.sum(1)))
        if amp[-1] > 0.3: break
        pi, A, B = em_pass(P, pi, A, B)
    return np.array(amp), sig

def escape_ws(v, m, ew, es, seed, K=64, steps=20000):
    rng = np.random.default_rng(seed); P, _ = joint(v, m, 0, rng); sig = cancorr(P).max(); g = start(P, K, 1e-9, rng); r = [x.copy() for x in g]; amp = []
    for _ in range(steps):
        amp.append(amplitude(g[1], P.sum(1)))
        if amp[-1] > 1e-2: break
        t = em_pass(P, *r); g, r = [(1 - ew) * x + ew * y for x, y in zip(g, t)], [(1 - es) * x + es * y for x, y in zip(r, g)]
    mu = 1 - (ew + es) / 2 + np.sqrt(((ew - es) / 2) ** 2 + ew * es * (1 + sig)); return slope(np.array(amp)) - 1, mu - 1, sig * ew * es / (ew + es)

def fixed_set(args):
    v, m, k, x, seed, K, passes = args; rng = np.random.default_rng(seed); P, cls = joint(v, m, k, rng); nv = len(np.unique(cls)); sig = cancorr(P)[:nv - 1]
    Tc = np.sqrt(P.size) / (sig ** 2).mean(); N = rng.poisson(x * Tc * P).astype(float); k1, k2 = N.sum(1) > 0, N.sum(0) > 0; Nk = N[k1][:, k2]
    pi, A, B = start(Nk, K, 1e-2, rng)
    for _ in range(passes): pi, A, B = em_pass(Nk, pi, A, B, smooth=1e-2)
    Af = np.tile(P.sum(1), (K, 1)); Af[:, k1] = A * P.sum(1)[k1].sum(); Bf = np.tile(P.sum(0), (K, 1)); Bf[:, k2] = B * P.sum(0)[k2].sum()   # states never seen keep their marginal share
    S = (Af * pi[:, None]).T @ Bf; ind = np.outer(P.sum(1), P.sum(0)); nz = P > 0; gain = (P[nz] * np.log(S[nz] / ind[nz])).sum() / (P[nz] * np.log(P[nz] / ind[nz])).sum()
    R = (Af * pi[:, None]).T; R = R / R.sum(1, keepdims=True); mu = R.mean(0); tot = ((R - mu) ** 2).sum()
    raw = sum((cls == c).sum() * ((R[cls == c].mean(0) - mu) ** 2).sum() for c in np.unique(cls)) / tot; chance = (nv - 1) / (len(cls) - 1); return gain, (raw - chance) / (1 - chance), Tc

def run():
    v = 32; res = {}
    print('(1) EM, growth of the amplitude per pass against 1 + sigma_max   (v = 32)')
    for m in [4, 8]:
        for k in range(4):
            amp, sig = escape_em(v, m, k, 0); res[f'amp_m{m}_k{k}'] = amp; res[f'sig_m{m}_k{k}'] = sig
            print(f'  m={m} k={k}: sigma_max {sig.max():.4f}  rms {np.sqrt((sig[:v-2]**2).mean()):.4f}  (m^-(k+1)/2 sqrt(1-f)^(k+1) = {((1-m/v)/m)**((k+1)/2):.4f})   growth measured {slope(amp):.5f}  predicted {1+sig.max():.5f}   passes to leave {len(amp)}')
    print('\n(2) wake-sleep with a lagging reader (m = 8, k = 0): growth per step, measured / exact 2x2 / sigma ew es/(ew+es)')
    ws = []
    for ew, es in [(0.05, 0.05), (0.05, 0.2), (0.2, 0.05), (0.02, 0.5), (0.5, 0.02), (0.1, 1.0), (0.01, 0.01)]:
        a, b, c = escape_ws(v, 8, ew, es, 0); ws.append((ew, es, a, b, c)); print(f'  eta_w {ew:5.2f} eta_s {es:5.2f}: {a:.5f} / {b:.5f} / {c:.5f}')
    res['ws'] = np.array(ws)
    print('\n(3) codes read right with prob. rho (m = 8, k = 0): sigma_max(rho) / sigma_max(1) against rho^2')
    P, _ = joint(v, 8, 0, np.random.default_rng(0)); u = np.full(P.shape[0], 1 / P.shape[0]); s1 = cancorr(P).max()
    for rho in [0.9, 0.7, 0.5, 0.3]:
        Pr = rho * rho * P + rho * (1 - rho) * (np.outer(P.sum(1), u) + np.outer(u, P.sum(0))) + (1 - rho) ** 2 * np.outer(u, u); print(f'  rho {rho}: {cancorr(Pr).max() / s1:.4f}  {rho * rho:.4f}')
    print('\n(4) fixed set of T pairs, EM for 300 passes, K = v m: share of I(a; y) captured on the population | synonym overlap')
    xs = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]; seeds = 4; jobs = [(v, m, k, x, s, v * m, 300) for m in [4, 8] for k in range(3) for x in xs for s in range(seeds)]
    with Pool(24) as pool: out = np.array(list(tqdm(pool.imap(fixed_set, jobs, chunksize=1), total=len(jobs)))).reshape(2, 3, len(xs), seeds, 3)
    res['fixed'] = out; res['xs'] = np.array(xs)
    for im, m in enumerate([4, 8]):
        for k in range(3):
            print(f'  m={m} k={k} T_c={out[im, k, 0, 0, 2]:10.0f}  gain  ', ' '.join(f'{g:6.2f}' for g in out[im, k, :, :, 0].mean(1))); print(f'{"":27s}overlap', ' '.join(f'{g:6.2f}' for g in out[im, k, :, :, 1].mean(1)))
    np.savez(OUT, **res)

def report():
    import matplotlib, matplotlib.ticker; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = np.load(OUT); fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True); cols = plt.cm.viridis(np.linspace(0, 0.85, 4))
    for k in range(4):
        amp = d[f'amp_m8_k{k}']; t = np.arange(len(amp)); ax[0].semilogy(t, amp, color=cols[k], label=f'target {k} level{"s" * (k > 1)} below' if k else 'target read in symbols'); i = np.searchsorted(amp, 1e-6)
        ax[0].semilogy(t[i:], amp[i] * (1 + d[f'sig_m8_k{k}'].max()) ** (t[i:] - i), '--', color=cols[k], alpha=0.6)
    ax[0].set_xscale('log'); ax[0].set_ylim(1e-9, 1); ax[0].set_xlabel('EM passes'); ax[0].set_ylabel('distance from the diffuse start'); ax[0].set_title('leaving the diffuse start (m = 8)\ndashed: growth 1 + $\\sigma$ per pass'); ax[0].legend()
    ws = d['ws']; ax[1].loglog(ws[:, 3], ws[:, 2], 'o', label='against the exact 2 x 2 rate'); ax[1].loglog(ws[:, 4], ws[:, 2], 'x', label='against $\\sigma\\,\\eta_w\\eta_s/(\\eta_w+\\eta_s)$'); lim = [ws[:, 2:].min() / 1.5, ws[:, 2:].max() * 1.5]; ax[1].plot(lim, lim, 'k:', lw=1)
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[1].set_xlabel('predicted growth per step'); ax[1].set_ylabel('measured growth per step'); ax[1].set_title('wake-sleep with a lagging reader'); ax[1].legend()
    xs, out = d['xs'], d['fixed']
    for k in range(3):
        T = xs * out[1, k, 0, 0, 2]; g = out[1, k, :, :, 0]; ax[2].errorbar(T, g.mean(1), g.std(1) / np.sqrt(g.shape[1]), color=cols[k], marker='o', label=f'target {k} level{"s" * (k > 1)} below' if k else 'target read in symbols'); ax[2].axvline(out[1, k, 0, 0, 2], color=cols[k], ls=':')
    ax[2].set_xscale('log'); ax[2].axhline(0, color='k', lw=0.5); ax[2].set_ylim(-1.5, 1.05); ax[2].set_xlabel('sibling pairs in the fixed set, T'); ax[2].set_ylabel('share of the sibling information captured\n(population; below 0 = fitted to noise)'); ax[2].set_title('fixed set, EM run long (m = 8)\ndotted: predicted $T_c$'); ax[2].legend()
    fig.savefig(FIG, dpi=130, facecolor='white'); print(FIG)

if __name__ == '__main__':
    if len(sys.argv) == 1: run()
    report()
