"""Do own samples improve recognition, and does recognition run ahead of generation? Plain wake-sleep, redundancy s as the dial.
World: K causes (uniform); each cause shows one fixed token (of V) in each of s views, so there are K valid strings.
Learner: generator pi_j(x_j | c) = softmax(G_j[c]) with a uniform prior over K codes; recogniser q(c | x) = softmax(b + sum_j R_j[:, x_j]).
  wake   world batch, code c^ ~ q(. | x),            G_j[c^] += lr (onehot(x_j) - pi_j(. | c^))
  sleep  own batch c~ ~ prior, x~ ~ pi(. | c~),      R_j[:, x~_j] += lr (onehot(c~) - q(. | x~))        (the only update of the recogniser)
Scores (codes matched to causes through the generator's per-view accuracy matrix):
  read_world  recogniser gives a world string the code that writes its cause      read_exact  the same for the generator's exact posterior
  read_own    recogniser recovers the code an own sample was written from          own_exact   the same for the exact posterior
  a           per-view writing accuracy                                            valid       mass of own samples that are valid strings
Theory for a uniformly blurred generator (right token with probability a): rho(a) = a^s / (a^s + (K-1) ((1-a)/(V-1))^s), slope s at chance.
Part 2 (frozen): generator fixed at blur a, recogniser trained from zero by sleep alone.
Part 3 (kinds): what the generator must have for its posterior to read world strings. Every code writes r of its s views right and the
others always wrong (so it never writes a valid string), with a fixed random wrong token (frozen) or the token of one other cause (coherent);
each view is smoothed by EPS. Compared with the blurred generator of the same per-view accuracy.
Usage: python ws_reading_gain.py -> reports/ws-reading-gain.png, /var/tmp/jbauer/generative_practice_reading/ws_reading_gain.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

K = V = 8; SS = [1, 2, 4, 8, 16]; SEEDS, B, LR, STEPS, NEVAL = 16, 64, 1.0, 5000, 4096
DREAMS = [0, 0.25, 1, 4]; EPS, KINDS = 0.05, [(8, 2), (8, 3), (8, 4), (8, 6), (16, 3), (16, 4), (16, 6), (16, 8), (16, 12), (32, 6), (32, 8)]; FROZEN_A, FROZEN_S, FROZEN_STEPS = [0.15, 0.2, 0.3], 8, 10000
T = np.unique(np.r_[0, np.round(np.geomspace(1, STEPS, 70))].astype(int)); TF = np.unique(np.r_[0, np.round(np.geomspace(1, FROZEN_STEPS, 40))].astype(int))
OUT = Path("/var/tmp/jbauer/generative_practice_reading"); FIG = Path(__file__).resolve().parents[1] / "reports/ws-reading-gain.png"
KEYS = ["read_world", "read_exact", "read_own", "own_exact", "a", "valid", "kl"]


def sm(l): e = np.exp(l - l.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def rho(a, s): return a ** s / (a ** s + (K - 1) * ((1 - a) / (V - 1)) ** s)
def gather(W, x): n, s = np.arange(W.shape[0])[:, None, None], np.arange(W.shape[1])[None, :, None]; return W[n, s, :, x]      # W [n,s,K,V], x [n,s,b] -> [n,s,b,K]
def draw(rng, logits): return (logits + rng.gumbel(size=logits.shape)).argmax(-1)
def dream(rng, G, n): c = rng.integers(K, size=(G.shape[0], n)); return c, draw(rng, G[np.arange(G.shape[0])[:, None, None], np.arange(G.shape[1])[None, :, None], c[:, None, :]])


def greedy_match(A):                                                                                        # A [n, cause, code] -> code of each cause, one to one
    A, m = A.copy(), np.zeros(A.shape[:2], int); n = np.arange(len(A))
    for _ in range(K): flat = A.reshape(len(A), -1).argmax(1); c, k = flat // K, flat % K; m[n, c] = k; A[n, c, :] = -1; A[n, :, k] = -1
    return m


def score(rng, G, R, b, sig):
    n, logpi = np.arange(len(G))[:, None], np.log(sm(G)); xw = sig                                          # the K world strings, [n, s, K]
    lw = gather(logpi, xw); Lw, A = lw.sum(1), np.exp(lw).mean(1); m = greedy_match(A); cause = np.arange(K)[None]   # [n, cause, code]: log p(string of cause | code), per-view accuracy
    Q = sm(b[:, None] + gather(R, xw).sum(1)); c, x = dream(rng, G, NEVAL); qo, po = sm(b[:, None] + gather(R, x).sum(1)), sm(gather(logpi, x).sum(1))
    return {"read_world": Q[n, cause, m].mean(1), "read_exact": sm(Lw)[n, cause, m].mean(1), "read_own": np.take_along_axis(qo, c[..., None], 2).mean((1, 2)),
            "own_exact": np.take_along_axis(po, c[..., None], 2).mean((1, 2)), "a": A[n, cause, m].mean(1), "valid": np.exp(Lw).sum((1, 2)) / K,
            "kl": -np.log(np.exp(Lw).sum(2)).mean(1)}


def sleep(rng, G, R, b):
    c, x = dream(rng, G, B); g = np.eye(K)[c] - sm(b[:, None] + gather(R, x).sum(1))
    R += LR * np.einsum("nbk,njbv->njkv", g, np.eye(V)[x]) / B; b += LR * g.mean(1)


def run(s, dreams=1, seed=0, desc=""):
    rng = np.random.default_rng(seed); sig = np.argsort(rng.random((SEEDS, s, K)), -1)                      # token of cause c in view j
    G, R, b, log, debt = 0.1 * rng.standard_normal((SEEDS, s, K, V)), np.zeros((SEEDS, s, K, V)), np.zeros((SEEDS, K)), [], 0.0
    for t in tqdm(range(STEPS + 1), desc=desc, leave=False):
        if t in T: log.append(score(rng, G, R, b, sig))
        c = rng.integers(K, size=(SEEDS, B)); x = np.take_along_axis(sig, np.broadcast_to(c[:, None, :], (SEEDS, s, B)), 2)
        oc = np.eye(K)[draw(rng, b[:, None] + gather(R, x).sum(1))]                                         # wake: sampled code of each world string
        G += LR * (np.einsum("nbk,njbv->njkv", oc, np.eye(V)[x]) - oc.sum(1)[:, None, :, None] * sm(G)) / B
        debt += dreams
        while debt >= 1: sleep(rng, G, R, b); debt -= 1
    return {k: np.array([row[k] for row in log]) for k in KEYS}                                             # [time, seed]


def frozen(a, s, seed=0):
    rng = np.random.default_rng(seed); sig = np.argsort(rng.random((SEEDS, s, K)), -1); P = np.full((SEEDS, s, K, V), (1 - a) / (V - 1))
    np.put_along_axis(P, sig[..., None], a, 3); G, R, b, log = np.log(P), np.zeros((SEEDS, s, K, V)), np.zeros((SEEDS, K)), []
    for t in range(FROZEN_STEPS + 1):
        if t in TF: log.append(score(rng, G, R, b, sig))
        sleep(rng, G, R, b)
    return {k: np.array([row[k] for row in log]) for k in KEYS}


def kinds(s, r, kind, n=2000, seed=0):                                                                    # exact posterior reading of the K world strings
    rng = np.random.default_rng(seed); sig = np.argsort(rng.random((n, s, K)), -1); right = np.argsort(rng.random((n, K, s)), -1).transpose(0, 2, 1) < r   # [n, s, code]
    wrong = np.take_along_axis(sig, np.broadcast_to((np.arange(K) + 1) % K, sig.shape), 2) if kind == "coherent" else (sig + rng.integers(1, V, size=sig.shape)) % V
    P = np.full((n, s, K, V), EPS / V); np.put_along_axis(P, np.where(right, sig, wrong)[..., None], 1 - EPS + EPS / V, 3); lw = gather(np.log(P), sig)
    return {"a": np.exp(lw)[:, :, np.arange(K), np.arange(K)].mean(), "read": sm(lw.sum(1))[:, np.arange(K), np.arange(K)].mean(), "valid": np.exp(lw.sum(1)).sum((1, 2)).mean() / K}


def onset(t, y, level=0.5): hit = y >= level; return np.where(hit.any(0), t[hit.argmax(0)], np.nan)        # first evaluation at which each seed is above the level


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    main_runs = {s: run(s, desc=f"s={s}") for s in SS}; dream_runs = {d: run(8, dreams=d, desc=f"dreams={d}") for d in DREAMS}; frozen_runs = {a: frozen(a, FROZEN_S) for a in FROZEN_A}
    np.savez(OUT / "ws_reading_gain.npz", t=T, tf=TF, **{f"s{s}/{k}": v for s, r in main_runs.items() for k, v in r.items()},
             **{f"d{d}/{k}": v for d, r in dream_runs.items() for k, v in r.items()}, **{f"a{a}/{k}": v for a, r in frozen_runs.items() for k, v in r.items()})

    print("updates until the median seed is above 0.5 (seeds that get there)")
    for s, r in main_runs.items():
        row = {k: onset(T, r[k]) for k in ["read_world", "read_own", "a", "valid"]}
        print(f"  s={s:2d}  " + "  ".join(f"{k} {np.nanmedian(v):7.0f} ({np.isfinite(v).sum():2d}/{SEEDS})" for k, v in row.items()) + f"  end: read_world {np.median(r['read_world'][-1]):.2f} a {np.median(r['a'][-1]):.2f} valid {np.median(r['valid'][-1]):.2f} kl {np.median(r['kl'][-1]):.2f}"
              + f"  read ahead of a at {np.mean(r['read_world'][1:] > r['a'][1:]):.2f} of evaluations")
    print("dreams per world batch, s = 8: world batches until read_world / valid above 0.5")
    for d, r in dream_runs.items(): print(f"  dreams={d:<5}" + "  ".join(f"{k} {np.nanmedian(onset(T, r[k])):7.0f} ({np.isfinite(onset(T, r[k])).sum():2d}/{SEEDS})" for k in ["read_world", "valid"]) + f"  end kl {np.median(r['kl'][-1]):.2f}")
    print(f"frozen generator, s = {FROZEN_S}, sleep only: after {FROZEN_STEPS} own batches")
    for a, r in frozen_runs.items(): print(f"  a={a}: read_world {r['read_world'][-1].mean():.3f} (exact {r['read_exact'][-1].mean():.3f}, formula {rho(a, FROZEN_S):.3f})  read_own {r['read_own'][-1].mean():.3f} (exact {r['own_exact'][-1].mean():.3f})  valid {r['valid'][-1].mean():.2e}")

    print(f"kinds of generator error (exact posterior on world strings, smoothing {EPS}): per-view accuracy a | read (valid strings written) frozen, coherent, blur of the same a")
    for s, r in KINDS:
        f, c = kinds(s, r, "frozen"), kinds(s, r, "coherent")
        print(f"  s={s:2d} right views {r:2d}: a {f['a']:.2f} | frozen {f['read']:.3f} ({f['valid']:.1e})  coherent {c['read']:.3f} ({c['valid']:.1e})  blur {rho(f['a'], s):.3f} ({f['a'] ** s:.1e})")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True); med = lambda y: np.median(y, 1)
    r = main_runs[8]
    for k, lab, c, ls in [("read_world", "reads world strings", "C0", "-"), ("read_own", "reads own samples", "C0", ":"), ("a", "writes one view right", "C3", "-"), ("valid", "writes a whole valid string", "C3", "--")]:
        ax[0].plot(T[1:], med(r[k])[1:], color=c, ls=ls, label=lab); ax[0].fill_between(T[1:], np.quantile(r[k], 0.25, 1)[1:], np.quantile(r[k], 0.75, 1)[1:], color=c, alpha=0.12, edgecolor="none")
    ax[0].set(xscale="log", xlabel="updates (one world batch + one own batch each)", ylabel="share right", title="wake-sleep, s = 8 views"); ax[0].legend()
    grid = np.linspace(1 / V, 1, 400)
    for i, s in enumerate(SS):
        r = main_runs[s]; ax[1].plot(grid, rho(grid, s), color=f"C{i}", alpha=0.5); ax[1].plot(med(r["a"]), med(r["read_world"]), ".", color=f"C{i}", label=f"s = {s}")
    ax[1].plot([1 / V, 1], [1 / V, 1], color="k", ls=":"); ax[1].set(xlabel="writes one view right, a", ylabel="reads world strings", title="reading against writing (lines: exact posterior)"); ax[1].legend()
    for i, a in enumerate(FROZEN_A):
        r = frozen_runs[a]; ax[2].plot(TF[1:], r["read_world"].mean(1)[1:], color=f"C{i}", label=f"a = {a}"); ax[2].plot(TF[1:], r["read_own"].mean(1)[1:], color=f"C{i}", ls=":")
        ax[2].axhline(a, color=f"C{i}", ls="--", alpha=0.5)
    ax[2].set(xscale="log", xlabel="own batches (generator frozen)", ylabel="share read right", title=f"sleep alone on a frozen generator, s = {FROZEN_S}")
    key = [plt.Line2D([], [], color="gray", ls=ls, label=lab) for ls, lab in [("-", "world strings"), (":", "own samples"), ("--", "writing level a")]]
    ax[2].legend(handles=ax[2].get_legend_handles_labels()[0] + key, ncols=2, loc="center right")
    fig.savefig(FIG, dpi=150, transparent=True); print(FIG)


if __name__ == "__main__": main()
