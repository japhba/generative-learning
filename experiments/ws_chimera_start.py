"""A student that is as deeply structured as the teacher, with different deep latents: does wake-sleep rectify it, and what do own samples do?
World and learner as in ws_reading_gain.py: K causes, each shows one fixed token (of V) in each of S views; generator pi_j(x_j | code),
recogniser q(code | x) trained only by sleep on own samples. Here the student does not start blank. Every code writes one fixed token per
view (sharp up to EPS), and in every view the K codes write the K different tokens, so single tokens are distributed as in the world:
  chimera r  code k writes the token of cause k in r of its S views and the token of some other cause in the others (other causes differ
             from view to view): K strings, none of them valid for r < S
  relabel    code k writes the whole string of cause perm(k): the teacher's distribution under other names (within-layer control)
  blank      small random logits (the start of ws_reading_gain.py): no deep structure
Protocol: PRESLEEP own batches with the generator frozen (the student has practised in its own language), then wake-sleep on world
batches with `dreams` own batches per world batch. Arms without the earlier practice and without dreams show what own samples are for.
Usage: python ws_chimera_start.py -> reports/ws-chimera-start.png, /var/tmp/jbauer/generative_practice_reading/ws_chimera_start.npz"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ws_reading_gain import B, EPS, K, KEYS, LR, SEEDS, V, draw, gather, score, sleep, sm

S, STEPS, PRESLEEP = 8, 2000, 2000
T = np.unique(np.r_[0, np.round(np.geomspace(1, STEPS, 60))].astype(int))
STARTS = [("blank", None), ("chimera", 0), ("chimera", 2), ("chimera", 3), ("chimera", 4), ("chimera", 6), ("relabel", None)]
ARMS = [(kind, r, PRESLEEP, 1) for kind, r in STARTS] + [("chimera", 4, PRESLEEP, 0), ("chimera", 4, 0, 1), ("chimera", 4, 0, 0), ("blank", None, PRESLEEP, 0)]   # (start, right views, earlier own batches, dreams per world batch)
OUT = Path("/var/tmp/jbauer/generative_practice_reading"); FIG = Path(__file__).resolve().parents[1] / "reports/ws-chimera-start.png"


def start(rng, sig, kind, r):                                                                               # generator logits [n, S, K, V] of the student before any world example
    if kind == "blank": return 0.1 * rng.standard_normal((SEEDS, S, K, V))
    tok = sig.copy()                                                                                        # tok[n, j, k]: token code k writes in view j
    if kind == "relabel": tok = np.take_along_axis(sig, np.broadcast_to(np.argsort(rng.random((SEEDS, 1, K)), -1), sig.shape), 2)
    else:
        right = np.argsort(rng.random((SEEDS, K, S)), -1).transpose(0, 2, 1) < r                            # exactly r right views per code
        for n in range(SEEDS):
            for j in range(S):
                w = rng.permutation(np.flatnonzero(~right[n, j]))                                           # codes that are wrong in this view swap tokens in a cycle (a lone one stays right)
                tok[n, j, w] = sig[n, j, np.roll(w, 1)]
    P = np.full((SEEDS, S, K, V), EPS / V); np.put_along_axis(P, tok[..., None], 1 - EPS + EPS / V, 3); return np.log(P)


def surface(G, sig):                                                                                        # total variation teacher vs own samples: single tokens, pairs of views, whole strings; entropy of own samples (nats, codes told apart)
    P = sm(G); n = np.arange(SEEDS)[:, None, None]; world = np.zeros_like(P); np.put_along_axis(world, sig[..., None], 1.0, 3)   # world[n, j, cause, token]
    one = 0.5 * np.abs(P.mean(2) - world.mean(2)).sum(-1).max(1)
    two = np.mean([0.5 * np.abs(np.einsum("nkv,nkw->nvw", P[:, i], P[:, j]) - np.einsum("nkv,nkw->nvw", world[:, i], world[:, j])).sum((1, 2)) / K for i, j in combinations(range(S), 2)], 0)
    pw = np.exp(gather(np.log(P), sig).sum(1)).sum(2) / K                                                   # probability the student gives each world string, [n, cause]
    return np.array([one, two, 0.5 * (np.abs(1 / K - pw).sum(1) + 1 - pw.sum(1)), np.log(K) - (P * np.log(P)).sum(-1).mean(2).sum(1)])


def run(kind, r, presleep, dreams, seed=0):
    rng = np.random.default_rng(seed); sig = np.argsort(rng.random((SEEDS, S, K)), -1); G = start(rng, sig, kind, r); R, b, log = np.zeros((SEEDS, S, K, V)), np.zeros((SEEDS, K)), []; surf = surface(G, sig)
    for _ in range(presleep): sleep(rng, G, R, b)
    for t in tqdm(range(STEPS + 1), desc=f"{kind} {r} presleep {presleep} dreams {dreams}", leave=False):
        if t in T: log.append(score(rng, G, R, b, sig))
        c = rng.integers(K, size=(SEEDS, B)); x = np.take_along_axis(sig, np.broadcast_to(c[:, None, :], (SEEDS, S, B)), 2)
        oc = np.eye(K)[draw(rng, b[:, None] + gather(R, x).sum(1))]                                         # wake: sampled code of each world string
        G += LR * (np.einsum("nbk,njbv->njkv", oc, np.eye(V)[x]) - oc.sum(1)[:, None, :, None] * sm(G)) / B
        for _ in range(dreams): sleep(rng, G, R, b)
    return {k: np.array([row[k] for row in log]) for k in KEYS} | {"surface": surf}                         # [time, seed]; surface [4, seed]


def onset(y, level): hit = y >= level; return np.where(hit.any(0), T[hit.argmax(0)], np.nan)
def nmed(v): return np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True); res = {arm: run(*arm) for arm in ARMS}
    np.savez(OUT / "ws_chimera_start.npz", t=T, **{f"{'_'.join(map(str, arm))}/{k}": v for arm, r in res.items() for k, v in r.items()})
    name = lambda arm: f"{arm[0]}{'' if arm[1] is None else f' r={arm[1]}'}, earlier practice {arm[2]}, dreams {arm[3]}"; at = lambda y, t: np.median(y[np.searchsorted(T, t)])
    print("start: total variation teacher vs own (single tokens | pairs of views | whole strings), entropy of own samples in nats (teacher: {:.2f} + smoothing)".format(np.log(K)))
    for arm in ARMS[:len(STARTS)]: print(f"  {name(arm):45s} " + "  ".join(f"{v:.3f}" for v in res[arm]["surface"].mean(1)))
    print("before any world example: reads world strings | reads own samples | one view written right | valid strings written   ||   world batches until valid >= 0.5, >= 0.9 (seeds) | KL(teacher || student) at 0, 10, 100, 1000 world batches | valid at the end")
    for arm in ARMS:
        r = res[arm]; o5, o9 = onset(r["valid"], 0.5), onset(r["valid"], 0.9)
        print(f"  {name(arm):45s} {at(r['read_world'], 0):.2f} | {at(r['read_own'], 0):.2f} | {at(r['a'], 0):.2f} | {at(r['valid'], 0):.1e}   ||   {nmed(o5):6.0f} ({np.isfinite(o5).sum():2d}/{SEEDS})  {nmed(o9):6.0f} ({np.isfinite(o9).sum():2d}/{SEEDS}) | "
              + "  ".join(f"{at(r['kl'], t):.2f}" for t in (0, 10, 100, 1000)) + f" | {np.median(r['valid'][-1]):.2f} (worst seed {r['valid'][-1].min():.2f})")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True); med = lambda y: np.median(y, 1)
    for i, arm in enumerate(ARMS[:len(STARTS)]):
        lab = {"blank": "blank start", "relabel": "relabelled teacher (control)"}.get(arm[0], f"chimera, {arm[1]} of {S} views right")
        ax[0].plot(T + 1, med(res[arm]["kl"]), color=f"C{i}", label=lab); ax[1].plot(T + 1, med(res[arm]["valid"]), color=f"C{i}", label=lab)
    ax[0].set(xscale="log", yscale="log", xlabel="world batches + 1", ylabel="KL(teacher || student)", title="how fast the student is rectified"); ax[1].set(xscale="log", xlabel="world batches + 1", ylabel="share of own samples that are valid", title="what it writes")
    for arm, ls, lab in [(("chimera", 4, PRESLEEP, 1), "-", "earlier practice, dreams"), (("chimera", 4, PRESLEEP, 0), "--", "earlier practice, no dreams"), (("chimera", 4, 0, 1), "-.", "no earlier practice, dreams"), (("chimera", 4, 0, 0), ":", "no own samples at all")]:
        ax[2].plot(T + 1, med(res[arm]["valid"]), color="C4", ls=ls, label=f"chimera 4 of {S}: {lab}")
    for arm, ls, lab in [(("blank", None, PRESLEEP, 1), "-", "dreams"), (("blank", None, PRESLEEP, 0), ":", "no dreams")]: ax[2].plot(T + 1, med(res[arm]["valid"]), color="C0", ls=ls, label=f"blank: {lab}")
    ax[2].set(xscale="log", xlabel="world batches + 1", ylabel="share of own samples that are valid", title="what own samples are for")
    for a in ax: a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)


if __name__ == "__main__": main()
