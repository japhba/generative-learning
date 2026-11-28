"""Practice by estimating the EM responsibilities with recognized own trajectories."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from folding_trajectory import trajectory, tent


PEAKS = np.linspace(.18, .82, 48)
BINS, WATCH, BATCH, ROUNDS = 12, 128, 128, 48
ARMS = ("Adaptive odds", "Frozen odds", "Shuffled odds", "Watched exact EM", "Frozen exact odds",
        "Fresh posterior", "Fresh peak readback")
EXTRA = ("Centered trials", "Centered frozen", "Centered shuffled", "Learned E-step", "Foreign E-step")
METRICS = ("path_kl", "sensory_kl", "watched_kl", "regime_acc", "regime_logloss", "smaller_side", "effective_peaks")


def senses():
    # Each column is the exact distribution of one binned transition of that writer cell.
    edges = np.arange(BINS+1)/BINS
    l, r = edges[:-1, None, None], edges[1:, None, None]
    u, v = edges[None, :-1, None], edges[None, 1:, None]
    left = np.maximum(0, np.minimum(r, PEAKS*v) - np.maximum(l, PEAKS*u))
    right = np.maximum(0, np.minimum(r, 1-(1-PEAKS)*u) - np.maximum(l, 1-(1-PEAKS)*v))
    full = (left + right).reshape(BINS**2, -1)
    active = full.sum(1) > 0
    channel = full[active]
    assert np.allclose(channel.sum(0), 1)
    base = channel.mean(1)
    x, y = (np.indices((BINS, BINS)).reshape(2, -1)[:, active] + .5)/BINS
    q0 = np.exp(-.5*(BINS*(tent(x[:, None], PEAKS)-y[:, None]))**2)
    q0 /= q0.sum(1, keepdims=True)
    kernel0 = base[:, None] * q0
    kernel0 /= kernel0.sum(0)
    return channel, base, q0, kernel0, np.cumsum(active)-1


def teacher(rng):
    centers = rng.uniform([.26, .59], [.41, .74])
    peaks = PEAKS.reshape(2, -1)
    w = np.exp(-.5*((peaks-centers[:, None])/.028)**2)
    return (.5*w/w.sum(1, keepdims=True)).ravel()


def batch(pi, u, starts, lookup):
    chosen = np.searchsorted(np.cumsum(pi), u)
    x = trajectory(starts, PEAKS[chosen])
    # The model reads the selected peak through the generated word before crediting it.
    j = np.argmin(np.abs(tent(x[:, :1], PEAKS)-x[:, 1:2]), axis=1)
    assert np.array_equal(chosen, j)
    bins = np.minimum((BINS*x).astype(int), BINS-1)
    s = lookup[bins[:, :-1]*BINS + bins[:, 1:]]
    return j, s


def freq(s, n):
    return np.bincount(s.ravel(), minlength=n)/(s.shape[1])


def kl(p, q):
    return np.sum(p*np.log(p/q))


def metrics(pi, actual, channel, anchor, learned):
    observed = channel @ actual
    target_sides = np.column_stack([(channel[:, sl] @ actual[sl]) for sl in (PEAKS<.5, PEAKS>.5)])
    read = np.column_stack([(learned[:, sl] @ pi[sl]) for sl in (PEAKS<.5, PEAKS>.5)])
    read /= read.sum(1, keepdims=True)
    accuracy = target_sides[np.arange(len(read)), np.argmax(read, axis=1)].sum()
    logloss = -np.sum(target_sides*np.log(np.maximum(read, 1e-300)))
    sides = pi.reshape(2, -1).sum(1)
    return [kl(actual, pi), kl(observed, channel @ pi), kl(anchor, channel @ pi), accuracy,
            logloss, np.min(sides), np.exp(-np.sum(pi*np.log(pi)))]


def per_world(seed, design, supplement):
    channel, base, q0, kernel0, lookup = design
    n_s, n_j = channel.shape
    rng = np.random.default_rng(seed)
    actual = teacher(rng)
    _, watched = batch(actual, rng.random(WATCH), rng.random(WATCH), lookup)
    anchor = (freq(watched, n_s)+2*base)/(WATCH+2)
    warm = anchor @ q0
    arms = (*ARMS, *EXTRA) if supplement else ARMS
    pi = np.tile(warm, (len(arms), 1))
    modeled = [0, 1, 2] + (list(range(7, 12)) if supplement else [])
    joint = {i: 2*kernel0.copy() for i in modeled}
    frozen_exact = anchor/(channel @ warm)
    frozen_sample = None
    fresh_counts = WATCH*warm
    stats = np.empty((ROUNDS+1, len(arms), len(METRICS)))
    for t in range(ROUNDS+1):
        for i, p in enumerate(pi):
            learned = joint[i]/joint[i].sum(0) if i in joint else channel
            stats[t, i] = metrics(p, actual, channel, anchor, learned)
        if t == ROUNDS:
            break
        u, starts = rng.random((2, BATCH))
        shuffle = rng.permutation(BATCH)
        for i in modeled:
            source = np.full(n_j, 1/n_j) if i == 11 else pi[i]
            ids, s = batch(source, u, starts, lookup)
            flat = s*n_j+ids[:, None]
            joint[i] += np.bincount(flat.ravel(), minlength=n_s*n_j).reshape(n_s, n_j)/s.shape[1]
            if i in (10, 11):
                learned = joint[i]/joint[i].sum(0)
                weighted = pi[i]*((anchor/(learned @ pi[i])) @ learned)
                pi[i] = .5*pi[i]+.5*weighted
                continue
            halves = np.split(s, 2)
            other = [(freq(x, n_s)+8*base)/(len(x)+8) for x in halves[::-1]]
            odds = np.array([anchor/m for m in other])
            if t == 0 and i == 0:
                frozen_sample = odds.copy()
            if i in (1, 8):
                odds = frozen_sample
            reward = np.concatenate([r[x].mean(1) for r, x in zip(odds, halves)])
            if i in (2, 9):
                reward = reward[shuffle]
            weighted = np.bincount(ids, weights=reward, minlength=n_j)
            if i in (7, 8, 9):
                weighted = pi[i]*(weighted+1)/(np.bincount(ids, minlength=n_j)+1)
            pi[i] = .5*pi[i]+.5*weighted/weighted.sum()
        for i in (3, 4):
            ratio = anchor/(channel @ pi[i]) if i == 3 else frozen_exact
            weighted = pi[i]*(ratio @ channel)
            pi[i] = .5*pi[i]+.5*weighted/weighted.sum()
        ids, fresh = batch(actual, u, starts, lookup)
        hist = freq(fresh, n_s)/BATCH
        weighted = pi[5]*((hist/(channel @ pi[5])) @ channel)
        pi[5] = .5*pi[5]+.5*weighted/weighted.sum()
        fresh_counts += np.bincount(ids, minlength=n_j)
        pi[6] = fresh_counts/fresh_counts.sum()
    assert np.allclose(pi.sum(1), 1)
    assert np.max(np.diff(stats[:, 3, 2])) < 1e-12
    return stats, actual, anchor, warm, pi


def audit():
    channel, base, _, _, lookup = senses()
    rng = np.random.default_rng(602)
    actual, pi = teacher(rng), rng.dirichlet(np.ones(len(PEAKS)))
    h = channel @ actual
    for _ in range(30):
        m = channel @ pi
        new = pi*((h/m) @ channel)
        assert kl(actual, new) <= kl(actual, pi)-kl(h, m)+1e-12
        pi = new
    uniform_sensor = np.full((BINS,len(pi)), 1/BINS)
    empirical = rng.dirichlet(np.ones(BINS))
    assert np.allclose(pi*((empirical/(uniform_sensor @ pi)) @ uniform_sensor), pi)
    fixed_q = np.array([.8, .2])
    ratio = np.array([.5, .5])/fixed_q
    for predicted in ((.5,.5),(.2,.8)):
        fixed_q *= ratio
        fixed_q /= fixed_q.sum()
        assert np.allclose(fixed_q, predicted)
    for j in range(len(PEAKS)):
        x = (np.arange(20000)+.5)/20000
        b = np.minimum((BINS*np.stack([x, tent(x, PEAKS[j])])).astype(int), BINS-1)
        hist = np.bincount(lookup[b[0]*BINS+b[1]], minlength=len(base))/len(x)
        assert np.max(np.abs(hist-channel[:, j])) < .00011
    # The positive and negative parts of a symmetric null vector are distinct
    # writers with equal regime masses and the same distribution of recognized pairs.
    half = len(PEAKS)//2
    symmetric = channel[:, :half] + channel[:, half:][:, ::-1]
    _, _, vectors = np.linalg.svd(symmetric, full_matrices=False)
    v = np.r_[vectors[-1], vectors[-1, ::-1]]
    p, q = np.maximum(v, 0), np.maximum(-v, 0)
    p, q = .95*p/p.sum()+.05/len(p), .95*q/q.sum()+.05/len(q)
    assert np.allclose(channel @ p, channel @ q) and np.isclose(abs(p-q).sum()/2, .95)
    # At lag two the iterated tent has four full affine branches.
    edges = np.arange(BINS+1)/BINS
    l, r = edges[:-1,None,None], edges[1:,None,None]
    u, v = edges[None,:-1,None], edges[None,1:,None]
    knots = [np.zeros_like(PEAKS), PEAKS**2, PEAKS, 1-PEAKS*(1-PEAKS), np.ones_like(PEAKS)]
    pairs = np.zeros((BINS, BINS, len(PEAKS)))
    for i, (a, b) in enumerate(zip(knots, knots[1:])):
        width = b-a
        lo, hi = (a+width*u, a+width*v) if i%2 == 0 else (b-width*v, b-width*u)
        pairs += np.maximum(0, np.minimum(r, hi)-np.maximum(l, lo))
    assert np.linalg.matrix_rank(channel) == 31
    assert np.linalg.matrix_rank(pairs.reshape(-1,len(PEAKS))) == len(PEAKS)
    print("Verified channel integrals, KL bound, frozen overshoot, single-position control, indistinguishable writers and lag-two rank.")


def plot(data, continuous, dest):
    scores = data["scores"]
    fig, axes = plt.subplots(2, 1, sharex=True, layout="constrained")

    def band(ax, steps, runs, label, color):
        mean, se = runs.mean(0), runs.std(0, ddof=1)/np.sqrt(len(runs))
        ax.plot(steps, mean, label=label, color=color)
        ax.fill_between(steps, mean-1.96*se, mean+1.96*se, color=color, alpha=.15)

    w = list(continuous["watches"]).index(2048)
    for i, label, color in ((0, "Normalized judge", "C0"), (1, "Ordinary ReLU judge", "C1"), (2, "Shuffled verdict", "0.5")):
        band(axes[0], continuous["rounds"], continuous["scores"][w, :, i, :, 0].T, label, color)
    for name, label, color in (("Centered trials", "Adaptive verdict", "C0"), ("Centered frozen", "Frozen verdict", "C1"),
                               ("Learned E-step", "Learned E-step", "C2"), ("Fresh posterior", "Fresh teacher", "C3")):
        i = list(data["arms"]).index(name)
        band(axes[1], np.arange(scores.shape[1]), scores[:, :, i, 0], label, color)
    for ax in axes:
        ax.axvline(16, color="0.6", linestyle=":")
        ax.legend(ncols=2)
    axes[0].set(ylabel="Ten-step success")
    axes[1].set(ylabel="Trajectory KL (nats)", yscale="log", xlabel="Rounds after Watching")
    fig.savefig(dest.with_suffix(".svg"), transparent=True)
    fig.savefig(dest.with_suffix(".png"), facecolor="white", dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--continuous", type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--worlds", type=int, default=96)
    parser.add_argument("--seed", type=int, default=281129)
    args = parser.parse_args()
    if args.audit:
        audit()
    elif args.plot:
        plot(np.load(args.out), np.load(args.continuous), args.plot)
    else:
        design = senses()
        runs = [per_world(args.seed+i, design, args.supplement) for i in tqdm(range(args.worlds), desc="paired worlds")]
        scores, actual, anchor, warm, final = map(np.array, zip(*runs))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, scores=scores, actual=actual, anchor=anchor, warm=warm, final=final,
                            channel=design[0], peaks=PEAKS, arms=(*ARMS, *EXTRA) if args.supplement else ARMS,
                            metrics=METRICS, seed=args.seed)
