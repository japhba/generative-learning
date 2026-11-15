"""Controls for the review of model4_feasibility.py: is the decoder's generation needed, and is the constraint memory's prior matched to the world?

Arms continue from the same end-of-Watching state as the original study (seeds 300-331, 16 demonstrations, batch 128, learning rate 0.5):
  practice eps      the original forward-KL practice with a share eps of uniform strings read by the encoder (eps 0.2 = final model, 0 = pure ancestral,
                    1 = no string comes from the decoder)
  watch stored / fresh   the original Watching rule on the 16 demonstrations / on fresh teacher strings
  watch memory      the Watching rule on strings drawn from the learned constraint model pi_X itself: no decoder samples, no importance weights
`--bias b` draws the three latent signs with P(+1) = b (the constraint memory can only express a uniform distribution on a parity code).
"""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import model4_feasibility as m4

OUT = Path("/nfs/gatsbystor/jbauer/generative-learning/model4_feasibility_review")
FIG = Path(__file__).resolve().parent.parent / "reports" / "model4-feasibility-review.png"


def world(seed, n, bias):
    if bias == .5: return m4.world(seed, n, "all")
    rng = np.random.default_rng(seed); mask = rng.choice([-1., 1.], 6)
    latent = np.array(list(product([-1., 1.], repeat=3))); z, u, v = latent.T
    probs = np.where(latent > 0, bias, 1 - bias).prod(1)
    ix = ((np.stack([u, z*u, z, v, z*v, z], 1) * mask + 1) / 2 @ (2 ** np.arange(5, -1, -1))).astype(int)
    pop = np.zeros(64); pop[ix] = probs
    draws = rng.choice(ix, n, p=probs); empirical = np.bincount(draws, minlength=64) / n
    moments = empirical @ m4.CHI; selected = [j for j in range(len(m4.FEATURES)) if abs(moments[j]) >= 1 - 1e-10]
    energy = .5 * np.log(2*n + 1) * m4.CHI[:, selected] @ np.sign(moments[selected])
    return dict(mask=mask, latent=latent, ix=ix, pop=pop, empirical=empirical, energy=energy, logpi=energy - m4.lse(energy), selected=selected, draws=draws)


def metrics(params, data):
    lp, lq, *_ = m4.tables(params); logpx = m4.lse(lp, 1); ix = data["ix"]; pop = data["pop"][ix]
    qz = np.exp(lq).reshape(64, 2, 4, 4).sum((2, 3)); joint = np.zeros((2, 2)); np.add.at(joint, ((data["latent"][:, 0] + 1) / 2).astype(int), pop[:, None] * qz[ix])
    pz = joint.sum(1)
    return dict(kl=float(pop @ (np.log(pop) - logpx[ix])), validity=float(np.exp(logpx[ix]).sum()), root_mi=float(m4.information(joint) / -(pz * np.log(pz)).sum()))


def practice(params, data, rng, batch, eps):
    lp, lq, pt, pd, pr, ph = m4.tables(params); q = np.exp(lq); energy = data["energy"]
    nown = rng.binomial(batch, 1 - eps); counts = m4.own_counts(params, rng, nown)
    xi = rng.integers(64, size=batch - nown); hi = (rng.random((batch - nown, 1)) > q[xi].cumsum(1)).sum(1); np.add.at(counts, (xi, hi), 1)
    with np.errstate(divide="ignore"): ratio = energy[:, None] + lq - np.logaddexp(np.log(1 - eps) + lp, np.log(eps / 64) + lq)
    active = counts > 0; weights = np.zeros_like(lp); weights[active] = counts[active] * np.exp(ratio[active] - ratio[active].max()); weights /= weights.sum()
    return [*m4.decoder_grad(weights, pt, pd), *m4.encoder_grad(weights * (lp - lq + 6*np.log(2)), pr, ph)]


def watch(params, dist, rng, batch, exact_baseline=True):
    lp, lq, pt, pd, pr, ph = m4.tables(params); q = np.exp(lq); reward = lp - lq
    weights = rng.multinomial(batch, (dist[:, None] * q).ravel()).reshape(lp.shape) / batch
    return [*m4.decoder_grad(weights, pt, pd), *m4.encoder_grad(weights * (reward - ((q*reward).sum(1)[:, None] if exact_baseline else -6*np.log(2))), pr, ph)]


ARMS = {
    "practice eps 0.2": lambda par, data, rng, batch: practice(par, data, rng, batch, .2),
    "practice eps 0": lambda par, data, rng, batch: practice(par, data, rng, batch, 0.),
    "practice eps 0.5": lambda par, data, rng, batch: practice(par, data, rng, batch, .5),
    "practice eps 1": lambda par, data, rng, batch: practice(par, data, rng, batch, 1.),
    "watch stored": lambda par, data, rng, batch: watch(par, data["empirical"], rng, batch),
    "watch fresh": lambda par, data, rng, batch: watch(par, data["pop"], rng, batch),
    "watch memory": lambda par, data, rng, batch: watch(par, np.exp(data["logpi"]), rng, batch),
    "watch memory, fixed baseline": lambda par, data, rng, batch: watch(par, np.exp(data["logpi"]), rng, batch, False),
}


def run(spec):
    seed, n, bias, steps, watchsteps, lr, batch, arms = spec
    data = world(seed, n, bias); params = m4.init(seed, .8); rng = np.random.default_rng(seed + 200000)
    for _ in range(watchsteps): params = [a + lr*b for a, b in zip(params, m4.grads(params, data, "watch", rng, batch))]
    timeline = np.unique(np.r_[0, np.round(np.geomspace(1, steps, 70))]).astype(int); curves = {}
    for arm in arms:
        par = [a.copy() for a in params]; armrng = np.random.default_rng(seed + 300000); rows = []
        for t in range(steps + 1):
            if t in timeline: rows.append(dict(step=int(t), **metrics(par, data)))
            if t < steps: par = [a + lr*b for a, b in zip(par, ARMS[arm](par, data, armrng, batch))]
        curves[arm] = rows
    pop = data["pop"][data["ix"]]
    return dict(seed=seed, n=n, bias=bias, watchsteps=watchsteps, memory_kl=float(pop @ (np.log(pop) - data["logpi"][data["ix"]])),
                false_constraints=int(sum(abs(data["pop"] @ m4.CHI[:, j]) < 1 - 1e-10 for j in data["selected"])), curves=curves)


def table(runs):
    for arm in runs[0]["curves"]:
        kl = np.array([r["curves"][arm][-1]["kl"] for r in runs]); root = np.array([r["curves"][arm][-1]["root_mi"] for r in runs])
        print(f"{arm:30s} KL mean {kl.mean():.3f} +- {kl.std(ddof=1)/np.sqrt(len(kl)):.3f}  median {np.median(kl):.3f}  root information {root.mean():.3f}")
    print(f"constraint memory itself: KL mean {np.mean([r['memory_kl'] for r in runs]):.4f}, worlds with a false constraint {sum(r['false_constraints'] > 0 for r in runs)} of {len(runs)}")


def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    runs = json.loads((OUT / "controls.json").read_text())["runs"]
    shown = {"watch stored": "Watching, the 16 stored strings", "watch fresh": "Watching, fresh teacher strings", "practice eps 0.2": "practice (report's final rule)",
             "watch memory": "Watching, strings drawn from the constraint memory", "practice eps 1": "practice, no string from the decoder"}
    steps = np.array([row["step"] for row in runs[0]["curves"]["watch stored"]])
    fig, axes = plt.subplots(1, 2, figsize=plt.figaspect(.42), constrained_layout=True)
    for arm, label in shown.items():
        for ax, key in zip(axes, ["kl", "root_mi"]):
            c = np.array([[row[key] for row in r["curves"][arm]] for r in runs])
            ax.plot(steps[1:], np.median(c, 0)[1:], label=label, ls="--" if arm in ["watch memory", "practice eps 1"] else "-")
    axes[0].axhline(np.mean([r["memory_kl"] for r in runs]), color="k", ls=":", label="the constraint memory itself")
    axes[0].set(xscale="log", yscale="log", xlabel="network updates after Watching", ylabel="population KL (nats)")
    axes[1].set(xscale="log", xlabel="network updates after Watching", ylabel="root read by the encoder (share of entropy)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3); fig.savefig(FIG, facecolor="white"); print(FIG)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--name", default="controls"); p.add_argument("--bias", type=float, default=.5); p.add_argument("--n", type=int, default=16)
    p.add_argument("--watchsteps", type=int, default=100); p.add_argument("--steps", type=int, default=2000); p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--workers", type=int, default=16); p.add_argument("--plot", action="store_true"); args = p.parse_args()
    if args.plot: plot(); return
    specs = [(s, args.n, args.bias, args.steps, args.watchsteps, .5, 128, args.arms) for s in range(300, 332)]
    with ProcessPoolExecutor(args.workers) as pool: runs = list(tqdm(pool.map(run, specs), total=len(specs), desc=args.name))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"{args.name}.json").write_text(json.dumps(dict(configuration=vars(args), runs=runs), separators=(",", ":")))
    print(vars(args)); table(runs)


if __name__ == "__main__": main()
