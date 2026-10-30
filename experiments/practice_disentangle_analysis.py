"""Pre-registered claims (DISENTANGLE_PROTOCOL.md) and figures for the wake-sleep disentangling confirmation."""

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

AGNOSTIC = ["wake_frozen", "wake_fresh", "binary_ae", "linear_ae", "fastica"]
ASSIGNED = ["binary_ae_assigned", "fastica_assigned"]
STATIC = ["fastica", "fastica_assigned", "binary_ae_assigned", "init"]
LABEL = {"wake_sleep_bind": "wake-sleep + binding moves", "vae_bind": "VAE + binding moves", "binary_ae_bind": "binary AE + binding moves",
         "wake_sleep": "wake-sleep (practice)", "wake_attend": "in-context practice (attention over own samples)",
         "wake_frozen": "wake, recognizer frozen", "wake_fresh": "wake + fresh foreign samples, recognizer frozen",
         "binary_ae": "binary autoencoder", "linear_ae": "linear autoencoder", "vae": "VAE (exact prior density)",
         "exact_em": "exact EM (prior density, 2^K)", "wake_sleep_stream": "wake-sleep, unlimited fresh data (reference)",
         "fastica": "FastICA (one shot)", "fastica_assigned": "FastICA + relabeling", "binary_ae_assigned": "binary AE + relabeling"}


def load(root, pattern="*_conf*"):
    cells = defaultdict(dict)  # (prior, N) -> name -> dict(final, best, curves)
    for path in glob.glob(f"{root}/{pattern}/results.json"):
        name = Path(path).parent.name; prior = name.split("_")[0]; N = int(re.search(r"_N(\d+)_", name).group(1))
        r = json.load(open(path)); cell = cells[(prior, N)]; cell.setdefault("_weights", {}).update(r["weights"])
        for m, runs in r["results"].items():
            kl = np.array([[p["kl"] for p in run] for run in runs])
            cell[m] = dict(final=kl[:, -1], best=kl.min(1), kl=kl, steps=np.array([p["step"] for p in runs[0]]),
                           acc=np.array([[p["acc_sym" if prior == "chain" else "acc_matched"] for p in run] for run in runs]), recon=np.array([[p["recon"] for p in run] for run in runs]))
        for key in STATIC + [f"{m}_assigned" for m in r["results"]]:
            if key in r["static"][0]:
                v = np.array([s[key] for s in r["static"]]); cell[key] = dict(final=v, best=v)
    return cells


def boot(x, rng, n=10000):
    idx = rng.integers(0, len(x), (n, len(x))); m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def compare(cell, practice, rival, rng):
    a, b = cell[practice]["final"], cell[rival]["best"]
    lr = np.log(a) - np.log(b); mean, lo, hi = boot(lr, rng)
    return dict(practice=float(a.mean()), rival=float(b.mean()), log_ratio=mean, ci=[lo, hi], wins=int((a < b).sum()), n=len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/var/tmp/jbauer/generative_practice_disentangle/confirm")
    ap.add_argument("--budget", default="/var/tmp/jbauer/generative_practice_disentangle/budget")
    ap.add_argument("--bars", default="/var/tmp/jbauer/generative_practice_disentangle/bars")
    ap.add_argument("--figs", default="/nfs/nhome/live/jbauer/generative-learning/tex/figures")
    ap.add_argument("--bind", default="")  # root of the amendment-1 run: test B1/B2 only
    args = ap.parse_args(); rng = np.random.default_rng(0)
    if args.bind: return bind_analysis(args, rng)
    cells = load(args.root); claims = {}
    rivals = AGNOSTIC + ASSIGNED
    for practice, tag in [("wake_sleep", "C1"), ("wake_attend", "C4")]:
        for prior in ["chain", "deep"]:
            res = {r: compare(cells[(prior, 256)], practice, r, rng) for r in rivals}
            claims[f"{tag}_{prior}"] = dict(passed=all(v["ci"][1] < 0 and v["wins"] >= 14 for v in res.values()), comparisons=res)
    c2 = {f"{p}_N{N}": compare(cells[(p, N)], "wake_sleep", "wake_fresh", rng) for p in ["factorial", "chain", "deep"] for N in [64, 256, 1024]}
    claims["C2"] = dict(passed=all(v["ci"][1] < 0 for v in c2.values()), comparisons=c2)
    desc = {f"{p}_N{N}_{r}": compare(cells[(p, N)], "wake_sleep", r, rng) for p in ["factorial", "chain", "deep"] for N in [64, 256, 1024] for r in ["vae", "exact_em", "fastica", "wake_attend"] if r in cells[(p, N)]}
    claims["descriptive_vs_density_users"] = desc
    # C3 budget split
    c3 = {}; budget = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in glob.glob(f"{args.budget}/*_ss10.0_seed*/results.json"):
        prior = Path(path).parent.name.split("_")[0]; r = json.load(open(path))
        for N, d in r["split"].items():
            for f, runs in d.items(): budget[prior][int(N)][float(f)] += [run[-1] for run in runs]
    for prior, byN in budget.items():
        for N, byf in byN.items():
            a, b = np.array(byf[0.5]), np.array(byf[0.0]); mean, lo, hi = boot(np.log(a) - np.log(b), rng)
            c3[f"{prior}_N{N}"] = dict(f05=float(a.mean()), f0=float(b.mean()), log_ratio=mean, ci=[lo, hi], n=len(a),
                                       best_f=float(min(byf, key=lambda f: np.mean(np.log(byf[f])))))
    claims["C3"] = dict(passed=bool(c3) and all(v["ci"][1] < 0 for v in c3.values()), comparisons=c3)
    json.dump(claims, open(Path(args.root) / "claims.json", "w"), indent=1)
    for k, v in claims.items():
        if "passed" in v: print(k, "PASSED" if v["passed"] else "FAILED")
        for name, c in v.get("comparisons", v).items():
            print(f"   {name:34s} practice={c.get('practice', c.get('f05')):.3f} rival={c.get('rival', c.get('f0')):.3f} log-ratio={c['log_ratio']:+.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}]" + (f" wins {c['wins']}/{c['n']}" if "wins" in c else f" best f={c['best_f']}"))
    figures(cells, budget, args)


STYLE = {"wake_sleep_bind": ("C0", "--"), "vae_bind": ("C2", "--"), "binary_ae_bind": ("C4", "--"), "wake_sleep": ("C0", "-"), "wake_attend": ("C1", "-"), "vae": ("C2", "-"), "exact_em": ("C3", "-"),
         "binary_ae": ("C4", "-"), "binary_ae_assigned": ("C4", ":"), "wake_fresh": ("C5", "-"), "wake_frozen": ("C6", "-"),
         "linear_ae": ("C7", "-"), "fastica": ("C8", "--"), "fastica_assigned": ("C8", ":"), "wake_sleep_stream": ("k", ":")}


def band(ax, x, Y, m):
    c, ls = STYLE[m]; mu, se = Y.mean(0), Y.std(0) / np.sqrt(len(Y))
    ax.plot(x, mu, color=c, ls=ls, label=LABEL[m]); ax.fill_between(x, mu - se, mu + se, alpha=0.15, color=c)


def figures(cells, budget, args):
    priors = ["factorial", "chain", "deep"]; order = ["wake_sleep", "wake_attend", "vae", "exact_em", "binary_ae", "wake_fresh", "wake_frozen", "linear_ae"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for ax, prior in zip(axes, priors):
        cell = cells[(prior, 256)]
        for m in order + ["wake_sleep_stream"]: band(ax, cell[m]["steps"], cell[m]["kl"], m)
        for m in ["fastica", "fastica_assigned", "binary_ae_assigned"]: ax.axhline(cell[m]["final"].mean(), color=STYLE[m][0], ls=STYLE[m][1], label=LABEL[m])
        ax.set_yscale("log"); ax.set_xlabel("update"); ax.set_title(f"{prior} prior, N = 256 foreign samples")
    axes[0].set_ylabel("KL(true ‖ model) [nats]")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(f"{args.figs}/practice-disentangle-learning.png", dpi=130)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, prior in zip(axes, priors):
        cell = cells[(prior, 256)]
        for m in ["wake_sleep", "wake_attend", "vae", "binary_ae", "wake_fresh"]: band(ax, cell[m]["steps"], cell[m]["acc"], m)
        ax.set_xlabel("update"); ax.set_title(f"{prior} prior" + (" (up to the prior's symmetries)" if prior == "chain" else " (up to signed permutation)"))
    axes[0].set_ylabel("foreign samples: bit accuracy")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f"{args.figs}/practice-disentangle-recognition.png", dpi=130)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True); Ns = [64, 256, 1024]
    for ax, prior in zip(axes, priors):
        for m in ["wake_sleep", "wake_attend", "vae", "exact_em", "binary_ae_assigned", "fastica_assigned", "wake_fresh", "wake_sleep_stream"]:
            key = "final" if m in ["wake_sleep", "wake_attend", "wake_sleep_stream"] else "best"
            vals = [cells[(prior, N)][m][key] for N in Ns]
            ax.errorbar(Ns, [v.mean() for v in vals], [v.std() / np.sqrt(len(v)) for v in vals], marker="o", color=STYLE[m][0], ls=STYLE[m][1], label=LABEL[m])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("stored foreign samples N"); ax.set_title(f"{prior} prior")
    axes[0].set_ylabel("KL [nats]")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(f"{args.figs}/practice-disentangle-N.png", dpi=130)
    if budget:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
        for ax, prior in zip(axes, priors):
            for N in sorted(budget[prior]):
                fs = sorted(budget[prior][N]); vals = [np.array(budget[prior][N][f]) for f in fs]
                ax.errorbar(fs, [v.mean() for v in vals], [v.std() / np.sqrt(len(v)) for v in vals], marker="o", label=f"N = {N} stored" if N else "no stored data (stream)")
            ax.set_yscale("log"); ax.set_xlabel("fraction of extra updates spent on own samples"); ax.set_title(f"{prior} prior")
        axes[0].set_ylabel("final KL [nats]"); axes[0].legend(); fig.tight_layout(); fig.savefig(f"{args.figs}/practice-disentangle-budget.png", dpi=130)
    alignment(cells, args)


def alignment(cells, args):
    """Cosine between learner units (rows) and true factors in the learner's own concept order (columns), chain prior."""
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from practice_disentangle import World, assign
    cell = cells[("chain", 256)]; w = int(np.argsort(cell["wake_sleep"]["final"])[len(cell["wake_sleep"]["final"]) // 2])  # median world
    rng = np.random.default_rng([0, w]); world = World(6, 12, 0.5, "chain", rng); Xdata, _ = world.sample_x(256, rng)
    Wb = np.array(cell["_weights"]["binary_ae"][w]["W"]); Wa, _ = assign(Wb, np.zeros(12) + Xdata.mean(0), Xdata, world)
    panels = [("start", cell["_weights"]["init"][w]["W"], cell["wake_sleep"]["kl"][w, 0])]
    panels += [(LABEL[m], cell["_weights"][m][w]["W"], cell[m]["final"][w]) for m in ["wake_sleep", "wake_attend", "vae", "exact_em", "binary_ae"]]
    panels += [(LABEL["binary_ae_assigned"], Wa, cell["binary_ae_assigned"]["final"][w])]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.6))
    for ax, (lab, W, kl) in zip(axes, panels):
        W = np.array(W); C = (W / np.linalg.norm(W, axis=0)).T @ world.A
        ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1); ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_title(f"{lab}\nKL {kl:.2f}"); ax.set_xlabel("true factor (own concept order)")
    axes[0].set_ylabel("learner unit"); fig.tight_layout(); fig.savefig(f"{args.figs}/practice-disentangle-alignment.png", dpi=130)


def binding_rate(W, A, prior):
    from itertools import permutations
    perms = np.array(list(permutations(range(A.shape[1])))); K = A.shape[1]
    C = (W / np.linalg.norm(W, axis=0)).T @ (A / np.linalg.norm(A, axis=0))
    p = perms[np.abs(C[np.arange(K)[None, :], perms]).sum(1).argmax()]; sg = np.sign(C[np.arange(K), p])
    if prior == "factorial": return True
    ok = [list(range(K))] + ([list(range(K - 1, -1, -1))] if prior == "chain" else [])
    return list(p) in ok and abs(sg.sum()) == K


def bind_analysis(args, rng):
    cells = load(args.bind, "*_b[0-9]*"); claims = {}
    for prior in ["chain", "deep"]:
        cell = cells[(prior, 256)]
        res = {r: compare(cell, "wake_sleep_bind", r, rng) for r in ["binary_ae_bind", "binary_ae_assigned"]}
        claims[f"B1_{prior}"] = dict(passed=all(v["ci"][1] < 0 and v["wins"] >= 14 for v in res.values()), comparisons=res)
    a, b = cells[("chain", 256)]["wake_sleep_bind"]["final"], cells[("chain", 256)]["wake_sleep"]["final"]; mean, lo, hi = boot(np.log(a) - np.log(b), rng)
    claims["B2_chain"] = dict(passed=hi < 0, comparisons={"wake_sleep (final)": dict(practice=float(a.mean()), rival=float(b.mean()), log_ratio=mean, ci=[lo, hi], wins=int((a < b).sum()), n=len(a))})
    claims["descriptive"] = {f"{p}_{r}": compare(cells[(p, 256)], "wake_sleep_bind", r, rng) for p in ["factorial", "chain", "deep"] for r in ["vae_bind", "vae", "exact_em", "wake_fresh", "wake_sleep", "binary_ae_bind", "binary_ae_assigned"]}
    rates = {}
    for p in ["chain", "deep"]:
        cell = cells[(p, 256)]; A = [np.array(w["W"]) for w in cell["_weights"]["true"]]
        for m in ["wake_sleep_bind", "wake_sleep", "vae_bind", "vae", "exact_em", "binary_ae_bind", "binary_ae"]:
            rates[f"{p}_{m}"] = int(sum(binding_rate(np.array(w["W"]), A[i], p) for i, w in enumerate(cell["_weights"][m])))
    claims["binding_rate_of_16"] = rates
    json.dump(claims, open(Path(args.bind) / "claims.json", "w"), indent=1)
    for k, v in claims.items():
        if "passed" in v: print(k, "PASSED" if v["passed"] else "FAILED")
        for name, c in v.get("comparisons", v).items():
            if isinstance(c, dict): print(f"   {name:30s} practice={c['practice']:.3f} rival={c['rival']:.3f} log-ratio={c['log_ratio']:+.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] wins {c['wins']}/{c['n']}")
            else: print(f"   {name:30s} {c}/16")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
    for ax, prior in zip(axes, ["factorial", "chain", "deep"]):
        cell = cells[(prior, 256)]
        for m in ["wake_sleep_bind", "wake_sleep", "vae_bind", "vae", "exact_em", "binary_ae_bind", "wake_fresh"]: band(ax, cell[m]["steps"], cell[m]["kl"], m)
        ax.axhline(cell["binary_ae_assigned"]["final"].mean(), color=STYLE["binary_ae_assigned"][0], ls=STYLE["binary_ae_assigned"][1], label=LABEL["binary_ae_assigned"])
        ax.set_yscale("log"); ax.set_xlabel("update"); ax.set_title(f"{prior} prior, N = 256 (new worlds)")
    axes[0].set_ylabel("KL(true ‖ model) [nats]"); h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(f"{args.figs}/practice-disentangle-binding.png", dpi=130)


if __name__ == "__main__": main()
