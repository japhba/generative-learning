"""Figures for the follow-ups to the disentangle toy: more supervision per practice step, and learning a lesson from own
vs foreign examples."""

import glob
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = "/var/tmp/jbauer/generative_practice_disentangle"; FIGS = "/nfs/nhome/live/jbauer/generative-learning/tex/figures"


def supervision():
    panels = [("chain", 0, "rotated", "chain prior, fresh foreign data"), ("chain", 256, "rotated", "chain prior, 256 stored samples"),
              ("factorial", 0, "rotated", "independent prior, fresh foreign data"), ("chain", 0, "swapped", "chain prior, fresh data, swapped start")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.3), sharey=False)
    for ax, (prior, N, init, title) in zip(axes, panels):
        r = json.load(open(glob.glob(f"{ROOT}/supervision/{prior}_N{N}_{init}_2*/results.json")[0]))
        for i, (ratio, curves) in enumerate(r["results"].items()):
            t = [c["step"] for c in curves[0]]; kl = np.array([[c["kl"] for c in cu] for cu in curves])
            ax.plot(t, kl.mean(0), color=plt.cm.viridis(i / (len(r["results"]) - 1)), label=f"{float(ratio):g}")
        ax.set_yscale("log"); ax.set_title(title); ax.set_xlabel("practice (sleep) steps")
    axes[0].set_ylabel("KL(world || learner), nats"); axes[0].legend(title="foreign batches\nper own batch", fontsize="small")
    fig.tight_layout(); out = f"{FIGS}/practice-disentangle-supervision.png"; fig.savefig(out, dpi=110); print(out)


def own():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for col, prior in enumerate(["chain", "factorial"]):
        rows = json.load(open(glob.glob(f"{ROOT}/own/{prior}_rotated_2*/results.json")[0]))["rows"]
        Ts = sorted({x["T"] for x in rows}); n = 256
        for who, c in [("own", "C0"), ("peer", "C1"), ("world", "C2")]:
            for row, key in enumerate(["gain", "read"]):
                v = [[x[key] for x in rows if x["T"] == T and x["n"] == n and x["writer"] == who] for T in Ts]
                m, se = np.array([np.mean(a) for a in v]), np.array([np.std(a) / np.sqrt(len(a)) for a in v])
                axes[row, col].errorbar(np.array(Ts) + 50, m, se, color=c, marker="o", label={"own": "own examples", "peer": "a peer's examples", "world": "the world's examples"}[who])
        best = [np.mean([x["kl_none"] - x["kl_best"] for x in rows if x["T"] == T and x["n"] == n and x["writer"] == "own"]) for T in Ts]
        axes[0, col].plot(np.array(Ts) + 50, best, "k:", label="best this reader can learn")
        axes[0, col].set_title(f"{prior} prior: lesson learned from {n} examples"); axes[1, col].set_title("bits of the examples read correctly")
        axes[0, col].set_ylim(-1.0, 0.75); axes[1, col].set_xscale("symlog", linthresh=100); axes[1, col].set_xlabel("wake-sleep steps of reader and peer before the lesson")
    axes[0, 0].set_ylabel("held-out KL gain, nats"); axes[1, 0].set_ylabel("share of bits read correctly"); axes[0, 0].legend(fontsize="small")
    fig.tight_layout(); out = f"{FIGS}/practice-disentangle-own.png"; fig.savefig(out, dpi=110); print(out)


def conflict():
    """Learning from unlimited fresh data can make a good learner worse: bias of reading through an approximate recognizer,
    and a conflict between the learner's own prior and the data."""
    C = f"{ROOT}/conflict"; load = lambda pat: json.load(open(glob.glob(f"{C}/{pat}")[0]))
    LAB = {"practice": "practice (wake-sleep)", "watch": "watching only (recognizer frozen)", "em": "exact EM"}; COL = {"practice": "C0", "watch": "C1", "em": "C2"}
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    for lr, ls in [("0.02", "-"), ("0.005", "--")]:
        r = load(f"rw0.85_ro0.85_true_lr{lr}_2*/results.json"); scale = float(lr) / 0.02
        for m, runs in r["results"].items():
            t = np.array([p["step"] for p in runs[0]]) * scale; kl = np.array([[p["kl"] for p in run] for run in runs]).mean(0)
            axes[0].plot(t[1:], kl[1:], ls, color=COL[m], label=f"{LAB[m]}, lr {lr}")
    axes[0].set_xscale("log"); axes[0].set_xlabel("steps x (lr / 0.02)"); axes[0].set_ylabel("KL(world || learner), nats")
    axes[0].set_title("from the true decoder, prior matches the world"); axes[0].legend(fontsize="small")
    sig = [0.3, 0.5, 0.7, 0.9]
    for m in ["practice", "watch", "em"]:
        v = [np.array([run[-1]["align"] for run in load(f"*_sigma{s}_2*/results.json")["results"][m]]) for s in sig]
        axes[1].errorbar(sig, [x.mean() for x in v], [x.std() / np.sqrt(len(x)) for x in v], color=COL[m], marker="o", label=LAB[m])
    axes[1].set_xlabel("observation noise sigma"); axes[1].set_ylabel("concept purity (mean max |cos| to a world factor)"); axes[1].set_title("after 60,000 steps from the true decoder")
    for ax, rw in zip(axes[2:], ["0.5", "0.3"]):
        r = load(f"rw{rw}_ro0.85_true_lr0.02_2*/results.json")
        for m, runs in r["results"].items():
            t = np.array([p["step"] for p in runs[0]]); al = np.array([[p["align"] for p in run] for run in runs]).mean(0); kl = np.array([[p["kl"] for p in run] for run in runs]).mean(0)
            ax.plot(t[1:], al[1:], color=COL[m], label=f"{LAB[m]}: purity"); ax2 = ax.twinx() if m == "practice" else ax2; ax2.plot(t[1:], kl[1:], ":", color=COL[m])
        ax.set_xscale("log"); ax.set_xlabel("steps (fresh data)"); ax.set_ylabel("concept purity (solid)"); ax2.set_ylabel("KL, nats (dotted)")
        ax.set_title(f"own prior: correlation 0.85; world: {rw}")
    axes[2].legend(fontsize="small", loc="lower left")
    fig.tight_layout(); out = f"{FIGS}/practice-disentangle-conflict.png"; fig.savefig(out, dpi=110); print(out)


def prior():
    """Initialization assumption 1: a fixed, known prior against one learned jointly by a Hebbian/anti-Hebbian rule."""
    LAB = {"fixed_true": "prior fixed at the world's", "learned_from_true": "prior learned, starting from the world's",
           "learned_from_independent": "prior learned from independent latents (exact Boltzmann rule)", "slow_from_independent": "prior learned slowly, from independent",
           "dream_hebb_from_independent": "Hebbian in wake, anti-Hebbian on dreams", "top_layer": "a top layer, delta rule (two-layer wake-sleep)"}
    panels = [("chain", 256), ("chain", 0), ("deep", 256), ("deep", 0)]
    fig, axes = plt.subplots(2, len(panels), figsize=(5 * len(panels), 8), sharex=True)
    for col, (pr, N) in enumerate(panels):
        r = json.load(open(glob.glob(f"{ROOT}/prior/{pr}_N{N}_rotated_hebb_2*/results.json")[0]))
        for i, (c, runs) in enumerate(r["results"].items()):
            t = [p["step"] for p in runs[0]]
            for row, key in enumerate(["kl", "align"]):
                axes[row, col].plot(t, np.array([[p[key] for p in run] for run in runs]).mean(0), color=f"C{i}", label=LAB[c])
        axes[0, col].axhline(np.mean([s["binary_ae_refit_kl"] for s in r["static"]]), color="k", ls=":", label="binary autoencoder, prior fitted afterwards")
        axes[0, col].set_yscale("log"); axes[0, col].set_title(f"{pr} prior, " + ("fresh data" if N == 0 else f"{N} stored samples")); axes[1, col].set_xlabel("step")
    axes[0, 0].set_ylabel("KL(world || learner), nats"); axes[1, 0].set_ylabel("concept purity"); axes[0, 0].legend(fontsize="small")
    fig.tight_layout(); out = f"{FIGS}/practice-disentangle-prior.png"; fig.savefig(out, dpi=110); print(out)


def capacity():
    """Capacity gap: students of Ks features learning from teachers of Kt features (Kt = Kw is the world itself)."""
    r = json.load(open(sorted(glob.glob(f"{ROOT}/capacity/Kw8_decay0.5_2*/results.json"))[-1])); rows = r["rows"]
    Kss, Kts = sorted({x["Ks"] for x in rows}), sorted({x["Kt"] for x in rows}); LAB = {"wake_sleep": "wake-sleep student", "em": "exact-EM student"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for i, Ks in enumerate(Kss):
        for m, ls in [("wake_sleep", "-"), ("em", "--")]:
            v = np.array([[x["curve"][-1]["kl"] for x in rows if x["Ks"] == Ks and x["Kt"] == Kt and x["method"] == m] for Kt in Kts])
            axes[0].errorbar(Kts, v.mean(1), v.std(1) / np.sqrt(v.shape[1]), color=f"C{i}", ls=ls, marker="o", label=f"{LAB[m]}, {Ks} features")
            tk = np.array([np.mean([x["kl_teacher"] for x in rows if x["Ks"] == Ks and x["Kt"] == Kt and x["method"] == m]) for Kt in Kts])
            axes[1].plot(tk, v.mean(1), ls=ls, color=f"C{i}", marker="o")
        axes[0].axhline(np.mean([x["kl_best"] for x in rows if x["Ks"] == Ks]), color=f"C{i}", ls=":", lw=1)
    tk = [np.mean([x["kl_teacher"] for x in rows if x["Kt"] == Kt]) for Kt in Kts]
    axes[0].plot(Kts, tk, "k:", marker="x", label="teacher itself"); axes[0].set_yscale("log"); axes[0].set_xlabel("teacher features Kt (8 = the world)")
    axes[0].set_ylabel("student KL(world || student), nats"); axes[0].set_title("after 20,000 steps; dotted: best model of the student's size"); axes[0].legend(fontsize="small")
    axes[1].plot(tk, tk, "k:", lw=1); axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("teacher KL(world || teacher), nats"); axes[1].set_ylabel("student KL, nats")
    axes[1].set_title("student vs teacher loss (cf. distillation scaling laws)")
    Ks = Kss[len(Kss) // 2]
    for j, Kt in enumerate([Ks, max(Kts)]):
        for m, ls in [("wake_sleep", "-"), ("em", "--")]:
            cs = [x["curve"] for x in rows if x["Ks"] == Ks and x["Kt"] == Kt and x["method"] == m]; t = np.array([p["step"] for p in cs[0]]) * r["args"]["batch"]
            axes[2].plot(t[1:], np.array([[p["kl"] for p in c] for c in cs]).mean(0)[1:], ls=ls, color=f"C{j + 3}", label=f"{LAB[m]}, teacher {'= world' if Kt == max(Kts) else f'of {Kt} features'}")
    axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlabel("teacher samples seen"); axes[2].set_ylabel("student KL, nats")
    axes[2].set_title(f"learning curves, student of {Ks} features"); axes[2].legend(fontsize="small")
    fig.tight_layout(); out = f"{FIGS}/practice-disentangle-capacity.png"; fig.savefig(out, dpi=110); print(out)


if __name__ == "__main__": supervision(); own(); conflict(); prior(); capacity()
