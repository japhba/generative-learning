"""Learning curves per layer for the deep wake-sleep RHM toy."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

LABEL = {"practice": "practice (wake + sleep)", "watch": "watching only (wake, own recognition)", "watch_tied": "watching only, symmetric weights\n(recognition = bottom-up inverse of generator)",
         "watch_exact": "watching only, exact inference\n(Monte Carlo EM, ideal reference)", "watch_more": "watching, a fresh lecture\nin place of each dream"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out", required=True); args = ap.parse_args()
    r = json.load(open(f"{args.run}/results.json")); a = r["args"]; methods = list(r["results"]); L = a["L"]
    fig, axes = plt.subplots(3, len(methods), figsize=(6 * len(methods), 12), sharey="row", squeeze=False)
    for col, m in enumerate(methods):
        curves = r["results"][m]; t = np.array([c["t"] for c in curves[0]])
        for row, key in enumerate(["nmi", "valid"]):
            v = np.array([[c[key] for c in curve] for curve in curves])  # seeds x time x height
            for h in range(L):
                for s in range(len(v)): axes[row, col].plot(t, v[s, :, h], color=plt.cm.viridis(h / max(L - 1, 1)), alpha=0.25)
                axes[row, col].plot(t, v[:, :, h].mean(0), color=plt.cm.viridis(h / max(L - 1, 1)), label=f"layer {h + 1}" + (" (root)" if h == L - 1 else ""))
        kl = np.array([[c["kl"] for c in curve] for curve in curves])
        for s in range(len(kl)): axes[2, col].plot(t, kl[s], color="k", alpha=0.25)
        axes[2, col].plot(t, kl.mean(0), color="k"); axes[2, col].set_yscale("log")
        axes[0, col].set_title(LABEL[m]); axes[2, col].set_xlabel("update")
    axes[0, 0].set_ylabel("recognition: share of the true symbol's\ninformation carried by the learner's code")
    axes[1, 0].set_ylabel("generation: share of own samples\nthat parse correctly up to this layer")
    axes[2, 0].set_ylabel("KL(true || learner), nats"); axes[0, 0].legend()
    fig.suptitle(f"RHM v={a['v']}, classes={a['nc']}, m={a['m']}, s={a['s']}, L={L}; rule={a['rule']}, lr={a['lr']}, "
                 + ("fresh lectures" if not a["N"] else f"N={a['N']} stored lectures") + f"; thin lines: {len(r['results'][methods[0]])} seeds")
    fig.tight_layout(); fig.savefig(args.out, dpi=110); print(args.out)
    if "drive" not in r["results"][methods[0]][0][0]: return
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5), sharey=True, squeeze=False); axes = axes[0]
    T = a["updates"]; never = 2 * T
    for ax, m in zip(axes, methods):
        curves = r["results"][m]; t = np.array([c["t"] for c in curves[0]])
        for s, curve in enumerate(curves):
            nm = np.array([c["nmi"] for c in curve]); dr = np.array([c["drive"] for c in curve])
            on = [t[np.argmax(nm[:, h] >= 0.1)] if (nm[:, h] >= 0.1).any() else never for h in range(L)]
            dn = [t[np.argmax(dr[:, h] >= 0.1)] if (dr[:, h] >= 0.1).any() else never for h in range(L)]
            x = np.arange(1, L + 1) + 0.06 * (s - len(curves) / 2)
            ax.plot(x, np.maximum(on, 500), "o-", color="C0", alpha=0.6, label="layer's recognition carries 10% of its symbol" if s == 0 else None)
            ax.plot(x, np.maximum(dn, 500), "s:", color="C1", alpha=0.6, label="sibling information beneath it exceeds 0.1 nats" if s == 0 else None)
        ax.axhline(never, color="gray", lw=0.5); ax.text(1, never, "never", va="bottom", color="gray")
        ax.set_yscale("log"); ax.set_xticks(range(1, L + 1)); ax.set_xlabel("layer (1 = just above the data)"); ax.set_title(LABEL[m])
    axes[0].set_ylabel("onset (update)"); axes[0].legend(fontsize="small", loc="lower right")
    fig.tight_layout(); out = args.out.replace(".png", "_onsets.png"); fig.savefig(out, dpi=110); print(out)


if __name__ == "__main__": main()
