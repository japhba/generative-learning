"""Does an own sample that reaches the deep layer teach it? Linear two-layer caricature.

First layer taken as learned (reading and writing of h exact), so everything is in h-space (N units).
Teacher: h = Gs z, z ~ N(0, 1_K), Gs with K orthonormal columns. Learner: writer G (N x K, random
orthonormal columns), reader W = G^T (aligned start). Reading z_hat = W h.
Reader: Hebb with Oja's decay, dW = eta (z_hat h^T - z_hat z_hat^T W), on every sample it is shown.
Writer: delta rule on teacher samples, dG = eta (h - G z_hat) z_hat^T; on own samples (read through the
string) the same rule, which is zero when the sample is read back as the latent that wrote it.
Arms: a share of the samples are own samples (or a twin's: another random writer) at a fixed total.
"""
import json, sys, numpy as np
from tqdm.auto import tqdm

N, K, ETA, T, EVERY, SEEDS = 256, 16, 0.01, 4000, 20, 8
ARMS = [("teacher only", "own", 0.0), ("own share 1/4", "own", 0.25), ("own share 1/2", "own", 0.5), ("own share 3/4", "own", 0.75), ("twin share 1/2", "twin", 0.5)]

def orth(rng, n, k): return np.linalg.qr(rng.standard_normal((n, k)))[0]

def measures(W, G, Gs, Gt):
    """reach_*: share of a sample's power that arrives at the deep layer, E|W h|^2 / E|h|^2 per unit row norm; cov_cos: cosine between the covariances of own and teacher samples."""
    Q = np.linalg.qr(G)[0]; w = np.sum(W ** 2)
    return dict(reach_teacher=float(np.sum((W @ Gs) ** 2) / w), reach_own=float(np.sum((W @ Q) ** 2) / w), reach_twin=float(np.sum((W @ Gt) ** 2) / w),
                writer_in_teacher=float(np.sum((Q.T @ Gs) ** 2) / K), cov_cos=float(np.sum((G.T @ Gs) ** 2) / (np.linalg.norm(G.T @ G) * np.sqrt(K))), readback=float(np.sum((W @ G - np.eye(K)) ** 2) / K))

def run(source, share, seed):
    rng = np.random.default_rng(seed); Gs, G, Gt = orth(rng, N, K), orth(rng, N, K), orth(rng, N, K); W = G.T.copy(); rows = []
    for t in range(T + 1):
        if t % EVERY == 0: rows.append(dict(t=t, **measures(W, G, Gs, Gt)))
        z = rng.standard_normal(K); own = rng.random() < share
        h = ((G if source == "own" else Gt) if own else Gs) @ z; zh = W @ h
        W += ETA * (np.outer(zh, h) - np.outer(zh, zh) @ W)
        if not own or source == "own": G += ETA * np.outer(h - G @ zh, zh)
    return rows

if __name__ == "__main__":
    out = [dict(name=name, source=source, share=share, seed=seed, curve=run(source, share, seed)) for name, source, share in tqdm(ARMS) for seed in tqdm(range(SEEDS), leave=False)]
    json.dump(dict(N=N, K=K, eta=ETA, runs=out), open("reports/null-space-linear.json", "w"))
    med = lambda name, key: np.median([[r[key] for r in o["curve"]] for o in out if o["name"] == name], axis=0)
    ts = np.array([r["t"] for r in out[0]["curve"]])
    for name, _, _ in ARMS:
        a, b = med(name, "reach_teacher"), med(name, "writer_in_teacher")
        first = lambda c: int(ts[np.argmax(c > 0.9)]) if (c > 0.9).any() else None
        print(f"{name:16s} reach own/teacher/twin at 0: {med(name, 'reach_own')[0]:.3f} / {a[0]:.3f} / {med(name, 'reach_twin')[0]:.3f}   reader > 0.9 at {first(a)}   writer > 0.9 at {first(b)}   end reader {a[-1]:.3f} writer {b[-1]:.3f}  reach own/twin end {med(name, 'reach_own')[-1]:.3f} / {med(name, 'reach_twin')[-1]:.3f}  cov cos start {med(name, 'cov_cos')[0]:.3f}  readback err end {med(name, 'readback')[-1]:.3f}")
    if "--scan" in sys.argv:  # size of the null space: samples until the writer's patterns are 0.9 inside the teacher's subspace
        scan = []
        for n in tqdm([64, 256, 1024, 4096]):
            N = n; scan.append(dict(N=n, **{name: int(np.median([next(r["t"] for r in run("own", share, seed) if r["writer_in_teacher"] > 0.9) for seed in range(4)])) for name, share in [("teacher only", 0.0), ("own share 1/2", 0.5)]}))
            print(scan[-1], "ln(N / K) / (2 eta) =", round(np.log(n / K) / (2 * ETA)))
        json.dump(scan, open("reports/null-space-linear-scan.json", "w")); N = 256
    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for key, label, style in [("reach_own", "own samples", "--"), ("reach_teacher", "teacher samples", "-"), ("reach_twin", "a twin's samples", "-.")]: ax[0].plot(ts, med("teacher only", key), style, label=label)
        ax[0].axhline(K / N, color="gray", ls=":", label="K / N"); ax[0].set_xscale("symlog", linthresh=20); ax[0].set_xlim(0, T); ax[0].set(xlabel="teacher samples seen", ylabel="share of a sample that reaches the deep layer", title="teacher samples only"); ax[0].legend()
        for name, _, _ in ARMS: ax[1].plot(ts, med(name, "writer_in_teacher"), "--" if "twin" in name else "-", label=name)
        ax[1].set_xscale("symlog", linthresh=20); ax[1].set_xlim(0, T); ax[1].set(xlabel="samples (teacher, own and twin counted alike)", ylabel="writer's patterns inside the teacher's subspace"); ax[1].legend()
        fig.tight_layout(); fig.savefig("reports/null-space-linear.png", dpi=150); fig.savefig("reports/null-space-linear.svg")
