"""Which layer do teacher examples change? Two-level word model, students that are deeply structured with partly the teacher's latents.
World: K causes; cause c shows the mid symbol tau_j(c) (of M) in each of S slots; symbol m writes the token sigma_l(m) (of V) at each of
the L positions of its patch (one dictionary for all slots). K valid strings of S * L tokens; every slot shows every symbol equally often.
Learner: generator = top table pi_j(m | code) = softmax(A_j[code]) and dictionary pi_l(x | m) = softmax(D_l[m]), uniform prior over codes;
recogniser bottom-up, q(m | patch) = softmax(b1 + sum_l R1_l[:, x_l]) and q(code | m_1..m_S) = softmax(b2 + sum_j R2_j[:, m_j]).
  wake   world batch, m^_j ~ q(. | patch_j), code^ ~ q(. | m^); delta rule on A_j[code^] towards m^_j and on D_l[m^_j] towards x_jl
  sleep  own batch written from the top (code~ -> m~ -> x~): R1 learns m~ from the patch, R2 learns code~ from m~ (the reader's only teacher)
Starts: each table is blank (None: small random logits) or sharp up to EPS with r right entries per unit (r = all: the teacher's table,
r = 0: unrelated); wrong entries are another unit's, so every table stays a permutation per slot / position. "relabel": the teacher under
other names for codes and symbols (within-layer control).
Arms: (top r, dictionary r, earlier own batches with the generator frozen, own samples, world batches during which the top table is held, reader).
Own samples "top": written from the code, label both levels; "syllable": label only the symbol of each patch (R2 never learns);
"switch": syllables only for the first SWITCH world batches (observation that reaches the dictionary only), then from the top.
Reader "exact": code^ and m^ are drawn from the generator's own posterior (all K codes enumerated), no recogniser and no own samples: reading
that runs through the generator from the top, as the benchmark for the bottom-up recogniser.
Scores (exact unless noted): surface = share of own patches that are valid syllables; deep = share of own samples that are valid strings
among those whose patches are all valid; kl = KL(teacher || student); read1 / read2 = share of the teacher's syllables / strings given the
matched symbol / code (read2 from NREAD sampled readings per cause); dA, dD = how far the rows of each table have moved from the start.
Usage: python ws_two_level_chimera.py [report] -> reports/ws-two-level-chimera.png, /var/tmp/jbauer/generative_practice_reading/ws_two_level_chimera.npz
("report" prints and plots from the saved file)"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ws_reading_gain import B, EPS, K, LR, SEEDS, V, draw, greedy_match, sm

M, S, L, STEPS, PRESLEEP, NREAD, SWITCH = K, 8, 8, 3000, 2000, 128, 600
T = np.unique(np.r_[0, np.round(np.geomspace(1, STEPS, 60))].astype(int)); N, SR, LR_ = np.arange(SEEDS), np.arange(S), np.arange(L)
WS = (PRESLEEP, "top", 0, "bottom-up"); EX = (0, "none", 0, "exact")
ARMS = {"blank": (None, None, *WS), "top 4 of 8": (4, L, *WS), "dictionary 4 of 8": (S, 4, *WS), "both 4 of 8": (4, 4, *WS), "relabel": ("relabel", "relabel", *WS), "top unrelated": (0, L, *WS), "top blank": (None, L, *WS),
        "top given, dictionary blank": (S, None, *WS), "top given and held, dictionary blank": (S, None, PRESLEEP, "top", 400, "bottom-up"), "top 4 of 8, dictionary blank": (4, None, *WS),
        "top 4 of 8, own samples label syllables only": (4, L, PRESLEEP, "syllable", 0, "bottom-up"), "top 4 of 8, no earlier practice": (4, L, 0, "top", 0, "bottom-up"), "blank, own samples label syllables only": (None, None, PRESLEEP, "syllable", 0, "bottom-up"),
        "blank, syllables only, then from the top": (None, None, PRESLEEP, "switch", 0, "bottom-up"),
        "exact: blank": (None, None, *EX), "exact: top given, dictionary blank": (S, None, *EX), "exact: top 4 of 8, dictionary blank": (4, None, *EX), "exact: top unrelated, dictionary blank": (0, None, *EX),
        "exact: both 4 of 8": (4, 4, *EX), "exact: top 4 of 8": (4, L, *EX), "exact: top unrelated": (0, L, *EX)}
KEYS = ["kl", "valid", "surface", "deep", "read1", "read2", "dA", "dD"]
OUT = Path("/var/tmp/jbauer/generative_practice_reading"); FIG = Path(__file__).resolve().parents[1] / "reports/ws-two-level-chimera.png"


def words(tau, sigma, c): m = tau[N[:, None, None], SR[None, :, None], c[:, None, :]]; return sigma[N[:, None, None, None], LR_[None, None, :, None], m[:, :, None, :]]   # c [n, b] -> tokens [n, S, L, b]
def read1(R1, b1, x): return b1[:, None, None, :] + R1[N[:, None, None, None], LR_[None, None, :, None], :, x].sum(2)                                                    # x [n, S, L, b] -> logits over symbols [n, S, b, M]
def read2(R2, b2, m): return b2[:, None, :] + R2[N[:, None, None], SR[None, :, None], :, m].sum(1)                                                                       # m [n, S, b] -> logits over codes [n, b, K]


def dream(rng, A, D, n):
    c = rng.integers(K, size=(SEEDS, n)); m = draw(rng, A[N[:, None, None], SR[None, :, None], c[:, None, :]]); return c, m, draw(rng, D[N[:, None, None, None], LR_[None, None, :, None], m[:, :, None, :]])


def sleep(rng, A, D, R1, b1, R2, b2, own):
    c, m, x = dream(rng, A, D, B); g1 = np.eye(M)[m] - sm(read1(R1, b1, x))
    R1 += LR * np.einsum("njbm,njlbv->nlmv", g1, np.eye(V)[x]) / (S * B); b1 += LR * g1.mean((1, 2))
    if own == "top": g2 = np.eye(K)[c] - sm(read2(R2, b2, m)); R2 += LR * np.einsum("nbk,njbm->njkm", g2, np.eye(M)[m]) / B; b2 += LR * g2.mean(1)


def wake(rng, tau, sigma, A, D, R1, b1, R2, b2, top, reader):
    x = words(tau, sigma, rng.integers(K, size=(SEEDS, B)))
    if reader == "exact":
        joint = np.log(sm(A))[:, :, None] + np.log(sm(D))[N[:, None, None, None], LR_[None, None, :, None], :, x].sum(2)[:, :, :, None, :]; top_ = joint.max(-1, keepdims=True)   # [n, S, b, code, symbol]: log p(m, patch | code)
        kh = draw(rng, (top_[..., 0] + np.log(np.exp(joint - top_).sum(-1))).sum(1)); mh = draw(rng, joint[N[:, None, None], SR[None, :, None], np.arange(B)[None, None, :], kh[:, None, :]])
    else: mh = draw(rng, read1(R1, b1, x)); kh = draw(rng, read2(R2, b2, mh))
    om, ok = np.eye(M)[mh], np.eye(K)[kh]; D += LR * (np.einsum("njbm,njlbv->nlmv", om, np.eye(V)[x]) - om.sum((1, 2))[:, None, :, None] * sm(D)) / (S * B)
    if top: A += LR * (np.einsum("nbk,njbm->njkm", ok, om) - ok.sum(1)[:, None, :, None] * sm(A)) / B


def chimera(rng, sig, r):                                                                                   # sig [n, J, units]: exactly r right entries per unit, the wrong units of a row swap entries in a cycle (a lone one stays right)
    tok, right = sig.copy(), np.argsort(rng.random((SEEDS, sig.shape[2], sig.shape[1])), -1).transpose(0, 2, 1) < r
    for n in range(SEEDS):
        for j in range(sig.shape[1]): w = rng.permutation(np.flatnonzero(~right[n, j])); tok[n, j, w] = sig[n, j, np.roll(w, 1)]
    return tok


def sharp(tok): P = np.full(tok.shape + (V,), EPS / V); np.put_along_axis(P, tok[..., None], 1 - EPS + EPS / V, 3); return np.log(P)


def start(rng, tau, sigma, top, bottom):                                                                    # logits A [n, S, K, M], D [n, L, M, V] and the sharp tables behind them (None where blank)
    if top == "relabel":
        p, u = np.argsort(rng.random((SEEDS, 1, K)), -1), np.argsort(rng.random((SEEDS, 1, M)), -1)       # code k is cause p(k), symbol m is syllable u(m)
        ta, td = np.take_along_axis(np.argsort(u, -1), np.take_along_axis(tau, np.broadcast_to(p, tau.shape), 2), 2), np.take_along_axis(sigma, np.broadcast_to(u, sigma.shape), 2)
    else: ta, td = None if top is None else chimera(rng, tau, top), None if bottom is None else chimera(rng, sigma, bottom)
    return 0.1 * rng.standard_normal((SEEDS, S, K, M)) if ta is None else sharp(ta), 0.1 * rng.standard_normal((SEEDS, L, M, V)) if td is None else sharp(td), ta, td


def tv(a, b): u, inv = np.unique(np.concatenate([a, b]), axis=0, return_inverse=True); inv = inv.ravel(); return 0.5 * np.abs(np.bincount(inv[:K], minlength=len(u)) - np.bincount(inv[K:], minlength=len(u))).sum() / K


def surface(tau, sigma, ta, td):                                                                            # total variation teacher vs sharp student by window: single tokens (max), token pairs in a patch, whole patch, two patches, whole string (means)
    c = np.broadcast_to(np.arange(K), (SEEDS, K)); xt, xs = words(tau, sigma, c).transpose(0, 3, 1, 2), words(ta, td, c).transpose(0, 3, 1, 2); rows = []   # [n, string, S, L]
    for n in range(SEEDS):
        a, b = xt[n], xs[n]
        rows.append([max(tv(a[:, j, [l]], b[:, j, [l]]) for j in range(S) for l in range(L)), np.mean([tv(a[:, j, list(p)], b[:, j, list(p)]) for j in range(S) for p in combinations(range(L), 2)]),
                     np.mean([tv(a[:, j], b[:, j]) for j in range(S)]), np.mean([tv(a[:, list(p)].reshape(K, -1), b[:, list(p)].reshape(K, -1)) for p in combinations(range(S), 2)]), tv(a.reshape(K, -1), b.reshape(K, -1))])
    return np.mean(rows, 0)


def matched(J): return np.take_along_axis(J, greedy_match(J)[..., None], 2).sum((1, 2)) / J.sum((1, 2))   # J [n, truth, unit] -> share of the mass on the one-to-one matched unit


def score(rng, tau, sigma, A, D, R1, b1, R2, b2, A0, D0):
    PA, PD = sm(A), sm(D); E = np.exp(np.log(PD)[N[:, None, None], LR_[None, :, None], :, sigma].sum(1)).transpose(0, 2, 1)   # E [n, symbol, syllable]: the symbol writes that teacher syllable
    q = np.einsum("njkm,nmp->njkp", PA, E); Lw = np.log(q[N[:, None, None], SR[None, :, None], :, tau]).sum(1); pw = np.exp(Lw).mean(2)   # q [n, slot, code, syllable]; Lw [n, cause, code]; pw [n, cause]
    c = np.broadcast_to(np.repeat(np.arange(K), NREAD), (SEEDS, K * NREAD)); kh = draw(rng, read2(R2, b2, draw(rng, read1(R1, b1, words(tau, sigma, c))))); J = np.zeros((SEEDS, K, K)); np.add.at(J, (N[:, None], c, kh), 1)
    return {"kl": -np.log(K * pw).mean(1), "valid": pw.sum(1), "surface": q.mean(2).sum(2).mean(1), "deep": pw.sum(1) / q.sum(3).prod(1).mean(1),
            "read1": matched(sm(b1[:, None] + R1[N[:, None, None], LR_[None, :, None], :, sigma].sum(1))), "read2": matched(J), "dA": 0.5 * np.abs(PA - A0).sum(-1).mean((1, 2)), "dD": 0.5 * np.abs(PD - D0).sum(-1).mean((1, 2))}


def run(name, seed=0):
    top, bottom, presleep, own, hold, reader = ARMS[name]; rng = np.random.default_rng(seed); tau, sigma = np.argsort(rng.random((SEEDS, S, K)), -1), np.argsort(rng.random((SEEDS, L, M)), -1)
    A, D, ta, td = start(rng, tau, sigma, top, bottom); A0, D0 = sm(A), sm(D); R1, b1, R2, b2, log = np.zeros((SEEDS, L, M, V)), np.zeros((SEEDS, M)), np.zeros((SEEDS, S, K, M)), np.zeros((SEEDS, K)), []
    surf = np.full(5, np.nan) if ta is None or td is None else surface(tau, sigma, ta, td)
    for _ in range(presleep): sleep(rng, A, D, R1, b1, R2, b2, "syllable" if own == "switch" else own)
    for t in range(STEPS + 1):
        if t in T: log.append(score(rng, tau, sigma, A, D, R1, b1, R2, b2, A0, D0))
        wake(rng, tau, sigma, A, D, R1, b1, R2, b2, t >= hold, reader)
        if own != "none": sleep(rng, A, D, R1, b1, R2, b2, ("syllable" if t < SWITCH else "top") if own == "switch" else own)
    return {k: np.array([row[k] for row in log]) for k in KEYS} | {"surf": surf}                            # [time, seed]; surf [5]


def onset(y, level=0.5): hit = y >= level; return np.where(hit.any(0), T[hit.argmax(0)], np.nan)
def nmed(v): return np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan


def report():
    z = np.load(OUT / "ws_two_level_chimera.npz"); res = {name: {k: z[f"{name}/{k}"] for k in KEYS + ["surf"]} for name in ARMS}; at = lambda y, t: np.median(y[np.searchsorted(T, t)])
    print("start, sharp tables: total variation teacher vs own | single tokens (max) | token pairs in a patch | whole patch | two patches | whole string")
    for name, r in res.items(): print(f"  {name:48s} " + "  ".join(f"{v:.3f}" for v in r["surf"]))
    print("before any world example: syllables read | strings read | patches valid | combinations valid   ||   world batches until >= 0.5 (seeds): patches, combinations | KL at 0, 10, 100, 1000, end | seeds with KL > 1 at 1000, at the end | share of each table's final change done at 30 and 100 batches: top, dictionary")
    for name, r in res.items():
        on = [onset(r[k]) for k in ("surface", "deep")]; done = lambda k, t: at(r[k], t) / np.median(r[k][-1])
        print(f"  {name:48s} {at(r['read1'], 0):.2f} | {at(r['read2'], 0):.2f} | {at(r['surface'], 0):.2f} | {at(r['deep'], 0):.1e}   ||   " + "  ".join(f"{nmed(o):6.0f} ({np.isfinite(o).sum():2d}/{SEEDS})" for o in on)
              + " | " + "  ".join(f"{at(r['kl'], t):.2f}" for t in (0, 10, 100, 1000, STEPS)) + f" | {(r['kl'][np.searchsorted(T, 1000)] > 1).sum():2d}  {(r['kl'][-1] > 1).sum():2d} | {done('dA', 30):.2f} {done('dA', 100):.2f}   {done('dD', 30):.2f} {done('dD', 100):.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True); med = lambda y: np.median(y, 1)
    for c, name, lab in [("C0", "blank", "blank start"), ("C1", "both 4 of 8", "both tables 4 of 8 right")]:
        ax[0].plot(T + 1, med(res[name]["surface"]), color=c, label=f"{lab}: patches"); ax[0].plot(T + 1, med(res[name]["deep"]), color=c, ls="--", label=f"{lab}: combinations of patches")
    ax[0].set(xscale="log", xlabel="world batches + 1", ylabel="share of own samples that are valid", title="which level is right when")
    for c, ls, name, lab in [("C1", "-", "top 4 of 8", "4 of 8 right"), ("C0", "-", "top blank", "diffuse"), ("C2", "-", "top unrelated", "sharp, unrelated"), ("C1", ":", "top 4 of 8, own samples label syllables only", "4 of 8 right, own samples label patches only")]:
        ax[1].plot(T + 1, med(res[name]["deep"]), color=c, ls=ls, label=f"top table {lab}")
    ax[1].set(xscale="log", xlabel="world batches + 1", ylabel="share of valid combinations", title="the top table, dictionary already right")
    for ls, name, lab in [(":", "blank, own samples label syllables only", "own samples label patches only"), ("--", "blank, syllables only, then from the top", f"the same, written from the top after {SWITCH} batches"), ("-", "blank", "own samples written from the top throughout")]:
        ax[2].plot(T + 1, med(res[name]["kl"]), color="C0", ls=ls, label=lab)
    ax[2].axhline((S - 1) * np.log(K), color="gray", lw=plt.rcParams["lines.linewidth"] / 2, label="right patches in random combination"); ax[2].set(xscale="log", yscale="log", xlabel="world batches + 1", ylabel="KL(teacher || student)", title="blank start: watching, then generating from the top")
    for a in ax: a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(len(ARMS)) as pool: res = dict(zip(ARMS, tqdm(pool.imap(run, ARMS), total=len(ARMS))))
    np.savez(OUT / "ws_two_level_chimera.npz", t=T, **{f"{name}/{k}": v for name, r in res.items() for k, v in r.items()}); report()


if __name__ == "__main__": report() if sys.argv[1:] == ["report"] else main()
