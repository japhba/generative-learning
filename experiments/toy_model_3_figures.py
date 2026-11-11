"""Figures and quoted numbers for toy model 3 (the Oracle on a chain world), from the runs of toy_model_3_oracle.py.
Usage: MPLBACKEND=Agg python toy_model_3_figures.py -> reports/toy3-*.png and /var/tmp/jbauer/generative_practice_chain/numbers.json"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

from toy_model_3_oracle import L, M, N_GRID, OUT, Oracle, S, chains, grid, setup

FIG = Path(__file__).resolve().parents[1] / "reports"
n = 2 ** L + 1
TEACH, OWN, THIRD, REF, INK = "#eb6834", "#2a78d6", "#1baf7a", "#8a8984", "#0b0b0b"  # teacher side · own generation · third arm · reference lines · text
LEVEL = ["#86b6ef", "#3987e5", "#0d366b"]  # spans of 2, 4, 8 links: one hue, light to dark
SPAN = ["spans of 2 links", "spans of 4 links", "spans of 8 links"]
nums = {}


def runs(name, **match): return [r for r in json.loads((OUT / f"{name}.json").read_text())["runs"] if all(r["args"].get(k) == v for k, v in match.items())]
def arr(rs, key): return np.array([[row.get(key, np.nan) for row in r["out"]] for r in rs], float)
def xs(rs, key): return np.array([row[key] for row in rs[0]["out"]], float)
def band(ax, x, y, color, label=None, ls="-", mid=np.mean): ax.plot(x, mid(y, 0), color=color, ls=ls, label=label); ax.fill_between(x, y.min(0), y.max(0), color=color, alpha=0.15, lw=0)
def below(ax, ncol=1, **kw): return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, frameon=False, **kw)
def cross(x, y, level=0.5):
    """First x at which y passes level, interpolated in log x; nan if it never does."""
    i = np.flatnonzero(y >= level)
    if len(i) == 0: return np.nan
    i = i[0]
    if i == 0: return float(x[0])
    t = (level - y[i - 1]) / (y[i] - y[i - 1]); return float(np.exp(np.log(x[i - 1]) + t * (np.log(x[i]) - np.log(x[i - 1]))))
def med(v): return float(np.nanmedian(v))
def save(fig, name): fig.savefig(FIG / name, dpi=150); plt.close(fig); print(FIG / name)


# ------------------------------------------------------------------ scheme: the chain, the checker, the tree decoder
def scheme():
    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True); ax.set_xlim(-4.6, 9.7); ax.set_ylim(-1.45, 4.75); ax.set_aspect("equal"); ax.axis("off")
    for t in range(1, 10): ax.add_patch(plt.Circle((t, 0), 0.29, fc="white", ec=INK, zorder=3)); ax.text(t, 0, f"$x_{t}$", ha="center", va="center", zorder=4)
    for t in range(1, 9): ax.plot([t + 0.29, t + 0.71], [0, 0], color=INK, zorder=2); ax.plot([t + 0.12, t + 0.12, t + 0.88, t + 0.88], [-0.45, -0.62, -0.62, -0.45], color=TEACH)
    ax.text(0.6, -0.55, "the recogniser checks:", ha="right", va="center"); ax.text(5, -1.15, "one lookup of the pair table per link", ha="center", va="center")
    for l, k in enumerate([1, 2, 4]):
        for a in range(1, 9, 2 * k): ax.add_patch(Arc((a + k, 0.34), 2 * k, 2.1 * (l + 1), theta1=0, theta2=180, color=LEVEL[l])); ax.plot([a + k, a + k], [0.34 + 1.05 * (l + 1), 0.34], color=LEVEL[l], ls=":")
    for l, text in enumerate(["level 1 draws $x_2, x_4, x_6, x_8$", "level 2 draws $x_3$ and $x_7$", "level 3 draws $x_5$ from $(x_1, x_9)$"]): ax.text(0.6, 0.34 + 1.05 * (l + 1) - 0.25, text, ha="right", va="center", bbox=dict(fc="white", ec=LEVEL[l], boxstyle="round"))
    ax.text(0.6, 0.34 + 1.05 * 4 - 0.25, "the generator writes, top-down:", ha="right", va="center"); ax.text(5, 4.5, "each node holds the two ends of a span and draws its midpoint", ha="center", va="center")
    save(fig, "toy3-scheme.png")


# ------------------------------------------------------------------ watching only: recognition leads generation
def theory_watch(N, seeds=range(4)):
    """Expected share of population spans seen at least once in N chains, per level, and of population links (exact route counts of the worlds)."""
    out = np.zeros((L + 1, len(N))); out[0] = 1 - (1 - 1 / (S * M)) ** (N * (n - 1))
    for seed in seeds:
        _, _, T, _ = setup(seed, test=1); P = T.astype(np.float64)
        for l in range(1, L + 1):
            P = P @ P; pi = (P / P.sum())[P > 0]; out[l] += (pi[None] * (1 - (1 - pi[None]) ** (N[:, None] * (n - 2 ** l)))).sum(1) / len(seeds)
            nums[f"multi_route{l}"] = nums.get(f"multi_route{l}", 0) + float((P / P.sum())[P > 1.5].sum()) / len(seeds)  # share of population spans whose ends are joined by more than one route
    return out


def headstart():
    rs = runs("watch"); N = xs(rs, "N"); fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 5.4), constrained_layout=True)
    th = theory_watch(N); band(ax, N, arr(rs, "read0"), TEACH, "links the recogniser accepts"); ax.plot(N, th[0], color=INK, ls=":")
    for l in range(1, L + 1): band(ax, N, arr(rs, f"write{l}"), LEVEL[l - 1], f"{SPAN[l - 1]} the generator writes"); ax.plot(N, th[l], color=INK, ls=":", label="counting argument" if l == L else None)
    ax.set_xscale("log"); ax.set_xlabel("teacher chains watched"); ax.set_ylabel("share of the population's links or spans"); ax.set_title("Watching only: the recogniser leads"); below(ax)
    half = {l: np.array([cross(N, y) for y in arr(rs, f"write{l}" if l else "read0")]) for l in range(L + 1)}
    nums.update({"half_read0": med(half[0]), **{f"half_watch{l}": med(half[l]) for l in range(1, L + 1)}, **{f"lead{l}": med(half[l] / half[0]) for l in range(1, L + 1)},
                 **{f"lead_theory{l}": M ** (2 ** l - 1) * (n - 1) / (n - 2 ** l) for l in range(1, L + 1)}, "half_read0_theory": S * M * np.log(2) / (n - 1)})
    pts = [(M, L, S, half)]
    for Sx, mx, Lx in [(4096, 2, 3), (1024, 3, 2), (1024, 4, 2), (256, 2, 2)]:
        r2 = runs("watch_scale", S=Sx, m=mx, L=Lx); N2 = xs(r2, "N"); pts.append((mx, Lx, Sx, {l: np.array([cross(N2, y) for y in arr(r2, f"write{l}" if l else "read0")]) for l in range(Lx + 1)}))
    for i, (mx, Lx, Sx, h) in enumerate(pts):
        nx = 2 ** Lx + 1; pred = [mx ** (2 ** l - 1) * (nx - 1) / (nx - 2 ** l) for l in range(1, Lx + 1)]; meas = [med(h[l] / h[0]) for l in range(1, Lx + 1)]
        bx.plot(pred, meas, marker="osD^v"[i], ls="", color=[OWN, TEACH, THIRD, "#eda100", "#e87ba4"][i], label=f"{Sx:,} states, {mx} successors, chains of {nx}")
        nums[f"scale_{Sx}_{mx}_{Lx}"] = " · ".join(f"{a:,.0f} ({b:,.0f})" for a, b in zip(meas, pred))
    lim = [1.5, 2000]; bx.plot(lim, lim, color=REF, ls="--", label="measured = predicted"); bx.set_xscale("log"); bx.set_yscale("log"); bx.set_xlabel(r"predicted delay, $m^{2k-1}(n-1)/(n-2k)$"); bx.set_ylabel("measured delay of the generator")
    bx.set_title("The delay grows with the span"); below(bx); save(fig, "toy3-headstart.png")


# ------------------------------------------------------------------ practice closes the gap
def practice():
    w = runs("watch"); Nw = xs(w, "N"); N = np.array(N_GRID, float); pr = [runs("practice", N=int(v)) for v in N]; po = [runs("pairs_only", N=int(v)) for v in N]
    fin = lambda rs, key: np.array([[r["out"][-1][key] for r in g] for g in rs]).T; first = lambda rs, key: np.array([[r["out"][0][key] for r in g] for g in rs]).T  # (world, N)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 5.4), constrained_layout=True)
    band(ax, Nw, arr(w, f"write{L}"), TEACH, "watching only"); band(ax, N, fin(pr, f"write{L}"), OWN, "watching, then practice"); band(ax, N, fin(po, f"write{L}"), THIRD, "teacher gives the pair table only, then practice", ls="--")
    ax.plot(N, first(pr, f"reach{L}").mean(0), color=REF, ls=":", label="joined by accepted links (ceiling of practice)")
    ax.set_xscale("log"); ax.set_xlabel("teacher chains"); ax.set_ylabel(f"share of {SPAN[L - 1]} written in one pass"); ax.set_title("Practice needs far fewer teacher chains"); below(ax)
    hw = np.array([cross(Nw, y) for y in arr(w, f"write{L}")]); hp = np.array([cross(N, y) for y in fin(pr, f"write{L}")]); ho = np.array([cross(N, y) for y in fin(po, f"write{L}")])
    nums.update({"half_practice": med(hp), "half_pairs_only": med(ho), "saving": med(hw) / med(hp), "half_practice_theory": S * M * np.log(1 / (1 - 0.5 ** (1 / 2 ** L))) / (n - 1)})
    for l in range(1, L + 1):
        gain = fin(pr, f"write{l}") - first(pr, f"write{l}"); band(bx, N, gain, LEVEL[l - 1], SPAN[l - 1]); nums[f"gain_peak{l}"] = float(gain.mean(0).max()); nums[f"gain_peak_N{l}"] = int(N[gain.mean(0).argmax()])
        nums[f"gain_window{l}"] = " and ".join(f"{int(v):,}" for v in N[np.flatnonzero(gain.mean(0) > 0.5)][[0, -1]]) if (gain.mean(0) > 0.5).any() else "none"
    bx.set_xscale("log"); bx.set_xlabel("teacher chains watched before practice"); bx.set_ylabel("share of spans gained by practice"); bx.set_title("Practice pays in a window"); below(bx)
    for e in [6, 9, 10, 12, 15, 18]:
        i = int(np.flatnonzero(N == 2 ** e)[0]); nums.update({f"n{e}_read0": float(first(pr, "read0")[:, i].mean()), f"n{e}_reach": float(first(pr, f"reach{L}")[:, i].mean()), **{f"n{e}_w{l}": float(first(pr, f"write{l}")[:, i].mean()) for l in range(1, L + 1)},
                     **{f"n{e}_p{l}": float(fin(pr, f"write{l}")[:, i].mean()) for l in range(1, L + 1)}, **{f"n{e}_o{l}": float(fin(po, f"write{l}")[:, i].mean()) for l in range(1, L + 1)}})
    allr = [r for g in pr + po for r in g] + w; nums["read_write_gap"] = float(max(np.nanmax(np.abs(arr([r], f"read{l}") - arr([r], f"write{l}"))) for r in allr for l in range(1, L + 1)))
    nums["own_min"] = float(min(np.nanmin(arr([r], f"own{l}")) for r in allr for l in range(1, L + 1))); nums["pairs_only_gap"] = float(np.abs(fin(pr, f"write{L}") - fin(po, f"write{L}")).mean(0).max())
    save(fig, "toy3-practice.png")


# ------------------------------------------------------------------ the staircase: reading and writing, level by level
def staircase():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), constrained_layout=True, sharey=True)
    for ax, e in zip(axes, [12, 9]):
        rs = runs("pairs_only", N=2 ** e); Mx = xs(rs, "M"); Mx[0] = Mx[1] / 2; q = arr(rs, "read0")[:, 0].mean(); nums[f"stair_q{e}"] = float(q)
        ax.axhline(q, color=TEACH, label="links the recogniser accepts")
        for l in range(1, L + 1):
            band(ax, Mx, arr(rs, f"write{l}"), LEVEL[l - 1], f"{SPAN[l - 1]}: written in one pass"); ax.plot(Mx[2::5], arr(rs, f"read{l}").mean(0)[2::5], color=LEVEL[l - 1], marker="o", mfc="none", ls="", label="the same spans held as one unit (read in one lookup)" if l == L else None)
            ax.axhline(arr(rs, f"reach{l}")[:, 0].mean(), color=LEVEL[l - 1], ls=":")
            nums[f"stair{e}_{l}"] = med([cross(Mx, y, 0.5 * y[-1]) for y in arr(rs, f"write{l}")]); nums[f"stair{e}_end{l}"] = float(arr(rs, f"write{l}")[:, -1].mean()); nums[f"stair{e}_reach{l}"] = float(arr(rs, f"reach{l}")[:, 0].mean())
        ax.set_xscale("log"); ax.set_xlabel("practice episodes (no teacher data)"); ax.set_title(f"After {2 ** e:,} teacher chains (pair table {'complete' if q > 0.9995 else f'{100 * q:.0f}% complete'})")
    axes[0].plot([], [], color=REF, ls=":", label="dotted: joined by accepted links (ceiling)"); axes[0].set_ylabel("share of the population's links or spans"); below(axes[0])
    save(fig, "toy3-staircase.png")


# ------------------------------------------------------------------ how much to watch, how much to practise
def tradeoff():
    N = np.array(N_GRID, float); pr = [runs("practice", N=int(v)) for v in N]; Mx = xs(pr[0], "M"); Z = np.array([arr(g, f"write{L}").mean(0) for g in pr]).T  # (M, N)
    fig = plt.figure(figsize=(10, 9.6), constrained_layout=True); gs = fig.add_gridspec(2, 2); ax, bx, cx = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ye = np.r_[Mx[1] / 2.6, Mx[1] / 1.3, np.sqrt(Mx[1:-1] * Mx[2:]), Mx[-1] * 1.15]; xe = 2.0 ** np.arange(5.5, 21)
    im = ax.pcolormesh(xe, ye, Z, cmap="Blues", vmin=0, vmax=1); ax.contour(N, np.r_[Mx[1] / 1.8, Mx[1:]], Z, levels=[0.5], colors=[TEACH]); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("teacher chains watched"); ax.set_ylabel("practice episodes afterwards\n(bottom row: none)"); ax.set_title(f"Share of {SPAN[L - 1]} written in one pass (line: one half)"); fig.colorbar(im, ax=ax)
    for order, color, label in [("watch_first", OWN, "watch, then practise"), ("alternate", THIRD, "alternate in 64 blocks"), ("practise_first", TEACH, "practise, then watch")]:
        rs = runs("schedule", order=order); band(bx, xs(rs, "examples")[1:], arr(rs, f"write{L}")[:, 1:], color, label); nums[f"sched_{order}"] = float(arr(rs, f"write{L}")[:, -1].mean())
    nums["sched_N"], nums["sched_M"] = int(runs("schedule")[0]["out"][-1]["N"]), int(runs("schedule")[0]["out"][-1]["M"])
    bx.set_xscale("log"); bx.set_xlabel("examples so far (teacher chains + practice episodes)"); bx.set_ylabel(f"share of {SPAN[L - 1]} written"); bx.set_title("The same examples in three orders"); below(bx)
    sp = runs("split"); budgets = sorted({r["args"]["tokens"] for r in sp}); shares = sorted({r["args"]["share"] for r in sp})
    for i, t in enumerate(budgets):
        y = np.array([[r["out"][f"write{L}"] for r in sp if r["args"]["tokens"] == t and r["args"]["share"] == x] for x in shares]).T; band(cx, shares, y, plt.cm.Greys(0.35 + 0.2 * i), f"$2^{{{int(np.log2(t))}}}$ tokens in total")
        e = int(np.log2(t)); nums[f"split{e}_all"] = float(y[:, -1].mean()); nums[f"split{e}_half"] = float(y[:, shares.index(0.5)].mean()); nums[f"split{e}_best"] = f"{100 * shares[int(y.mean(0).argmax())]:.0f}%"; nums[f"split{e}_wins"] = int((y[:, -1] >= y[:, :-1].max(1)).sum())
    cx.set_xlabel("share of the tokens spent on fresh teacher chains"); cx.set_ylabel(f"share of {SPAN[L - 1]} written"); cx.set_title("Fresh teacher chains or practice"); below(cx, 2)
    save(fig, "toy3-tradeoff.png")


# ------------------------------------------------------------------ supervision at every level, and a teacher who slips
def supervision():
    N = np.array([2 ** e for e in range(8, 18)], float); arms = [(dict(theta=1), TEACH, "-", "accepts every pair it sees; teacher fills every level"), (dict(theta=2, levels=None), OWN, "-", "accepts a pair seen twice; teacher fills every level"),
                                                               (dict(theta=2, levels=False), THIRD, "--", "accepts a pair seen twice; levels left to practice")]
    get = lambda kw, row, key: np.array([[r["out"][row][key] for r in runs("noisy", N=int(v)) if r["args"]["theta"] == kw["theta"] and r["args"].get("levels") == kw.get("levels")] for v in N]).T
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 5.4), constrained_layout=True)
    for kw, color, ls, label in arms: band(ax, N, get(kw, -1, f"own{L}"), color, label, ls); band(bx, N, get(kw, -1, f"write{L}"), color, None, ls)
    band(ax, N, get(arms[1][0], 0, f"own{L}"), REF, "the second learner before practice", ":")
    for a_, t in [(ax, "Own chains that are valid"), (bx, f"Population {SPAN[L - 1]} written")]: a_.set_xscale("log"); a_.set_xlabel("teacher chains (3% of their states are slips)"); a_.set_title(t); a_.set_ylabel("share, after practice")
    fig.legend(loc="outside lower center", frameon=False); i = int(np.flatnonzero(N == 2 ** 13)[0])
    for tag, (kw, *_) in zip(["trust", "forced", "left"], arms):
        nums.update({f"noisy_{tag}_own": float(get(kw, -1, f"own{L}")[:, i].mean()), f"noisy_{tag}_write": float(get(kw, -1, f"write{L}")[:, i].mean()), f"noisy_{tag}_wrong": float(get(kw, 0, "wrong0")[:, i].mean()),
                     f"noisy_{tag}_own_lo": float(get(kw, -1, f"own{L}").mean(0).min()), f"noisy_{tag}_own_hi": float(get(kw, -1, f"own{L}").mean(0).max()), f"noisy_{tag}_half": med([cross(N, y) for y in get(kw, -1, f"write{L}")])})
    nums["noisy_forced_own0"] = float(get(arms[1][0], 0, f"own{L}")[:, i].mean()); nums["noisy_left_wins"] = int((get(arms[2][0], -1, f"own{L}")[:, i] > get(arms[1][0], -1, f"own{L}")[:, i]).sum())
    save(fig, "toy3-supervision.png")


# ------------------------------------------------------------------ a generator with wrong entries: whose examples to judge
def judge():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 5.4), constrained_layout=True)
    arms = [("teacher", TEACH, "-", "more teacher chains, taken as shown"), ("teacher_judged", TEACH, "--", "more teacher chains, judged"), ("own", OWN, "-", "own samples, judged"), ("own_drop", OWN, "--", "own samples, judged, wrong entries dropped")]
    for mode, color, ls, label in arms:
        rs = runs("judge", mode=mode); T = xs(rs, "tokens")[1:]; band(ax, T, arr(rs, f"own{L}")[:, 1:], color, label, ls, mid=np.median); band(bx, T, arr(rs, "write2")[:, 1:], color, None, ls, mid=np.median)
        for key in [f"own{L}", "write2", f"write{L}", f"wrong{L}"]: nums[f"judge_{mode}_{key}"] = float(np.median(arr(rs, key)[:, -1])); nums[f"judge_start_{key}"] = float(np.median(arr(rs, key)[:, 0]))
    for a_, t in [(ax, f"Own chains of {2 ** L} links that are valid"), (bx, "Population spans of 4 links written")]: a_.set_xscale("log"); a_.set_xlabel("further tokens (states shown or written)"); a_.set_ylabel("share"); a_.set_title(t)
    fig.legend(loc="outside lower center", frameon=False, ncol=2); nums["judge_wrong_pairs"] = float(np.median(arr(runs("judge", mode="own"), "wrong0")[:, 0])); save(fig, "toy3-judge.png")
    for mode in ["teacher_judged", "own_drop"]: nums[f"judge_{mode}_left"] = float(np.median(arr(runs("judge", mode=mode), f"nwrong{L}")[:, -1]))
    nums["judge_start_left"] = float(np.median(arr(runs("judge", mode="own"), f"nwrong{L}")[:, 0])); nums["judge_thin"] = float(np.median(arr(runs("judge", mode="teacher_judged"), f"size{L}")[:, -1] / arr(runs("judge", mode="teacher_judged"), f"size{L}")[:, 0]))


# ------------------------------------------------------------------ a generator with spurious entries of its own, never seeded by the teacher
def spur():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), constrained_layout=True)
    arms = [("teacher", TEACH, "-", "teacher chains"), ("own", OWN, ":", "own samples, judged; what passes is counted"), ("own_drop", OWN, "--", "and a failing entry loses a count"), ("own_delete", OWN, "-", "and a failing entry is removed")]
    for axis, rows, tag, title in zip(axes, [64, 2 ** 15], ["few", "many"], ["64 strong spurious entries", "32,768 weak spurious entries"]):
        for mode, color, ls, label in arms:
            rs = runs("spur", mode=mode, rows=rows); T = xs(rs, "tokens")[1:]; y = arr(rs, f"own{L}"); band(axis, T, y[:, 1:], color, label if rows == 64 else None, ls, mid=np.median)
            nums[f"spur_{tag}_{mode}_own"] = float(np.median(y[:, -1])); nums[f"spur_{tag}_{mode}_left"] = float(np.median(arr(rs, f"nwrong{L}")[:, -1])); nums[f"spur_{tag}_{mode}_own14"] = float(np.median(y[:, 1:][:, np.argmin(np.abs(T - 2 ** 14))]))
        nums[f"spur_{tag}_start_own"] = float(np.median(y[:, 0])); nums[f"spur_{tag}_start_left"] = float(np.median(arr(rs, f"nwrong{L}")[:, 0])); nums[f"spur_{tag}_share"] = float(np.median(arr(rs, f"wrong{L}")[:, 0]))
        axis.set_xscale("log"); axis.set_xlabel("further tokens (states shown or written)"); axis.set_ylabel(f"share of own chains of {2 ** L} links that are valid"); axis.set_title(title)
    fig.legend(loc="outside lower center", frameon=False, ncol=2); save(fig, "toy3-spurious.png")


# ------------------------------------------------------------------ what the check is for
def checks():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 5.6), constrained_layout=True)
    for gate, ls in [(True, "-"), (False, "--")]:
        rs = runs("slip", gate=gate); Mx = xs(rs, "M"); late = Mx >= 3000  # before that the top table is nearly empty and its share is noise
        for l in range(1, L + 1): band(ax, Mx[late], arr(rs, f"own{l}")[:, late], LEVEL[l - 1], f"{2 ** l} links, {'checked' if gate else 'unchecked'}", ls); nums[f"slip_{'gate' if gate else 'nogate'}{l}"] = float(arr(rs, f"own{l}")[:, -1].mean())
    ax.set_xscale("log"); ax.set_xlabel("practice episodes (1 written state in 10 is a slip)"); ax.set_ylabel("share of own samples that are valid"); ax.set_title("The check keeps slips out of the tables"); below(ax, 2)
    links = np.array([2 ** l for l in range(1, L + 1)])
    for i, F in enumerate([0, 32, 128, 512]):
        rs = runs("false", false_pairs=F); y = np.array([[r["out"][-1][f"own{l}"] for l in range(1, L + 1)] for r in rs]); color = plt.cm.Greys(0.35 + 0.2 * i)
        j = np.arange(40); pj = np.exp(-F / S) * (F / S) ** j / np.cumprod(np.r_[1.0, np.arange(1.0, 40)]); f_next = (pj * j / (M + j)).sum(); f_first = F / (S * M + F)  # wrong share of a state's accepted successors
        bx.errorbar(links, y.mean(0), yerr=[y.mean(0) - y.min(0), y.max(0) - y.mean(0)], marker="o", ls="", color=color, label=f"{F} wrong pairs"); bx.plot(links, (1 - f_first) * (1 - f_next) ** (links - 1), color=color, ls=":")
        nums.update({f"false{F}_{l}": float(y[:, l - 1].mean()) for l in range(1, L + 1)}); nums[f"false{F}_pred3"] = float((1 - f_first) * (1 - f_next) ** 7)
    bx.plot([], [], color=REF, ls=":", label="product over links"); bx.set_xscale("log", base=2); bx.set_xlabel("links in the span"); bx.set_ylabel("share of own samples that are valid"); bx.set_title("Wrong pairs in the recogniser compound"); below(bx, 2)
    save(fig, "toy3-checks.png")


# ------------------------------------------------------------------ what practice converges to
def bridge():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 5.4), constrained_layout=True); rs = runs("bridge"); Mx = xs(rs, "M")
    for l in [1, 2]:
        band(ax, Mx, arr(rs, f"cond{l}"), LEVEL[2 * l - 2], f"{SPAN[l - 1]}: midpoint given the ends"); band(ax, Mx, arr(rs, f"joint{l}"), LEVEL[2 * l - 2], f"{SPAN[l - 1]}: ends and midpoint", "--")
        nums[f"bridge_cond{l}"] = float(arr(rs, f"cond{l}")[:, -1].mean()); nums[f"bridge_joint{l}"] = float(arr(rs, f"joint{l}")[:, -1].mean()); nums[f"bridge_world{l}"] = float(arr(rs, f"wcond{l}")[:, -1].mean())
    ax.plot(Mx[Mx > 3e3], 30 * Mx[Mx > 3e3] ** -0.5, color=REF, ls=":", label=r"slope of $1/\sqrt{\mathrm{episodes}}$"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("practice episodes"); ax.set_ylabel("total variation distance to the exact tables")
    ax.set_title("Practised tables approach the exact ones"); below(ax)
    Sx, mx, Lx = 64, 3, 2; rng, succ, T, _ = setup(0, Sx, mx, Lx); o = Oracle(Sx, Lx); o.watch(chains(succ, 2 ** 15, 2 ** Lx + 1, rng), levels=False); done = 0  # one small world again, to show the tables themselves
    for Mc in grid(64, 2 ** 22, 40): o.rounds(int(Mc - done), rng); done = Mc
    C = o.pairs.reshape(Sx, Sx).astype(float); U = C / C.sum()
    for k in o.ks: Q = U[:, :, None] * (U / U.sum(1, keepdims=True))[None]; U = Q.sum(1)  # Q[a, c, b] at the top level
    c = o.code[2]; P = np.zeros(Sx ** 3); P[(c // Sx ** 2) * Sx ** 2 + (c % Sx) * Sx + c // Sx % Sx] = o.cnt[2]; P = P.reshape(Sx, Sx, Sx); pc, qc = P / np.maximum(P.sum(1, keepdims=True), 1), Q / np.maximum(Q.sum(1, keepdims=True), 1e-300)
    sel = Q > 0; bx.plot(qc[sel], pc[sel], marker=".", ls="", color=OWN, alpha=0.3); bx.plot([0, 1], [0, 1], color=REF, ls="--"); bx.set_xlabel("exact probability of the midpoint given the ends"); bx.set_ylabel("the Oracle's table after practice")
    bx.set_title(f"{SPAN[1].capitalize()}, one world"); nums["bridge_entries"] = int(sel.sum()); nums["bridge_corr"] = float(np.corrcoef(qc[sel], pc[sel])[0, 1]); nums["bridge_multi"] = float((qc[sel] < 0.999).mean())
    save(fig, "toy3-bridge.png")


# ------------------------------------------------------------------ own versus foreign
def own_foreign():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.4), constrained_layout=True); (ax, bx), (cx, dx) = axes
    arms = [(1.0, OWN, "own chains (start inside the map)", "own"), (0.5, THIRD, "half own, half foreign", "mix"), (0.0, TEACH, "foreign chains (the population)", "foreign")]
    for share, color, label, tag in arms:
        rs = runs("map", own_share=share); P = xs(rs, "P")[1:]; y = arr(rs, "map")[:, 1:]; band(ax, P, y, color, label, mid=np.median); band(cx, P, arr(rs, "write1")[:, 1:], color, label, mid=np.median); cx.plot(P, np.median(arr(rs, "write2")[:, 1:], 0), color=color, ls="--")
        t10 = np.array([cross(P, v, 0.1) for v in y]); nums[f"map_{tag}_10"] = med(t10); nums[f"map_{tag}_50"] = med([cross(P, v, 0.5) for v in y]); nums[f"map_{tag}_w1"] = med([cross(P, v, 0.5) for v in arr(rs, "write1")[:, 1:]])
        nums[f"map_{tag}_lo10"], nums[f"map_{tag}_hi10"] = float(np.nanmin(t10)), float(np.nanmax(t10)); bx.plot(arr(rs, "map").ravel(), arr(rs, "usable").ravel(), marker=".", ls="", color=color, alpha=0.4, label=label)
    for share, color in [(1.0, OWN), (0.0, TEACH)]:
        rs = runs("map_big", own_share=share); P = xs(rs, "P")[1:]; ax.plot(P, np.median(arr(rs, "map")[:, 1:], 0), color=color, ls=":"); nums[f"big_{'own' if share else 'foreign'}_10"] = med([cross(P, v, 0.1) for v in arr(rs, "map")[:, 1:]])
    ax.plot([], [], color=REF, ls=":", label="dotted: a world of 65,536 states"); ax.set_xscale("log"); ax.set_xlabel("examples"); ax.set_ylabel("share of the states in the Oracle's map"); ax.set_title("What the Oracle can read (16,384 states)"); below(ax)
    a = np.geomspace(5e-4, 1, 200); bx.plot(a, 1 - (1 - a) ** n, color=REF, ls="--", label=r"a scattered map: $1-(1-a)^9$"); bx.set_xscale("log"); bx.set_xlabel("share $a$ of the states in the map"); bx.set_ylabel("share of foreign chains that touch the map"); bx.set_title("Foreign chains the Oracle can use"); below(bx)
    rs = runs("map", own_share=1.0); amap, use = arr(rs, "map").ravel(), arr(rs, "usable").ravel(); sel = (amap > 0.07) & (amap < 0.14); nums["usable_at_10"] = float(use[sel].mean()); nums["usable_pred_10"] = float((1 - (1 - amap[sel]) ** n).mean()); nums["usable_a"] = float(amap[sel].mean())
    cx.plot([], [], color=REF, ls="--", label="dashed: spans of 4 links"); cx.set_xscale("log"); cx.set_xlabel("examples"); cx.set_ylabel("share of the population's spans of 2 links written"); cx.set_title("What the Oracle can write"); below(cx)
    sp = runs("map_split"); shares = sorted({r["args"]["own_share"] for r in sp})
    for i, P in enumerate([256, 1024, 4096]):
        y = np.array([[r["out"][-1]["map"] for r in sp if r["args"]["P_max"] == P and r["args"]["own_share"] == x] for x in shares]).T; band(dx, shares, y, plt.cm.Greys(0.4 + 0.25 * i), f"{P:,} examples in total", mid=np.median)
        nums[f"msplit{P}_0"], nums[f"msplit{P}_50"], nums[f"msplit{P}_100"] = (float(np.median(y[:, shares.index(v)])) for v in [0.0, 0.5, 1.0])
    dx.set_xlabel("share of the examples that are own"); dx.set_ylabel("share of the states in the map at the end"); dx.set_title("A fixed number of examples, split"); below(dx); save(fig, "toy3-own-foreign.png")


# ------------------------------------------------------------------ practice that skips what the Oracle holds, against fresh teacher chains
def aim():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.4), constrained_layout=True); (ax, bx), (cx, dx) = axes
    modes = [("fresh", TEACH, "-", "fresh teacher chains only"), ("blind", OWN, "--", "{N0} teacher chains, then blind practice"), ("aim", OWN, "-", "{N0} teacher chains, then practice that skips what it holds")]
    for axis, rare, N0, tag, title in [(ax, None, 2 ** 12, "u", "All steps equally likely"), (bx, 0.1, 2 ** 13, "r", "Common and rare steps (9 to 1)")]:
        for mode, color, ls, label in modes:
            rs = runs("aim_curve", mode=mode, rare=rare); T = xs(rs, "tokens"); y = arr(rs, "write3"); band(axis, T, y, color, label.format(N0=f"{N0:,}"), ls, mid=np.median)
            for lev in [0.5, 0.9, 0.98]: nums[f"aim_{tag}_{mode}_t{round(100 * lev)}"] = med([cross(T, v, lev) for v in y])
            for e in [19, 20, 21, 23]: nums[f"aim_{tag}_{mode}_w3_{e}"] = float(np.median(y[:, np.argmin(np.abs(T - 2 ** e))])); nums[f"aim_{tag}_{mode}_miss_{e}"] = 1 - nums[f"aim_{tag}_{mode}_w3_{e}"]
        nums[f"aim_{tag}_N0"] = f"{N0:,}"
        used, looked = arr(rs, "used")[:, -1], arr(rs, "looked")[:, -1]; nums[f"aim_{tag}_stop_used"] = med(used); nums[f"aim_{tag}_stop_w3"] = float(np.median(y[:, -1])); nums[f"aim_{tag}_looks"] = med(looked / (used - N0 * n))
        for lev in [0.5, 0.9]: nums[f"aim_{tag}_aim_t{round(100 * lev)}_alt"] = med([cross(u + k, v, lev) for u, k, v in zip(arr(rs, "used"), arr(rs, "looked"), y)])  # if every look-up cost one token
        for e in [20, 21]: nums[f"aim_{tag}_fail{e}"] = (1 - nums[f"aim_{tag}_fresh_w3_{e}"]) / (1 - nums[f"aim_{tag}_aim_w3_{e}"])
        for lev in [50, 90]: nums[f"aim_{tag}_gain{lev}"] = nums[f"aim_{tag}_fresh_t{lev}"] / nums[f"aim_{tag}_aim_t{lev}"]
        f, a = arr(runs("aim_curve", mode="fresh", rare=rare), "write3"), y; nums[f"aim_{tag}_cross"] = med([T[np.flatnonzero((av > fv) & (T > N0 * n))[0]] for fv, av in zip(np.array([np.interp(T, xs(runs("aim_curve", mode="fresh", rare=rare), "tokens"), v) for v in f]), a)])
        axis.set_xscale("log"); axis.set_xlabel("tokens (states shown or written)"); axis.set_ylabel("share of the population's spans of 8 links written"); axis.set_title(title); below(axis)
    rs = runs("aim_curve", mode="aim", rare=None); T = xs(rs, "tokens"); lv = np.stack([arr(rs, f"lvl{l}") for l in range(1, L + 1)]); d = np.diff(lv, axis=2); tot = d.sum(0)
    for l in range(L): cx.plot(T[1:], np.nanmedian(np.where(tot > 0, d[l] / np.where(tot > 0, tot, 1), np.nan), 0), color=LEVEL[l], label=f"joins for {SPAN[l]}")
    nums["aim_u_lvl2_first"] = float(np.median(d[1, :, 0] / tot[:, 0])); nums["aim_u_lvl3_last"] = float(np.nanmedian(np.where(tot > 0, d[2] / np.where(tot > 0, tot, 1), np.nan)[:, np.argmin(np.abs(T[1:] - 2 ** 20))]))
    cx.set_xscale("log"); cx.set_xlabel("tokens"); cx.set_ylabel("share of the practice tokens written since the last point"); cx.set_title("The practice moves up the levels by itself"); below(cx)
    for rare, tag in [(None, "u"), (0.1, "r")]:
        sp = runs("aim", rare=rare); shares = sorted({r["args"]["share"] for r in sp})
        for i, t in enumerate([2 ** 19, 2 ** 21, 2 ** 23]):
            y = {a: np.array([[r["out"]["write3"] for r in sp if r["args"]["tokens"] == t and r["args"]["share"] == x and r["args"]["aim"] == a] for x in shares]).T for a in [True, False]}; e = round(np.log2(t)); best = int(np.argmax(np.median(y[True], 0)))
            nums[f"asplit_{tag}_{e}_best"] = f"{100 * shares[best]:.0f}%"; nums[f"asplit_{tag}_{e}_aim"] = float(np.median(y[True][:, best])); nums[f"asplit_{tag}_{e}_fresh"] = float(np.median(y[True][:, -1])); nums[f"asplit_{tag}_{e}_blind"] = float(np.median(y[False], 0).max())
            nums[f"asplit_{tag}_{e}_wins"] = int((y[True][:, best] > y[True][:, -1]).sum())
            if rare is None: band(dx, shares, y[True], plt.cm.Greys(0.4 + 0.25 * i), f"$2^{{{e}}}$ tokens in total", mid=np.median); dx.plot(shares, np.median(y[False], 0), color=plt.cm.Greys(0.4 + 0.25 * i), ls="--")
    dx.plot([], [], color=REF, ls="--", label="dashed: blind practice"); dx.set_xscale("log"); dx.set_xlabel("share of the tokens spent on fresh teacher chains"); dx.set_ylabel("share of the population's spans of 8 links written"); dx.set_title("A token budget, split (all steps equally likely)"); below(dx, ncol=2); save(fig, "toy3-aim.png")


# ------------------------------------------------------------------ own versus foreign with nothing granted: a learner at home in one of 16 regions
def reach():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.4), constrained_layout=True); (ax, bx), (cx, dx) = axes; B = 1 / 64
    arms = [(1.0, OWN, "practice only (own examples)", "own"), (0.5, THIRD, "half practice, half fresh chains", "mix"), (0.0, TEACH, "fresh chains of the population only", "fresh")]
    at = lambda rs, key, T: np.median(arr(rs, key)[:, np.argmin(np.abs(xs(rs, "tokens") - T))])
    for i, (b, label, tag) in enumerate([(0.0, "no state leads to another region", "b0"), (1 / 1024, "1 state in 1,024 leads to another region", "b1024"), (B, "1 state in 64 leads to another region", "b64")]):
        rs = runs("reach", share=0.0, bridge=b, attach=None); x = xs(rs, "fresh")[1:]; y = arr(rs, "usable")[:, 1:]; band(ax, x, y, plt.cm.Greys(0.4 + 0.25 * i), label, mid=np.median)
        nums[f"reach_use0_{tag}"] = float(np.median(arr(rs, "usable")[:, 0])); nums[f"reach_use_end_{tag}"] = float(np.median(y[:, -1])); nums[f"reach_half_{tag}"] = med([cross(x, v, 0.5) for v in y]); nums[f"reach_most_{tag}"] = med([cross(x, v, 0.9) for v in y])
        nums[f"reach_lost_{tag}"] = float(np.median(1 - arr(rs, "taken")[:, -1] / arr(rs, "fresh")[:, -1])); nums[f"reach_never_{tag}"] = int(np.sum(y[:, -1] < 0.5))
    ax.axhline(1 / 16, color=REF, ls=":", label="the home region's share of the population"); ax.set_xscale("log"); ax.set_xlabel("fresh chains shown so far"); ax.set_ylabel("share of population chains the Oracle can take in"); ax.set_title("Fresh chains the Oracle can read"); below(ax)
    for share, color, label, tag in arms:
        rs = runs("reach", share=share, bridge=B, attach=None); T = xs(rs, "tokens")[1:]; band(bx, T, arr(rs, "map")[:, 1:], color, label, mid=np.median); band(cx, T, arr(rs, "home3")[:, 1:], color, label, mid=np.median); cx.plot(T, np.median(arr(rs, "home2")[:, 1:], 0), color=color, ls="--")
        band(dx, T, arr(rs, "pop2")[:, 1:], color, label, mid=np.median); dx.plot(T, np.median(arr(runs("reach", share=share, bridge=0.0, attach=None), "pop2")[:, 1:], 0), color=color, ls=":")
        for key in ["home1", "home2", "home3", "pop1", "pop2", "pop3", "map"]:
            for e in [16, 20, 23]: nums[f"reach_{tag}_{key}_{e}"] = float(at(rs, key, 2 ** e))
        for key in ["home3", "pop2", "pop3"]: nums[f"reach_{tag}_{key}_23_b0"] = float(at(runs("reach", share=share, bridge=0.0, attach=None), key, 2 ** 23))
        nums[f"reach_{tag}_half_home3"] = med([cross(T, v, 0.5) for v in arr(rs, "home3")[:, 1:]]); nums[f"reach_{tag}_half_pop2"] = med([cross(T, v, 0.5) for v in arr(rs, "pop2")[:, 1:]])
        ms = runs("reach", share=share, K=1)
        for key in ["home1", "home2", "home3", "map", "links"]: nums[f"mixed_{tag}_{key}_0"] = float(np.median(arr(ms, key)[:, 0])); nums[f"mixed_{tag}_{key}_20"] = float(at(ms, key, 2 ** 20)); nums[f"mixed_{tag}_{key}_23"] = float(at(ms, key, 2 ** 23))
        nums[f"mixed_{tag}_usable_0"] = float(np.median(arr(ms, "usable")[:, 0]))
    rs = runs("reach", share=0.0, bridge=B, attach=False); T = xs(rs, "tokens")[1:]; dx.plot(T, np.median(arr(rs, "pop2")[:, 1:], 0), color=REF, ls="--", label="fresh chains only, a learner that keeps every chain")
    nums["reach_keep_half_pop2"] = med([cross(T, v, 0.5) for v in arr(rs, "pop2")[:, 1:]]); nums["reach_start_home2"], nums["reach_start_home3"] = (float(np.median(arr(rs, k)[:, 0])) for k in ["home2", "home3"])
    nums["reach_stage1_map"] = float(np.median(arr(rs, "map")[:, 0])); nums["reach_stage1_links"] = float(np.median(arr(rs, "links")[:, 0])); nums["reach_keep_home3_23"] = float(at(rs, "home3", 2 ** 23))
    bx.set_xscale("log"); bx.set_xlabel("tokens in stage 2 (states shown or written)"); bx.set_ylabel("share of the world's states the Oracle knows"); bx.set_title("Only fresh chains enlarge the map"); below(bx)
    cx.plot([], [], color=REF, ls="--", label="dashed: spans of 4 links"); cx.set_xscale("log"); cx.set_xlabel("tokens in stage 2"); cx.set_ylabel("share of the home region's spans of 8 links written"); cx.set_title("Writing at home"); below(cx)
    dx.plot([], [], color=REF, ls=":", label="dotted: no state leads to another region"); dx.set_xscale("log"); dx.set_xlabel("tokens in stage 2"); dx.set_ylabel("share of the population's spans of 4 links written"); dx.set_title("Writing in the whole population"); below(dx); save(fig, "toy3-reach.png")


if __name__ == "__main__":
    for f in [scheme, headstart, practice, staircase, tradeoff, aim, supervision, judge, spur, checks, bridge, own_foreign, reach]: f()
    nums.update({"S": S, "m": M, "n": n, "L": L}); (OUT / "numbers.json").write_text(json.dumps(nums, indent=1)); print(OUT / "numbers.json")
