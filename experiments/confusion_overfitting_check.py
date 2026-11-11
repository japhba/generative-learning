"""What Canatar, Bordelon & Pehlevan (2021, arXiv:2006.13198) predict for the README's 'confusion overfitting'.

(a) Kernel regression, their Eq. 4, on a two-band spectrum: N1 modes the learner can fit now, a long tail it cannot
    fit yet. The target's power in the tail acts as label noise; c = (target power in the tail / in the head) divided by
    (kernel eigenvalue mass in the tail / in the head), so c = 1 is a target drawn from the kernel's own prior.
(b) Linear regression (white band, N features) around the interpolation threshold, simulated and from the theory:
    teacher examples only, half as many, the best ridge, and half teacher + half own examples (inputs from the same
    distribution, labels from the learner's function before training, here zero) that the learner fits as well.
Prints the threshold for a non-monotonic curve against their closed form g(lambda).
"""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm

OUT = Path(__file__).resolve().parents[1] / "reports" / "confusion-overfitting-check.png"


def kappa(P, eta, deg, lam):  # kappa = lam + sum_rho kappa eta_rho / (kappa + P eta_rho); the left minus the right side is convex with one root
    lo, hi = 0.0, lam + (deg * eta).sum()
    for _ in range(200):
        k = 0.5 * (lo + hi)
        lo, hi = (k, hi) if k - lam - (deg * k * eta / (k + P * eta)).sum() < 0 else (lo, k)
    return 0.5 * (lo + hi)


def eg(P, eta, deg, power, lam, sigma2=0.0):  # their Eq. 4 for bands: eigenvalue eta[b], degeneracy deg[b], target power power[b] = sum over the band of eta wbar^2
    k = kappa(P, eta, deg, lam); gamma = (deg * P * eta**2 / (k + P * eta) ** 2).sum()
    return ((k**2 * power / (k + P * eta) ** 2).sum() + sigma2 * gamma) / (1 - gamma)


def kappa_band(alpha, lam): return 0.5 * ((1 + lam - alpha) + np.sqrt((1 + lam + alpha) ** 2 - 4 * alpha))
def eg_band(alpha, lam, sigma2): k = kappa_band(alpha, lam); return (k**2 + sigma2 * alpha) / ((k + alpha) ** 2 - alpha)  # their Eq. 7, target power 1


def g(lam):  # their Eq. 8, with the branch of the arctangent that is continuous in lambda
    theta = (np.pi + np.arctan2(8 * np.sqrt(lam), 9 * lam * (3 * lam + 2) - 1)) / 3
    return 3 * lam * (3 * lam + 2 - 2 * np.sqrt(1 + lam) * np.sqrt(9 * lam + 1) * np.cos(theta))


def threshold(lam, alphas=np.geomspace(1e-3, 1e3, 200001)):  # smallest effective noise with a rising stretch in E_g(alpha)
    lo, hi = 0.0, 1e4
    for _ in range(60):
        s = 0.5 * (lo + hi)
        lo, hi = (lo, s) if (np.diff(eg_band(alphas, lam, s)) > 0).any() else (s, hi)
    return 0.5 * (lo + hi)


def with_own(alpha, lam, sigma2):  # half of the alpha N examples from the teacher, half own with label 0: mean label f/2, label variance 1/4 + sigma2/2
    k = kappa_band(alpha, lam); gamma = alpha / (k + alpha) ** 2
    return (1 - 0.5 * alpha / (alpha + k)) ** 2 + gamma / (1 - gamma) * (0.25 + 0.5 * sigma2 + 0.25 * k**2 / (k + alpha) ** 2)


def simulate(N, Ps, sigma2, lam_small, trials, rng):
    arms = {"teacher": np.zeros((trials, len(Ps))), "ridge": np.zeros((trials, len(Ps))), "half": np.zeros((trials, len(Ps))), "own": np.zeros((trials, len(Ps)))}
    def fit(X, y, lam): Psi = X / np.sqrt(N); return np.linalg.solve(Psi.T @ Psi + lam * np.eye(N), Psi.T @ y)
    for t in tqdm(range(trials), desc="trials"):
        w = rng.standard_normal(N); w *= np.sqrt(N) / np.linalg.norm(w)
        for i, P in enumerate(Ps):
            X = rng.standard_normal((P, N)); y = X @ w / np.sqrt(N) + np.sqrt(sigma2) * rng.standard_normal(P)
            h = P // 2; y_own = np.concatenate([y[:h], np.zeros(P - h)])
            for name, (Xa, ya, lam) in {"teacher": (X, y, lam_small), "ridge": (X, y, sigma2), "half": (X[:h], y[:h], lam_small), "own": (X, y_own, lam_small)}.items():
                arms[name][t, i] = ((fit(Xa, ya, lam) - w) ** 2).mean()
    return arms


def main():
    print("lambda   threshold (numerical)   g(lambda) | 2 lambda + 1   threshold / lambda")
    for lam in [0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]: print(f"{lam:6.2f}   {threshold(lam):12.4f}   {g(lam):12.4f} | {2 * lam + 1:6.2f}   {threshold(lam) / lam:8.2f}")

    fig, (ax, bx) = plt.subplots(2, 1, layout="constrained")

    N1, N2, lam_t = 100, 10**6, 0.2  # head: N1 modes, total eigenvalue mass 1; tail: N2 modes, total mass lam_t (the effective ridge)
    eta, deg = np.array([1 / N1, lam_t / N2]), np.array([N1, N2]); Ps = np.geomspace(1, 30 * N1, 400)
    print(f"panel a: effective ridge {lam_t}, threshold for a bump: c > {threshold(lam_t) / lam_t:.2f}")
    for c in [1, 3, 10, 30]:
        power = np.array([1.0, c * lam_t]); curve = np.array([eg(P, eta, deg, power, 1e-9) for P in Ps]) / power.sum()
        ax.plot(Ps / N1, curve, label=f"c = {c}"); print(f"  c = {c}: max over P of E_g / E_g(0) = {curve.max():.3f} at P / N1 = {Ps[curve.argmax()] / N1:.2f}; at P / N1 = 30: {curve[-1]:.3f} (floor {c * lam_t / power.sum():.3f})")
    ax.axvline(1 + lam_t, ls=":", c="k"); ax.axhline(1, ls=":", c="k"); ax.set_xscale("log"); ax.set_xlabel("teacher examples / features in reach, $P/N_1$"); ax.set_ylabel("error / error with no examples"); ax.set_title("a  kernel regression, theory"); ax.legend(title="tail power of the target\nrelative to the kernel", loc="center left", bbox_to_anchor=(1, 0.5))

    N, sigma2, lam_small, trials = 200, 0.5, 1e-3, 40; Psim = np.unique(np.round(np.linspace(0.1, 3, 30) * N).astype(int)); al = np.linspace(0.05, 3, 600)
    arms = simulate(N, Psim, sigma2, lam_small, trials, np.random.default_rng(0))
    theory = {"teacher": eg_band(al, lam_small, sigma2), "ridge": eg_band(al, sigma2, sigma2), "half": eg_band(al / 2, lam_small, sigma2), "own": with_own(al, lam_small, sigma2)}
    labels = {"teacher": "$P$ teacher, fits all", "ridge": "$P$ teacher, best ridge", "half": "$P/2$ teacher", "own": "$P/2$ teacher + $P/2$ own"}
    for name in arms:
        line, = bx.plot(al, theory[name], label=labels[name]); bx.errorbar(Psim / N, arms[name].mean(0), arms[name].std(0) / np.sqrt(trials), fmt="o", c=line.get_color(), ms=3)
        print(f"panel b, {name}: simulated / theory at P/N = {Psim[[4, 9, 14, 29]] / N}: {np.round(arms[name].mean(0)[[4, 9, 14, 29]], 3)} / {np.round(np.interp(Psim[[4, 9, 14, 29]] / N, al, theory[name]), 3)}")
    bx.set_yscale("log"); bx.set_xlabel("examples / features, $P/N$"); bx.set_ylabel("population error"); bx.set_title("b  linear regression, simulation (dots) and theory"); bx.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig(OUT); print(OUT)


if __name__ == "__main__":
    main()
