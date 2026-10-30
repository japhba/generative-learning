"""Two-factor theory figure: the wake-sleep rotation torque and the rotation angle under practice vs a frozen recognizer."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World


def Q(t): return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def torque(theta, sigma, Y):
    U = Y @ Q(theta)  # rows are u = Q^T y
    return 0.5 * np.mean(np.tanh(U[:, 0] / sigma ** 2) * U[:, 1] - np.tanh(U[:, 1] / sigma ** 2) * U[:, 0])


def angle(W):  # rotation angle of the orthogonal polar factor of W, folded to (-45, 45] degrees
    U, _, Vt = np.linalg.svd(W); R = U @ Vt
    a = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return (a + 45) % 90 - 45


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    rng = np.random.default_rng(0); n = 200000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    thetas = np.linspace(0, np.pi / 2, 61)
    for sigma in [0.3, 0.5, 0.8]:
        Y = rng.choice([-1., 1.], size=(n, 2)) + sigma * rng.standard_normal((n, 2))
        axes[0].plot(np.degrees(thetas), [torque(t, sigma, Y) for t in thetas], label=f"σ = {sigma}")
    axes[0].axhline(0, color="k"); axes[0].set_xlabel("misalignment θ [degrees]"); axes[0].set_ylabel("expected dθ/dt (wake, exact recognizer)")
    axes[0].set_title("rotation torque, two factors"); axes[0].legend()
    sigma, lr, B, steps = 0.5, 0.02, 32, 6000
    for frozen in [False, True]:
        traj = []
        for s in tqdm(range(args.seeds), desc=f"frozen={frozen}"):
            r = np.random.default_rng([1, s]); world = World(2, 2, sigma, "factorial", r); world.A = np.eye(2)
            L = Learner(Q(np.radians(30)), np.zeros(2), np.zeros((2, 2)), np.zeros(2)); L.R = L.W.T / sigma ** 2
            for _ in range(5000): L.sleep(world, r, 0.2, B)
            X = world.sample_x(256, r)[0]; tr = []
            for t in range(steps):
                if t % 50 == 0: tr.append(angle(L.W))
                L.wake(X[r.integers(0, 256, B)], r, lr)
                if frozen: L.wake(X[r.integers(0, 256, B)], r, lr)
                else: L.sleep(world, r, lr * 10, B)
            traj.append(tr)
        traj = np.array(traj); t = np.arange(traj.shape[1]) * 50
        axes[1].plot(t, traj.mean(0), label="frozen recognizer (wake only)" if frozen else "wake-sleep (practice)")
        axes[1].fill_between(t, traj.min(0), traj.max(0), alpha=0.2)
    axes[1].axhline(0, color="k"); axes[1].set_xlabel("update"); axes[1].set_ylabel("rotation of the decoder θ [degrees]")
    axes[1].set_title(f"two factors, start at 30°, {args.seeds} seeds"); axes[1].legend()
    fig.tight_layout(); fig.savefig(args.out, dpi=150); print(args.out)


if __name__ == "__main__": main()
