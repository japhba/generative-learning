"""Five online learners of a repeated branching rule. One update sees one visible string per world."""
import argparse
import copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

M = 10
RHO = .94
LIMIT = np.arctanh(RHO)
WATCH = 32
RATE = .18
THRESHOLD = .55
NAMES = ("blind", "direct", "local", "scalar", "pairs")
MARKS = (0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)
METRICS = ("kl", "deep_kl", "root", "oriented_root", "shallow", "elig_teacher", "elig_own",
           "alignment", "reader_alignment", "shallow_alignment", "coverage", "deep_tv", "visible_lower", "visible_upper")
PATTERNS = 2 * ((np.arange(2 ** M)[:, None] >> np.arange(M)) % 2) - 1.
DATA = Path("/nfs/gatsbystor/jbauer/generative-learning/results/repeated_branch")


def coins(rng, means):
    return np.where(rng.random(means.shape) < (1 + means) / 2, 1., -1.)


def world(rng, n):
    motif = coins(rng, np.zeros((n, M)))
    motif[:, 0] = 1.
    return motif


def draw(rng, top, bottom, reference=True, flat=False):
    """Uniform masks cancel in parity pooling. Return pooled leaves and the root for sleep/eval only."""
    z = coins(rng, np.zeros((len(top), 1)))
    h = z * coins(rng, top)
    if flat:
        return h, z
    errors = coins(rng, np.broadcast_to(bottom[:, None, :], (len(top), M, M)))
    if reference:
        errors[:, :, 0] = 1.
    return h[:, :, None] * errors, z


def mass(means, patterns=PATTERNS):
    """P(pattern | root=+1), independent binary channels."""
    return np.exp(.5 * np.log1p(-means ** 2).sum(1)[:, None]
                  - M * np.log(2) + np.arctanh(means) @ patterns.T)


def gate(field, method, gate_mode):
    if gate_mode == "off":
        return np.ones_like(field)
    if (method == "pairs" and gate_mode != "hard") or gate_mode == "smooth":
        return np.tanh((field / THRESHOLD) ** 2) ** 2
    return (np.abs(field) >= THRESHOLD).astype(float)


class Learner:
    def __init__(self, rng, start, method, gate_mode="default", reference=True, flat=False):
        self.method, self.gate_mode, self.reference = method, gate_mode, reference
        self.limit, self.noise = LIMIT, RHO
        self.hold_shallow = False
        self.lag_score = False
        self.previous_modulation = 0.
        self.flat = flat
        self.verdict_sign = np.ones_like(start)
        self.writer = start.copy()
        self.reader = start + rng.normal(0, .04, size=start.shape)
        self.bottom = np.zeros_like(start)
        self.shallow = rng.normal(0, .12, size=start.shape)
        self.shallow[:, 0] = .12

    def bottom_mean(self):
        return self.noise * self.bottom

    def reading_weights(self):
        return self.shallow if self.method == "direct" else self.reader

    def field(self, values):
        weights = self.reading_weights()
        return (values * weights).sum(1, keepdims=True) / np.abs(weights).sum(1, keepdims=True)

    def update(self, leaves, mode="both", trace=None):
        # The reference leaf fixes polarity for the local correlation and participates in the vote.
        if self.flat:
            h = leaves
            corr = h[:, :1] * h
        else:
            local_field = (leaves * self.shallow[:, None, :]).sum(2)
            h = np.where(local_field == 0, leaves[:, :, 0], np.sign(local_field))
            corr = (leaves[:, :, :1] * leaves).mean(1)
        field = self.field(h)
        z = np.sign(field)
        active = gate(field, self.method, self.gate_mode)
        assoc = z * h
        b = self.shallow * self.verdict_sign
        if self.method == "local":
            modulation = np.maximum(b * assoc, 0)
        elif self.method == "scalar":
            modulation = (b * assoc).mean(1, keepdims=True)
        elif self.method == "pairs":
            modulation = ((b * h).sum(1, keepdims=True) ** 2 - (b * b).sum(1, keepdims=True)) / (M * (M - 1))
        else:
            modulation = 1.
        if self.lag_score:
            modulation, self.previous_modulation = self.previous_modulation, modulation
        signal = active if mode in ("classic_wake", "plain_both") else active * modulation
        if mode == "sleep":
            self.reader = np.clip(self.reader + RATE * (trace * h - np.tanh(self.reader)), -self.limit, self.limit)
        else:
            self.writer = np.clip(self.writer + RATE * signal * (assoc - np.tanh(self.writer)), -self.limit, self.limit)
            if mode in ("both", "plain_both") and self.method != "direct":
                self.reader = np.clip(self.reader + RATE * signal * (assoc - np.tanh(self.reader)), -self.limit, self.limit)
        # Online correlations shared across equivalent bottom nodes of this one string.
        if mode in ("both", "wake", "classic_wake", "plain_both") and not self.flat:
            self.bottom = np.clip(self.bottom + .12 * corr, -1, 1)
        if mode in ("both", "sleep", "plain_both") and not self.hold_shallow:
            self.shallow = np.clip(self.shallow + .16 * corr, -1, 1)
            self.shallow[:, 0] = 1.

    def generate(self, rng):
        return draw(rng, np.tanh(self.writer), self.bottom_mean(), self.reference, self.flat)


def row_fidelity(reader, bottom, reference):
    # The first relative sign is +1 by construction. PATTERNS stores its absolute sign in column 0.
    if not reference:
        # The vote breaks ties in favor of the first leaf.
        fields = reader @ PATTERNS.T
        correct = (fields > 0) | ((fields == 0) & (PATTERNS[None, :, 0] > 0))
        return (mass(bottom) * correct).sum(1)
    patterns = PATTERNS[PATTERNS[:, 0] == 1]
    mass_rows = mass(np.column_stack((np.zeros(len(reader)), bottom[:, 1:])), patterns) * 2
    return (mass_rows * (reader @ patterns.T >= 0)).sum(1)


def measure(net, target, motif):
    g = np.tanh(net.writer)
    probs, plus = mass(g), mass(target)
    teacher = (plus + plus[:, ::-1]) / 2
    own = (probs + probs[:, ::-1]) / 2
    kl = (teacher * np.log(teacher / own)).sum(1)
    columns = slice(1, None) if net.reference else slice(None)
    p = (1 + RHO * motif[:, columns]) / 2
    q = (1 + net.bottom_mean()[:, columns]) / 2
    bottom_kl = M * (p * np.log(p / q) + (1-p) * np.log((1-p) / (1-q))).sum(1)

    shallow = row_fidelity(net.shallow, RHO * motif, net.reference)
    own_fid = row_fidelity(net.shallow, net.bottom_mean(), net.reference)
    if net.flat:
        bottom_kl = np.zeros(len(g))
        shallow = own_fid = np.ones(len(g))
    read_plus = mass(target * (2 * shallow[:, None] - 1))
    own_read_plus = mass(g * (2 * own_fid[:, None] - 1))
    weights = net.reading_weights()
    field = (weights @ PATTERNS.T) / np.abs(weights).sum(1)[:, None]
    eligibility = gate(field, net.method, net.gate_mode)
    root_corr = (read_plus * eligibility * np.sign(field)).sum(1)
    # Probability of the two teacher prototypes, sensitive to excess concentration as well as missing modes.
    coverage = (own * (np.abs(np.sign(target) @ PATTERNS.T) == M)).sum(1)
    aligned = (g * np.sign(target)).mean(1)
    reader_alignment = (np.tanh(weights) * np.sign(target)).mean(1)
    upper = kl + bottom_kl
    lower = upper
    if not net.reference:
        # Data processing on the recovered row signs below and on the joint (h, x) above.
        mix_t = (read_plus + read_plus[:, ::-1]) / 2
        mix_g = (own_read_plus + own_read_plus[:, ::-1]) / 2
        lower = (mix_t * np.log(mix_t / mix_g)).sum(1)
    return np.stack((upper, kl, (1 + np.abs(root_corr)) / 2, (1 + root_corr) / 2, shallow,
                     (read_plus * eligibility).sum(1), (own_read_plus * eligibility).sum(1), aligned,
                     reader_alignment, (net.bottom[:, 1:] * motif[:, 1:]).mean(1), coverage,
                     np.abs(teacher - own).sum(1)/2, lower, upper), axis=-1)


def setup(seed, n, method, variant):
    rng = np.random.default_rng([seed, 1])
    motif = world(rng, n)
    target_sign = world(rng, n) if variant == "untied" else motif
    target = RHO * target_sign
    start = .90 * coins(rng, np.zeros_like(motif))
    if variant == "orthogonal":
        balance = np.ones_like(motif)
        balance[:, :M//2] = -1
        start = .90 * rng.permuted(balance, axis=1) * target_sign
    if variant == "aligned":
        start = .90 * target_sign
    if variant == "diffuse":
        start *= .10
    gate_mode = variant if variant in ("off", "smooth", "hard") else "default"
    net = Learner(rng, start, method, gate_mode, variant != "no_reference", variant in ("depth_one", "flat_held"))
    twin = Learner(rng, .90 * coins(rng, np.zeros_like(motif)), method, gate_mode, variant != "no_reference", variant in ("depth_one", "flat_held"))
    if variant == "uncoupled":
        net.reader = .90 * coins(rng, np.zeros_like(motif)) + rng.normal(0, .04, size=start.shape)
    return motif, target, net, twin


def train(seed, n, method, arm, variant="default", watching=WATCH, marks=MARKS):
    motif, target, net, twin = setup(seed, n, method, variant)
    watch_rng = np.random.default_rng([seed, 2])
    rng = np.random.default_rng([seed, 3])
    twin_rng = np.random.default_rng([seed, 4])
    permutation_rng = np.random.default_rng([seed, 5])
    perm = permutation_rng.permuted(np.broadcast_to(np.arange(M), (n, M)), axis=1)
    pre = measure(net, target, motif)
    if variant == "fit_noise":
        corr_sum = np.zeros_like(motif[:, 1:])
        for learner in (net, twin):
            learner.limit, learner.noise = np.arctanh(.8), .8
    for index in range(watching):
        leaves, _ = draw(watch_rng, target, RHO * motif, net.reference, net.flat)
        net.update(leaves)
        twin.update(leaves)
        if variant == "fit_noise":
            corr_sum += (leaves[:, :, :1] * leaves[:, :, 1:]).sum(1)
            noise = np.abs(corr_sum).sum(1, keepdims=True) / ((index+1)*M*(M-1) + 2)
            for learner in (net, twin):
                learner.noise, learner.limit = noise, np.arctanh(noise)
                np.clip(learner.reader, -learner.limit, learner.limit, out=learner.reader)
                np.clip(learner.writer, -learner.limit, learner.limit, out=learner.writer)
    if variant == "erase_prime":
        _, _, init, _ = setup(seed, n, method, variant)
        net.writer, net.reader = init.writer, init.reader
    if variant == "reset_motif":
        net.shallow[:, 1:] = 0
    if variant == "open_at_fork":
        net.gate_mode = "off"
    net.hold_shallow = variant == "flat_held"
    net.lag_score = variant == "lag_score"
    if variant == "scramble_verdict":
        net.verdict_sign = coins(permutation_rng, np.zeros_like(motif))
    if arm == "matched":
        donor = copy.deepcopy(net)
    curve = []
    for t in range(max(marks)+1):
        if t in marks:
            curve.append(measure(net, target, motif))
        if t == max(marks):
            break
        if arm in ("teacher", "watching", "plain_wake", "plain_both") or (arm == "classic" and t % 2 == 0):
            leaves, _ = draw(watch_rng if arm == "watching" else rng, target, RHO * motif, net.reference, net.flat)
            net.update(leaves, "classic_wake" if arm in ("classic", "plain_wake") else "plain_both" if arm == "plain_both" else "both")
        elif arm == "random":
            leaves, _ = draw(rng, np.zeros_like(target), net.bottom_mean(), net.reference, net.flat)
            net.update(leaves)
        elif arm == "twin":
            leaves, _ = twin.generate(twin_rng)
            twin.update(leaves)
            net.update(leaves)
        elif arm == "matched":
            leaves, _ = donor.generate(rng)
            donor.update(leaves)
            # Transform the entire visible row, preserving its internal rendering. The teacher law is invariant.
            signed = leaves * np.sign(target)[:, :, None]
            leaves = np.take_along_axis(signed, perm[:, :, None], axis=1) * np.sign(target)[:, :, None]
            net.update(leaves)
        elif arm in ("own", "classic", "sleep", "frozen_reader"):
            leaves, z = net.generate(rng)
            mode = {"own": "both", "classic": "sleep", "sleep": "sleep", "frozen_reader": "wake"}[arm]
            net.update(leaves, mode, trace=z if mode == "sleep" else None)
        else:
            raise ValueError(f"Unknown continuation {arm}")
    return pre, np.stack(curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worlds", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=("main", "probes", "budgets", "watching", "pair_gate", "lag", "calibration", "challenges"), default="main")
    args = parser.parse_args()
    if args.suite == "main":
        jobs = [(m, a, "default", WATCH, MARKS) for m in NAMES for a in ("teacher", "own", "twin", "matched", "classic", "sleep", "plain_wake", "plain_both", "random", "frozen_reader")]
    elif args.suite == "probes":
        jobs = [(m, a, v, 0 if v == "no_watch" else WATCH, (0, 32, 128, 512)) for m in NAMES[2:]
                for v in ("untied", "orthogonal", "off", "smooth", "hard", "erase_prime", "no_watch", "reset_motif", "no_reference", "scramble_verdict", "diffuse", "uncoupled", "aligned", "depth_one")
                for a in ("teacher", "own")]
    elif args.suite == "challenges":
        jobs = [(m, a, v, WATCH, MARKS) for m in NAMES[2:] for a in ("teacher", "own")
                for v in ("open_at_fork", "flat_held")]
        jobs += [(m, a, "default", 50, (0, 50)) for m in NAMES for a in ("teacher", "own")]
    elif args.suite == "calibration":
        jobs = [(m, a, "fit_noise", WATCH, MARKS) for m in NAMES[2:] for a in ("own", "teacher", "matched", "classic")]
    elif args.suite == "lag":
        jobs = [(m, "own", "lag_score", WATCH, MARKS) for m in NAMES[2:]]
    elif args.suite == "watching":
        jobs = [(m, "watching", "default", 0, (0, 1, 2, 4, 8, 16, 24, 32)) for m in NAMES]
    elif args.suite == "pair_gate":
        jobs = [("pairs", a, v, WATCH, MARKS) for v in ("hard", "smooth")
                for a in ("teacher", "own", "twin", "matched", "classic", "plain_both", "random", "frozen_reader")]
    else:
        jobs = [(m, "own", "default", b-w, (w,)) for m in NAMES[2:] for b in (32, 64, 128, 256, 512)
                for w in (0, b//4, b//2, 3*b//4, b)]
    payload = [(args.seed, args.worlds, *job) for job in jobs]
    arrays = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for job, (pre, curve) in zip(jobs, tqdm(pool.map(run_job, payload), total=len(jobs))):
            m, a, v, watch, marks = job
            key = f"{m}/{a}/{v}/{watch}/{','.join(map(str, marks))}"
            arrays[key] = curve
            arrays[f"{m}/initial/{v}"] = pre
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, seed=args.seed, worlds=args.worlds, metrics=METRICS, **arrays)


def run_job(job):
    return train(*job)


if __name__ == "__main__":
    main()
