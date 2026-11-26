"""Three-level replication of the local and scalar rules, with an independent teacher panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from repeated_branch import Learner, LIMIT, M, RATE, RHO, THRESHOLD, WATCH, coins, row_fidelity, world

TIMES = (0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 224, 288, 416, 544)
METRICS = ("visible_kl_mc", "visible_kl_mc_se", "shallow", "middle", "top", "middle_alignment", "top_alignment")


def relay(children, reader):
    # [world, site, branch]. Lower-level silence appears as a zero input.
    signal = (children * reader[:, None]).sum(-1) / np.abs(reader).sum(1)[:, None]
    return np.where(np.abs(signal) >= THRESHOLD, np.sign(signal), 0.)


def update_group(group, children, output, shallow, method):
    association = output[:, :, None] * children
    active = (output[:, :, None] != 0) & (children != 0)
    local = association * shallow[:, None]
    score = np.maximum(local, 0) if method == "local" else local.mean(-1, keepdims=True)
    for weights in (group.reader, group.writer):
        step = RATE * (active * score * (association - np.tanh(weights)[:, None])).mean(1)
        np.clip(weights + step, -LIMIT, LIMIT, out=weights)


def broadcast(rng, parents, mean):
    shape = (*parents.shape, M)
    rates = mean.reshape((len(mean),) + (1,) * (parents.ndim - 1) + (M,))
    return parents[..., None] * coins(rng, np.broadcast_to(rates, shape))


def log_prob(h, middle, top):
    """Exact log probability of the array of first-level causes, marginalizing the root and its children."""
    mids = (h * np.arctanh(middle)[:, None, None]).sum(-1)
    offset = (.5 * np.log1p(-middle ** 2).sum(-1) - M*np.log(2))[:, None, None]
    logp = np.log((1 + top[:, None]) / 2)
    logm = np.log((1 - top[:, None]) / 2)
    positive = (np.logaddexp(mids + logp, -mids + logm) + offset).sum(-1)
    negative = (np.logaddexp(mids + logm, -mids + logp) + offset).sum(-1)
    return np.logaddexp(positive, negative) - np.log(2)


def run(job):
    seed, n, panel, method, arm = job
    rng = np.random.default_rng([seed, 1])
    motif = world(rng, n)
    target = RHO * motif
    middle = Learner(rng, .90 * coins(rng, np.zeros_like(motif)), method)
    top = Learner(rng, .90 * coins(rng, np.zeros_like(motif)), method)
    watch_rng, continuation = (np.random.default_rng([seed, i]) for i in (2, 3))
    test_rng = np.random.default_rng([seed, 4])
    z = coins(test_rng, np.zeros((n, panel)))
    k = broadcast(test_rng, z, target)
    h = broadcast(test_rng, k, target)
    read_noise = test_rng.random(h.shape)
    teacher_log = log_prob(h, target, target)
    curve = []
    for t in range(TIMES[-1] + 1):
        if t in TIMES:
            shallow = row_fidelity(middle.shallow, target, True)
            read_h = h * np.where(read_noise < shallow[:, None, None, None], 1., -1.)
            read_k = relay(read_h.reshape(n, -1, M), middle.reader).reshape(n, panel, M)
            read_z = relay(read_k, top.reader)
            acc_mid = (1 + np.abs((k * read_k).mean((1, 2))))/2
            acc_top = (1 + np.abs((z * read_z).mean(1)))/2
            p, q = (1+target[:, 1:])/2, (1+middle.bottom_mean()[:, 1:])/2
            bottom_kl = M**2 * (p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))).sum(1)
            ratio = teacher_log - log_prob(h, np.tanh(middle.writer), np.tanh(top.writer))
            curve.append(np.stack((bottom_kl + ratio.mean(1), ratio.std(1, ddof=1)/np.sqrt(panel),
                                   shallow, acc_mid, acc_top, (np.tanh(middle.writer)*motif).mean(1),
                                   (np.tanh(top.writer)*motif).mean(1)), axis=-1))
        if t == TIMES[-1]:
            break
        watching = t < WATCH or arm == "teacher"
        rng = watch_rng if t < WATCH else continuation
        top_mean = target if watching else np.tanh(top.writer)
        middle_mean = target if watching else np.tanh(middle.writer)
        bot_mean = target if watching else middle.bottom_mean()
        root = coins(rng, np.zeros(n))
        mid_state = broadcast(rng, root, top_mean)
        row_state = broadcast(rng, mid_state, middle_mean)
        leaves = broadcast(rng, row_state, bot_mean)
        leaves[..., 0] = row_state
        vote = (leaves * middle.shallow[:, None, None]).sum(-1)
        read_h = np.where(vote == 0, leaves[..., 0], np.sign(vote))
        read_k = relay(read_h, middle.reader)
        read_z = relay(read_k[:, None], top.reader)
        update_group(top, read_k[:, None], read_z, middle.shallow, method)
        if t < WATCH or arm != "freeze_middle":
            update_group(middle, read_h, read_k, middle.shallow, method)
        corr = (leaves[..., :1]*leaves).mean((1, 2))
        middle.bottom = np.clip(middle.bottom + .12*corr, -1, 1)
        middle.shallow = np.clip(middle.shallow + .16*corr, -1, 1)
        middle.shallow[:, 0] = 1.
    return np.stack(curve)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seed, worlds, panel = 261128, 96, 512
    jobs = [(seed, worlds, panel, m, a) for m in ("local", "scalar") for a in ("teacher", "own", "freeze_middle")]
    with ProcessPoolExecutor(max_workers=3) as pool:
        arrays = {f"{m}/{a}": x for (_, _, _, m, a), x in zip(jobs, tqdm(pool.map(run, jobs), total=len(jobs)))}
    np.savez_compressed(args.out, seed=seed, worlds=worlds, panel=panel, times=TIMES, metrics=METRICS, **arrays)
