"""Exact-enumeration Helmholtz experiment; no checker or true latents in training."""

import argparse
import hashlib
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def softmax(a):
    shifted = a - a.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return np.exp(logp), logp


def states(bits):
    return np.array(list(itertools.product((-1., 1.), repeat=bits)))


def joint(x, w, h, sigma):
    means = h @ w.T
    return (x @ means.T - .5 * np.sum(means**2, axis=1)) / sigma**2


def recognition(x, r, b, h, family):
    a = x @ r + b
    return softmax(a if family == "categorical" else a @ h.T)


def wake(x, w, r, b, h, sigma, family, oracle=False, forward=False):
    j = joint(x, w, h, sigma)
    q, logq = softmax(j) if oracle else recognition(x, r, b, h, family)
    mu = h @ w.T
    gw = ((q.sum(axis=0)[:, None] * mu - q.T @ x).T @ h) / (len(x) * sigma**2)
    if oracle: return gw, np.zeros_like(r), np.zeros_like(b)
    residual = logq - j
    delta = q - softmax(j)[0] if forward else q * (residual - np.sum(q * residual, axis=1, keepdims=True))
    if family == "factorized": delta = delta @ h
    return gw, x.T @ delta / len(x), delta.mean(axis=0)


def sleep(x, indices, r, b, h, family):
    q, _ = recognition(x, r, b, h, family)
    if family == "categorical":
        delta = q.copy()
        delta[np.arange(len(x)), indices] -= 1
    else: delta = q @ h - h[indices]
    return x.T @ delta / len(x), delta.mean(axis=0)


def example(config):
    rng = np.random.default_rng(config["seed"])
    bits, dim, rho = config["bits"], config["dim"], config["rho"]
    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    gram = (1 - rho) * np.eye(bits) + rho * np.ones((bits, bits))
    target = rotation[:, :bits] @ np.linalg.cholesky(gram).T
    initial = .8 * target + config["init_noise"] * rng.normal(size=target.shape) / np.sqrt(dim)
    return target, initial


def flops(config):
    """Leading dense multiply/add FLOPs; excludes elementwise ops and all evals."""
    batch, dim, bits = config["batch"], config["dim"], config["bits"]
    count = 2**bits
    out = count if config["family"] == "categorical" else bits
    rec = 2 * batch * dim * out
    state_map = 0 if config["family"] == "categorical" else 2 * batch * bits * count
    # joint logits, W gradient, and recognition forward/backward.
    wcost = 6 * count * bits * dim + 4 * batch * dim * count
    wake_cost = wcost + 2 * rec + 2 * state_map
    sleep_cost = 2 * batch * bits * dim + 2 * rec + 2 * state_map
    return {"wake": wake_cost, "sleep": sleep_cost, "oracle": wcost}


def diagnostics(x, target_j, target_logz, target, w, r, b, h, sigma, family):
    j = joint(x, w, h, sigma)
    post, logpost = softmax(j)
    q, logq = recognition(x, r, b, h, family)
    truepost, truelogpost = softmax(target_j)
    logz = np.logaddexp.reduce(j, axis=1)
    logratio = logz - target_logz
    # E_p*[p_w/p* - 1 - log(p_w/p*)] = KL(p* || p_w), nonnegative integrand.
    marginal = np.expm1(logratio) - logratio
    assert np.isfinite(marginal).all()
    cross = w.T @ target
    permutations = np.array(list(itertools.permutations(range(w.shape[1]))))
    score = np.abs(cross[np.arange(w.shape[1]), permutations]).sum(axis=1)
    best = permutations[score.argmax()]
    aligned = w * np.sign(cross[np.arange(w.shape[1]), best])[None, :]
    weight_error = np.sum((aligned - target[:, best])**2) / np.sum(target**2)
    gw, _, _ = wake(x, w, r, b, h, sigma, family)
    exact, _, _ = wake(x, w, r, b, h, sigma, family, oracle=True)
    numerator = np.sum(gw * exact)
    denom = np.linalg.norm(gw) * np.linalg.norm(exact)
    logplus = np.stack([np.logaddexp.reduce(logpost[:, h[:, bit] == 1], axis=1) for bit in range(h.shape[1])], axis=1)
    logminus = np.stack([np.logaddexp.reduce(logpost[:, h[:, bit] == -1], axis=1) for bit in range(h.shape[1])], axis=1)
    logfact = logplus @ ((h + 1) / 2).T + logminus @ ((1 - h) / 2).T
    learned_linear = r @ h / len(h) if family == "categorical" else r
    learned_pairs = h.T @ (b[:, None] * h) / len(h) if family == "categorical" else np.zeros((h.shape[1], h.shape[1]))
    ideal_pairs = -w.T @ w / sigma**2
    offdiag = ~np.eye(h.shape[1], dtype=bool)
    return {
        "marginal_kl": float(marginal.mean()), "marginal_kl_se": float(marginal.std(ddof=1) / np.sqrt(len(x))),
        "ordinary_excess_nll": float(-logratio.mean()), "density_ratio_mean": float(np.exp(logratio).mean()),
        "recognition_forward_kl": float(np.sum(post * (logpost - logq), axis=1).mean()),
        "recognition_reverse_kl": float(np.sum(q * (logq - logpost), axis=1).mean()),
        "true_posterior_kl": float(np.sum(truepost * (truelogpost - logq), axis=1).mean()),
        "posterior_total_correlation": float(np.sum(post * (logpost - logfact), axis=1).mean()),
        "weight_error": float(weight_error), "weight_alignment": float(score.max() / (np.linalg.norm(w) * np.linalg.norm(target))),
        "gradient_bias_squared": float(np.sum((gw - exact)**2)),
        "exact_gradient_squared": float(np.sum(exact**2)), "gradient_alignment": float(numerator / denom),
        "recognition_linear_weight_error": float(np.sum((learned_linear - w / sigma**2)**2) / np.sum((w / sigma**2)**2)),
        "recognition_pair_weight_mse": float(np.mean((learned_pairs - ideal_pairs)[offdiag]**2)),
    }


def run(config):
    destination = Path(config["output"])
    destination.mkdir(parents=True, exist_ok=True)
    name = f'{config["family"]}_rho{config["rho"]:g}_{config["arm"]}_s{config["seed"]}_lr{config["lr_r"]:g}_lw{config["lr_w"]:g}'
    h = states(config["bits"])
    target, initial = example(config)
    w = initial.copy()
    out = len(h) if config["family"] == "categorical" else config["bits"]
    r, b = np.zeros((config["dim"], out)), np.zeros(out)
    real_rng = np.random.default_rng(config["seed"] + 100_000)
    dream_rng = np.random.default_rng(config["seed"] + 200_000)
    eval_rng = np.random.default_rng(config["seed"] + config["eval_seed_offset"])
    x_eval = h[eval_rng.integers(len(h), size=config["eval_samples"])] @ target.T + config["sigma"] * eval_rng.normal(size=(config["eval_samples"], config["dim"]))
    target_j = joint(x_eval, target, h, config["sigma"])
    target_logz = np.logaddexp.reduce(target_j, axis=1)
    costs = flops(config)
    unit = costs["oracle"] if config["arm"] == "oracle" else costs["wake"] + (costs["sleep"] if config["arm"] in ("sleep", "frozen_sleep") else 0)
    budget = config["equivalent_steps"] * costs["wake"]
    steps = int(budget // unit)
    checkpoints = set(np.unique(np.rint(np.geomspace(1, steps, config["eval_points"])).astype(int))) | {0, steps}
    rows, cpu, stopped = [], 0., "budget_completed"
    for step in range(steps + 1):
        if step in checkpoints:
            metric = diagnostics(x_eval, target_j, target_logz, target, w, r, b, h, config["sigma"], config["family"])
            assert metric["marginal_kl"] < 20, f"Broken trajectory: {name}, {metric}"
            rows.append({"step": step, "flops": step * unit, "wake_equivalent": step * unit / costs["wake"], "real_samples": step * config["batch"], "dream_samples": step * config["batch"] if config["arm"] in ("sleep", "frozen_sleep") else 0, "training_cpu_seconds": cpu, **metric})
            result = {"config": config, "costs": costs, "stop_reason": stopped, "records": rows}
            (destination / f"{name}.json").write_text(json.dumps(result, indent=2))
            if step >= 2048:
                window = [row for row in rows if row["step"] >= step / 4]
                slopes = [np.polyfit(np.log([row["step"] for row in window]), np.log([row[key] for row in window]), 1)[0] for key in ("marginal_kl", "recognition_reverse_kl")]
                if min(slopes) > -.02:
                    stopped = "stationary_loss_and_eval"
                    break
        if step == steps: break
        start = time.process_time()
        x = h[real_rng.integers(len(h), size=config["batch"])] @ target.T + config["sigma"] * real_rng.normal(size=(config["batch"], config["dim"]))
        gw, gr, gb = wake(x, w, r, b, h, config["sigma"], config["family"], oracle=config["arm"] == "oracle", forward=config["arm"] == "wake_forward")
        if config["arm"] in ("sleep", "frozen_sleep"):
            indices = dream_rng.integers(len(h), size=config["batch"])
            source = initial if config["arm"] == "frozen_sleep" else w
            dreams = h[indices] @ source.T + config["sigma"] * dream_rng.normal(size=(config["batch"], config["dim"]))
            sr, sb = sleep(dreams, indices, r, b, h, config["family"])
            gr, gb = gr + sr, gb + sb
        decay = 1. / np.sqrt(1. + step * unit / costs["wake"] / config["decay_scale"])
        w -= decay * config["lr_w"] * gw
        r -= decay * config["lr_r"] * gr
        b -= decay * config["lr_r"] * gb
        assert np.isfinite(w).all() and np.isfinite(r).all() and np.isfinite(b).all(), name
        cpu += time.process_time() - start
    result = {"config": config, "costs": costs, "stop_reason": stopped, "records": rows}
    (destination / f"{name}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(destination / f"{name}.npz", target=target, initial=initial, w=w, r=r, b=b)
    return {"name": name, "stop": stopped, "cpu": cpu, "final": rows[-1]}


def audit():
    rng = np.random.default_rng(891)
    x, w, h = rng.normal(size=(13, 4)), rng.normal(size=(4, 3)) * .3, states(3)
    errors = []
    for family in ("categorical", "factorized"):
        out = len(h) if family == "categorical" else h.shape[1]
        r, b = rng.normal(size=(4, out)) * .2, rng.normal(size=out) * .1
        gradients = wake(x, w, r, b, h, .8, family)
        def objective():
            q, logq = recognition(x, r, b, h, family)
            return np.mean(np.sum(q * (logq - joint(x, w, h, .8)), axis=1))
        for param, grad in zip((w, r, b), gradients):
            for index in np.ndindex(param.shape):
                old = param[index]
                param[index] = old + 1e-5
                plus = objective()
                param[index] = old - 1e-5
                minus = objective()
                param[index] = old
                errors.append(abs((plus - minus) / 2e-5 - grad[index]))
        indices = rng.integers(len(h), size=len(x))
        for param, grad in zip((r, b), sleep(x, indices, r, b, h, family)):
            for index in np.ndindex(param.shape):
                old = param[index]
                param[index] = old + 1e-5
                plus = -recognition(x, r, b, h, family)[1][np.arange(len(x)), indices].mean()
                param[index] = old - 1e-5
                minus = -recognition(x, r, b, h, family)[1][np.arange(len(x)), indices].mean()
                param[index] = old
                errors.append(abs((plus - minus) / 2e-5 - grad[index]))
    assert max(errors) < 1e-7
    exact_r = w @ h.T / .8**2
    exact_b = -.5 * np.sum((h @ w.T)**2, axis=1) / .8**2
    q = recognition(x, exact_r, exact_b, h, "categorical")[0]
    assert np.max(np.abs(q - softmax(joint(x, w, h, .8))[0])) < 1e-12
    print(json.dumps({"finite_difference_max_error": max(errors), "gradient_coordinates_checked": len(errors), "categorical_exact_posterior_error": float(np.max(np.abs(q - softmax(joint(x, w, h, .8))[0])))}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit()
        return
    protocol = json.loads(Path(args.protocol).read_text())
    output = Path(protocol["base"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (output / "runtime.py").write_text(Path(__file__).read_text())
    configs = [{**protocol["base"], **case} for case in protocol["cases"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, config) for config in configs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Coupled wake-sleep runs"):
            result = future.result()
            print(json.dumps(result), flush=True)
    (output / "runtime.sha256").write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "\n")


if __name__ == "__main__": main()
