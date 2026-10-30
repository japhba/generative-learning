"""Finite sigmoid belief networks: exact wake gradients and sampled sleep targets."""

import argparse
import hashlib
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def states(n): return np.array(list(itertools.product((-1., 1.), repeat=n)))


def softmax(a):
    a = a - a.max(axis=-1, keepdims=True)
    logq = a - np.log(np.exp(a).sum(axis=-1, keepdims=True))
    return np.exp(logq), logq


def sigmoid(a): return np.exp(-np.logaddexp(0, -a))


class BeliefNetwork:
    def __init__(self, layers, visible, rng, strength):
        self.layers, self.visible = layers, visible
        self.h, self.x = states(sum(layers)), states(visible)
        self.blocks = np.split(self.h, np.cumsum(layers)[:-1], axis=1)
        dims = layers + [visible]
        self.weights = [rng.normal(size=(a, b)) * strength / np.sqrt(a) for a, b in zip(dims[:-1], dims[1:], strict=True)]
        self.biases = [rng.normal(size=b) * .15 for b in dims[1:]]

    def parameters(self): return self.weights + self.biases

    def distribution(self):
        logprior = np.full(len(self.h), -self.layers[0] * np.log(2))
        for parent, child, w, b in zip(self.blocks[:-1], self.blocks[1:], self.weights[:-1], self.biases[:-1], strict=True):
            a = parent @ w + b
            logprior += (-np.logaddexp(0, -child * a)).sum(axis=1)
        a = self.blocks[-1] @ self.weights[-1] + self.biases[-1]
        logjoint = .5 * self.x @ a.T - np.logaddexp(-a / 2, a / 2).sum(axis=1)[None, :] + logprior[None, :]
        joint = np.exp(logjoint)
        assert abs(joint.sum() - 1) < 1e-10
        return joint, logjoint

    def gradient(self, responsibilities):
        marginal = responsibilities.sum(axis=0)
        gw, gb = [], []
        for parent, child, w, b in zip(self.blocks[:-1], self.blocks[1:], self.weights[:-1], self.biases[:-1], strict=True):
            delta = marginal[:, None] * (sigmoid(parent @ w + b) - (child + 1) / 2)
            gw.append(parent.T @ delta); gb.append(delta.sum(axis=0))
        parent = self.blocks[-1]
        delta = marginal[:, None] * sigmoid(parent @ self.weights[-1] + self.biases[-1]) - responsibilities.T @ ((self.x + 1) / 2)
        gw.append(parent.T @ delta); gb.append(delta.sum(axis=0))
        return gw + gb


class Recognizer:
    def __init__(self, dim, count, depth, width, rng, neglect):
        dims = [dim] + [width] * (depth - 1) + [count]
        self.weights = [rng.normal(size=(a, b)) / np.sqrt(a) for a, b in zip(dims[:-1], dims[1:], strict=True)]
        self.weights[-1] *= .05
        self.biases = [np.zeros(b) for b in dims[1:]]
        # Random state preferences, independent of the generator and evaluation labels.
        self.biases[-1] = rng.normal(size=count) * neglect

    def parameters(self): return self.weights + self.biases

    def forward(self, x):
        hidden = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1], strict=True): hidden.append(np.tanh(hidden[-1] @ w + b))
        q, logq = softmax(hidden[-1] @ self.weights[-1] + self.biases[-1])
        return q, logq, hidden

    def backward(self, hidden, delta):
        gw, gb = [], []
        for k in reversed(range(len(self.weights))):
            gw.append(hidden[k].T @ delta); gb.append(delta.sum(axis=0))
            if k: delta = (delta @ self.weights[k].T) * (1 - hidden[k]**2)
        return list(reversed(gw)) + list(reversed(gb))


class Optimizer:
    def __init__(self, parameters, name, lr):
        assert name in ("sgd", "adam", "adam_eps12")
        self.name, self.lr, self.step = name, lr, 0
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]

    def update(self, parameters, gradients):
        self.step += 1
        for p, g, m, v in zip(parameters, gradients, self.m, self.v, strict=True):
            assert np.isfinite(g).all()
            if self.name == "sgd": p -= self.lr * g
            else:
                m *= .9; m += .1 * g; v *= .999; v += .001 * g**2
                epsilon = 1e-12 if self.name == "adam_eps12" else 1e-8
                p -= self.lr * (m / (1 - .9**self.step)) / (np.sqrt(v / (1 - .999**self.step)) + epsilon)


def components(generator, recognizer, px, arm, rng, batch, sampled):
    joint, logjoint = generator.distribution()
    model_px = joint.sum(axis=1)
    post, logpost = softmax(logjoint)
    q, logq, hidden = recognizer.forward(generator.x)
    residual = logq - logpost
    delta = px[:, None] * q * (residual - (q * residual).sum(axis=1, keepdims=True))
    gr = recognizer.backward(hidden, delta)
    gw = generator.gradient(px[:, None] * q)
    if arm == "forward": gr = recognizer.backward(hidden, px[:, None] * (q - post))
    if arm == "sleep":
        if sampled:
            index = rng.choice(joint.size, size=batch, p=joint.ravel())
            xindex, zindex = np.unravel_index(index, joint.shape)
            sq, _, shidden = recognizer.forward(generator.x[xindex])
            sq[np.arange(batch), zindex] -= 1
            extra = recognizer.backward(shidden, sq / batch)
        else: extra = recognizer.backward(hidden, model_px[:, None] * (q - post))
        gr = [a + b for a, b in zip(gr, extra, strict=True)]
    return gw, gr


def metrics(generator, recognizer, truejoint, trueweights):
    joint, logjoint = generator.distribution()
    px, truepx = joint.sum(axis=1), truejoint.sum(axis=1)
    post, logpost = softmax(logjoint)
    truepost = truejoint / truepx[:, None]
    q, logq, _ = recognizer.forward(generator.x)
    forward = (truepost * (np.log(truepost) - logq)).sum(axis=1)
    reverse = (q * (logq - logpost)).sum(axis=1)
    responsibility_error = np.sum(truepx[:, None] * (q - truepost)**2)
    model_kl = np.sum(truepx * np.log(truepx / px))
    state_mass = np.sum(truejoint * (q < .1 * truepost), axis=0)
    design = np.column_stack((generator.x, np.ones(len(generator.x))))
    centered_q = logq - logq.mean(axis=1, keepdims=True)
    centered_p = logpost - logpost.mean(axis=1, keepdims=True)
    qr = np.linalg.lstsq(design * np.sqrt(truepx[:, None]), centered_q * np.sqrt(truepx[:, None]), rcond=None)[0]
    pr = np.linalg.lstsq(design * np.sqrt(truepx[:, None]), centered_p * np.sqrt(truepx[:, None]), rcond=None)[0]
    return {"posterior_kl": float(truepx @ forward), "reverse_kl": float(truepx @ reverse),
            "elbo_excess": float(model_kl + truepx @ reverse), "marginal_kl": float(max(0, model_kl)),
            "posterior_brier": float(responsibility_error), "neglected_mass": float(state_mass.sum()),
            "posterior_map_agreement": float(np.sum(truepx * (q.argmax(axis=1) == truepost.argmax(axis=1)))),
            "recognition_coefficient_mse": float(np.mean((qr - pr)**2)),
            "recognition_weight_overlap": float(np.sum(qr * pr) / (np.linalg.norm(qr) * np.linalg.norm(pr))),
            "per_state_brier": np.sum(truepx[:, None] * (q - truepost)**2, axis=0).tolist(),
            "per_state_neglected_mass": state_mass.tolist(),
            "weight_mse": float(np.mean(np.concatenate([(a - b).ravel()**2 for a, b in zip(generator.weights, trueweights, strict=True)]))),
            "per_state_kl": np.sum(truejoint * (np.log(truepost) - logq), axis=0).tolist()}


def costs(generator, recognizer, batch, sampled):
    nx, nz, h = len(generator.x), len(generator.h), generator.visible
    rec = sum(w.size for w in recognizer.weights)
    # Counts also hidden activation backward work conservatively as dense-equivalent.
    recognition = 6 * nx * rec
    gen = 4 * nx * nz * h + 6 * nz * sum(w.size for w in generator.weights)
    sleep = 6 * (batch if sampled else nx) * rec + 2 * nz * sum(w.size for w in generator.weights)
    return {"wake": recognition + gen, "sleep": sleep}


def run(config):
    rng = np.random.default_rng(config["seed"])
    gen = BeliefNetwork(config["layers"], config["visible"], rng, config["strength"])
    truejoint, _ = gen.distribution()
    trueweights = [w.copy() for w in gen.weights]
    if config["coupled"]:
        for p in gen.parameters(): p += rng.normal(size=p.shape) * config["generator_noise"]
    rec = Recognizer(config["visible"], len(gen.h), config["recognition_depth"], config["width"], rng, config["neglect"])
    optg, optr = Optimizer(gen.parameters(), config["optimizer"], config["lr_g"]), Optimizer(rec.parameters(), config["optimizer"], config["lr_r"])
    px = truejoint.sum(axis=1)
    draw_rng = np.random.default_rng(config["seed"] + 200000)
    real_rng = np.random.default_rng(config["seed"] + 100000)
    cost = costs(gen, rec, config["batch"], config["sampled"])
    unit = cost["wake"] + (cost["sleep"] if config["arm"] == "sleep" else 0)
    nsteps = int(config["budget"] * cost["wake"] / unit)
    checkpoints = set(np.rint(np.geomspace(1, nsteps, config["eval_points"])).astype(int)) | {0, nsteps}
    name = config["name"]
    output = Path(config["output"]) / name
    output.mkdir(parents=True, exist_ok=False)
    rows, cpu, reason = [], 0., "budget"
    for step in range(nsteps + 1):
        if step in checkpoints:
            metric = metrics(gen, rec, truejoint, trueweights)
            assert np.isfinite(metric["posterior_kl"]) and metric["posterior_kl"] < 200
            rows.append({"step": step, "compute": step * unit, "wake_equivalent": step * unit / cost["wake"], "cpu_seconds": cpu,
                         "dreams": step * config["batch"] if config["sampled"] and config["arm"] == "sleep" else 0, **metric})
            (output / "run.json").write_text(json.dumps({"config": config, "costs": cost, "rows": rows, "stop_reason": reason}, indent=2))
            if step >= 256:
                window = [r for r in rows if r["step"] >= step / 4]
                slopes = [np.polyfit(np.log([r["step"] for r in window]), np.log([max(r[key], 1e-15) for r in window]), 1)[0] for key in ("elbo_excess", "posterior_kl")]
                if min(slopes) > -.02: reason = "stationary_loss_and_eval"; break
            if metric["posterior_kl"] < config["accuracy_stop"]: reason = "accuracy_target"; break
        if step == nsteps: break
        start = time.process_time()
        batch_px = np.bincount(real_rng.choice(len(px), size=config["batch"], p=px), minlength=len(px)) / config["batch"] if config["sampled"] else px
        gg, gr = components(gen, rec, batch_px, config["arm"], draw_rng, config["batch"], config["sampled"])
        optr.update(rec.parameters(), gr)
        if config["coupled"]: optg.update(gen.parameters(), gg)
        cpu += time.process_time() - start
    (output / "run.json").write_text(json.dumps({"config": config, "costs": cost, "rows": rows, "stop_reason": reason}, indent=2))
    np.savez_compressed(output / "weights.npz", **{f"g{i}": w for i, w in enumerate(gen.parameters())}, **{f"q{i}": w for i, w in enumerate(rec.parameters())})
    return {"name": name, "stop": reason, "cpu_seconds": cpu, "initial_kl": rows[0]["posterior_kl"], "final_kl": rows[-1]["posterior_kl"]}


def audit():
    rng = np.random.default_rng(100)
    errors = []
    for layers in ([3], [2, 2], [1, 1, 1]):
        gen = BeliefNetwork(list(layers), 3, rng, 1.5)
        joint, logjoint = gen.distribution(); px = joint.sum(axis=1)
        for depth in (1, 2, 3):
            rec = Recognizer(3, len(gen.h), depth, 5, rng, 1.)
            for arm in ("elbo", "sleep", "forward"):
                gg, gr = components(gen, rec, px, arm, rng, 128, False)
                def rec_loss():
                    q, logq, _ = rec.forward(gen.x)
                    reverse = np.sum(px[:, None] * q * (logq - logjoint))
                    forward = -np.sum(joint * logq)
                    return reverse if arm == "elbo" else forward if arm == "forward" else reverse + forward
                for p, g in zip(rec.parameters(), gr, strict=True):
                    direction = rng.normal(size=p.shape); direction /= np.linalg.norm(direction)
                    p += 1e-5 * direction; plus = rec_loss()
                    p -= 2e-5 * direction; minus = rec_loss()
                    p += 1e-5 * direction
                    errors.append(abs((plus - minus) / 2e-5 - np.sum(direction * g)))
            q = rec.forward(gen.x)[0]
            for p, g in zip(gen.parameters(), gg, strict=True):
                direction = rng.normal(size=p.shape); direction /= np.linalg.norm(direction)
                p += 1e-5 * direction; plus = -np.sum(px[:, None] * q * gen.distribution()[1])
                p -= 2e-5 * direction; minus = -np.sum(px[:, None] * q * gen.distribution()[1])
                p += 1e-5 * direction
                errors.append(abs((plus - minus) / 2e-5 - np.sum(direction * g)))
    # Directional local-Hessian equality, for arbitrary logits and directions.
    p, lp = softmax(rng.normal(size=(31, 17)))
    direction = rng.normal(size=p.shape)
    hessian_errors = []
    for epsilon in (1e-2, 1e-3, 1e-4):
        q, lq = softmax(lp + epsilon * direction)
        reverse = q * ((lq - lp) - np.sum(q * (lq - lp), axis=1, keepdims=True))
        forward = q - p
        hessian_errors.append({"epsilon": epsilon, "relative_gradient_difference": float(np.linalg.norm(reverse - forward) / np.linalg.norm(forward))})
    assert max(errors) < 1e-7
    result = {"directional_derivatives": len(errors), "max_gradient_error": max(errors), "local_curvature": hessian_errors}
    print(json.dumps(result, indent=2)); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol"); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit: audit(); return
    protocol = json.loads(Path(args.protocol).read_text())
    output = Path(protocol["base"]["output"]); output.mkdir(parents=True, exist_ok=False)
    (output / "runtime.py").write_bytes(Path(__file__).read_bytes())
    (output / "runtime.sha256").write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, {**protocol["base"], **case}) for case in protocol["cases"]]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Deep wake-sleep"):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__": main()
