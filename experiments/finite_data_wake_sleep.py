"""Fixed-task, finite-data wake/sleep with a learned generator and observable tests."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from deep_wake_sleep import BeliefNetwork, Optimizer, Recognizer, sigmoid, softmax, states


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("wake", "sleep", "forward", "predictive")


class FactorizedRecognizer:
    """Mean-field binary recognition, using exact expectations for wake updates."""

    def __init__(self, dim, count, depth, width, rng, neglect):
        bits = int(np.log2(count))
        assert 2 ** bits == count and depth == 1 and neglect == 0
        self.h = states(bits)
        self.weights = [rng.normal(size=(dim, bits)) * .05 / np.sqrt(dim)]
        self.biases = [np.zeros(bits)]

    def parameters(self): return self.weights + self.biases

    def forward(self, x):
        q, logq = softmax((x @ self.weights[0] + self.biases[0]) @ self.h.T)
        return q, logq, [x]

    def backward(self, hidden, delta):
        field_delta = delta @ self.h
        return [hidden[0].T @ field_delta, field_delta.sum(axis=0)]


def save_json(path, value):
    path.with_suffix(".tmp").write_text(json.dumps(value, indent=2, allow_nan=False))
    path.with_suffix(".tmp").replace(path)


def digest(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_world(config):
    gen = BeliefNetwork([config["latent_bits"]], config["context_bits"] + config["target_bits"], np.random.default_rng(config["world_seed"]), config["strength"])
    return gen.distribution()[0].sum(axis=1)


def data(config, seed, n):
    real = np.random.default_rng(seed).choice(2 ** (config["context_bits"] + config["target_bits"]), size=config["maximum_data"] , p=make_world(config))[:n]
    validation = np.arange(n) % 8 == 7
    return real, real[~validation], real[validation]


def generator_tables(gen, context_bits):
    logits = gen.h @ gen.weights[0] + gen.biases[0]
    context = states(context_bits)
    targets = states(gen.visible - context_bits)
    a = logits[:, :context_bits]
    context_logjoint = .5 * context @ a.T - np.logaddexp(-a / 2, a / 2).sum(axis=1)[None, :] - gen.layers[0] * np.log(2)
    post, logpost = softmax(context_logjoint)
    a = logits[:, context_bits:]
    pyh = np.exp(.5 * a @ targets.T - np.logaddexp(-a / 2, a / 2).sum(axis=1)[:, None])
    coefficient = np.vstack((.5 * logits[:, :context_bits].T, -np.logaddexp(-logits[:, :context_bits] / 2, logits[:, :context_bits] / 2).sum(axis=1)))
    coefficient -= coefficient.mean(axis=1, keepdims=True)
    return {"context": context, "post": post, "logpost": logpost, "pyh": pyh, "coefficient": coefficient, "model_prediction": post @ pyh}


def categorical_kl(prob, prediction):
    value = float(np.sum(prob * (np.log(prob / prob.sum(axis=1, keepdims=True)) - np.log(prediction))))
    assert value >= -1e-12 and np.isfinite(value)
    return max(0., value)


def fitted_generator(config, seed, n, destination, resume):
    destination.mkdir(parents=True, exist_ok=True)
    if resume and (destination / "generator.json").exists(): return json.loads((destination / "generator.json").read_text())
    real, train, validation = data(config, seed, n)
    truth = make_world(config)
    train_px = np.bincount(train, minlength=len(truth)) / len(train)
    val_px = np.bincount(validation, minlength=len(truth)) / len(validation)
    models, records = [], []
    total_steps, cpu = 0, 0.
    for restart in range(config["generator_restarts"]):
        gen = BeliefNetwork([config["latent_bits"]], config["context_bits"] + config["target_bits"], np.random.default_rng(seed + 100000 + restart), 1.)
        opt = Optimizer(gen.parameters(), "adam", config["generator_lr"])
        rows = []
        reason = "budget"
        for step in range(config["generator_steps"] + 1):
            start = time.process_time()
            joint, logjoint = gen.distribution()
            post, _ = softmax(logjoint)
            logpx = np.log(joint.sum(axis=1))
            prior = config["prior_precision"] / len(train)
            loss = float(-train_px @ logpx + .5 * prior * sum(np.sum(p ** 2) for p in gen.parameters()))
            if step % 64 == 0 or step == config["generator_steps"]:
                rows.append({"step": step, "train_map": loss, "validation_nll": float(-val_px @ logpx), "test_marginal_kl": float(truth @ (np.log(truth) - logpx))})
                assert np.isfinite(loss) and loss < 100
                if step >= 512:
                    window = [r for r in rows if r["step"] >= step / 4]
                    slopes = [float(np.polyfit(np.log([r["step"] for r in window]), np.log([r[key] for r in window]), 1)[0]) for key in ("train_map", "validation_nll")]
                    if min(slopes) > -.002:
                        reason = "stationary_train_and_validation"
                        break
            if step == config["generator_steps"]: break
            grad = gen.gradient(train_px[:, None] * post)
            opt.update(gen.parameters(), [g + prior * p for g, p in zip(grad, gen.parameters(), strict=True)])
            cpu += time.process_time() - start
        total_steps += step
        models.append(gen)
        records.append({"restart": restart, "rows": rows, "stop_reason": reason})
    chosen = int(np.argmin([r["rows"][-1]["train_map"] for r in records]))
    gen = models[chosen]
    table = generator_tables(gen, config["context_bits"])
    truth_xy = truth.reshape(2 ** config["context_bits"], 2 ** config["target_bits"])
    per_step = 4 * len(gen.h) * gen.layers[0] * gen.visible + 4 * len(gen.x) * len(gen.h) * gen.visible
    info = {"seed": seed, "n": n, "train_n": len(train), "validation_n": len(validation), "unique_observed_values": len(np.unique(real)), "data_sha256": digest(real), "train_sha256": digest(train), "validation_sha256": digest(validation), "selected_restart": chosen, "records": records, "total_steps": total_steps, "leading_dense_flops": total_steps * per_step, "training_cpu_seconds": cpu, "compiled_prediction_kl": categorical_kl(truth_xy, table["model_prediction"]), "config": config}
    np.savez_compressed(destination / "generator.npz", w=gen.weights[0], b=gen.biases[0], real=real, train=train, validation=validation)
    save_json(destination / "generator.json", info)
    return info


def recognition_gradients(rec, x, context_indices, target_indices, table, arm):
    q, logq, hidden = rec.forward(x)
    if arm in ("wake", "sleep"):
        residual = logq - table["logpost"][context_indices]
        delta = q * (residual - np.sum(q * residual, axis=1, keepdims=True))
    elif arm == "forward": delta = q - table["post"][context_indices]
    else:
        assert arm == "predictive"
        responsibilities = q * table["pyh"][:, target_indices].T
        delta = q - responsibilities / responsibilities.sum(axis=1, keepdims=True)
    return rec.backward(hidden, delta / len(x))


def draw_dreams(gen, rng, batch, context_bits):
    latent = rng.integers(len(gen.h), size=batch)
    probabilities = sigmoid(gen.h[latent] @ gen.weights[0][:, :context_bits] + gen.biases[0][:context_bits])
    x = 2. * (rng.random(probabilities.shape) < probabilities) - 1
    return x, latent


def measure(rec, table, truth_xy, train, validation, target_count):
    q, logq, _ = rec.forward(table["context"])
    prediction = q @ table["pyh"]
    train_c, train_y = train // target_count, train % target_count
    val_c, val_y = validation // target_count, validation % target_count
    reverse = np.sum(q * (logq - table["logpost"]), axis=1)
    design = np.column_stack((table["context"], np.ones(len(table["context"]))))
    coefficient = design.T @ (logq - logq.mean(axis=1, keepdims=True)) / len(design)
    target = table["coefficient"]
    denominator = np.linalg.norm(coefficient) * np.linalg.norm(target)
    return {"test_prediction_kl": categorical_kl(truth_xy, prediction), "test_prediction_nll": float(-np.sum(truth_xy * np.log(prediction))), "test_accuracy": float(truth_xy[np.arange(len(truth_xy)), prediction.argmax(axis=1)].sum()), "train_reverse_kl": float(max(0., reverse[train_c].mean())), "train_prediction_nll": float(-np.log(prediction[train_c, train_y]).mean()), "validation_prediction_nll": float(-np.log(prediction[val_c, val_y]).mean()), "own_posterior_kl": float(np.sum(truth_xy.sum(axis=1)[:, None] * table["post"] * (table["logpost"] - logq))), "coefficient_mse": float(np.mean((coefficient - target) ** 2)), "weight_overlap": float(np.sum(coefficient * target) / denominator)}


def run_recognizer(config, seed, n, arm, lr, destination, resume):
    source = destination / f"s{seed}_n{n}"
    folder = source / f"{arm}_lr{lr:g}"
    folder.mkdir(parents=True, exist_ok=True)
    if resume and (folder / "run.json").exists():
        complete = json.loads((folder / "run.json").read_text())
        if complete["stop_reason"] != "running": return {"folder": str(folder), "final": complete["rows"][-1], "stop_reason": complete["stop_reason"]}
    gen_info = json.loads((source / "generator.json").read_text())
    saved = np.load(source / "generator.npz")
    gen = BeliefNetwork([config["latent_bits"]], config["context_bits"] + config["target_bits"], np.random.default_rng(0), 1.)
    gen.weights[0][:] = saved["w"]; gen.biases[0][:] = saved["b"]
    train, validation = saved["train"], saved["validation"]
    table = generator_tables(gen, config["context_bits"])
    truth_xy = make_world(config).reshape(len(table["context"]), 2 ** config["target_bits"])
    recognizer_type = {"categorical": Recognizer, "factorized": FactorizedRecognizer}[config["recognition_family"]]
    rec = recognizer_type(config["context_bits"], len(gen.h), 1, 1, np.random.default_rng(seed + 300000), 0.)
    opt = Optimizer(rec.parameters(), "adam", lr)
    real_rng, dream_rng = np.random.default_rng(seed + 400000), np.random.default_rng(seed + 500000)
    b, c, k, h = config["batch"], config["context_bits"], len(gen.h), config["latent_bits"]
    wake_cost = 4 * b * sum(w.size for w in rec.weights) + (4 * b * h * k if config["recognition_family"] == "factorized" else 0)
    unit = wake_cost + (wake_cost + 2 * b * h * c if arm == "sleep" else 2 * b * k if arm == "predictive" else 0)
    steps = config["budget"] * wake_cost // unit
    checkpoints = set(np.rint(np.geomspace(1, steps, config["eval_points"])).astype(int)) | {0, steps}
    checkpoints |= {int(budget * wake_cost // unit) for budget in config["report_budgets"]}
    rows, cpu, start_step, reason = [], 0., 0, "running"
    load_dotenv(Path.home() / ".env")
    local = Path(os.environ["FAST_CACHE_DIR"]) / "finite-data-wake-sleep" / destination.name / source.name / folder.name
    local.mkdir(parents=True, exist_ok=True)
    if resume and (folder / "checkpoint.json").exists():
        state = json.loads((folder / "checkpoint.json").read_text())
        arrays = np.load(folder / "checkpoint.npz")
        for group, parameters in (("p", rec.parameters()), ("m", opt.m), ("v", opt.v)):
            for i, parameter in enumerate(parameters): parameter[:] = arrays[f"{group}{i}"]
        opt.step = start_step = state["step"]
        real_rng.bit_generator.state, dream_rng.bit_generator.state = state["real_rng"], state["dream_rng"]
        rows, cpu = state["rows"], state["cpu"]
    metadata = {"config": config, "seed": seed, "n": n, "arm": arm, "lr": lr, "generator_data_sha256": gen_info["data_sha256"], "initialization": "small random affine weights; zero biases; independent of teacher", "costs": {"wake": wake_cost, "per_step": unit, "shared_generator": gen_info["leading_dense_flops"]}, "compiled_prediction_kl": gen_info["compiled_prediction_kl"]}
    for step in range(start_step, steps + 1):
        if step in checkpoints and (not rows or rows[-1]["step"] != step):
            metrics = measure(rec, table, truth_xy, train, validation, 2 ** config["target_bits"])
            assert metrics["own_posterior_kl"] < 100 and all(np.isfinite(v) for v in metrics.values()), (metadata, step, metrics)
            rows.append({"step": step, "recognition_flops": step * unit, "total_training_flops": step * unit + gen_info["leading_dense_flops"], "wake_equivalent": step * unit / wake_cost, "real_presentations": step * b, "dream_samples": step * b if arm == "sleep" else 0, "training_cpu_seconds": cpu, **metrics})
            arrays = {f"{group}{i}": p for group, ps in (("p", rec.parameters()), ("m", opt.m), ("v", opt.v)) for i, p in enumerate(ps)}
            np.savez(local / "checkpoint.npz", **arrays)
            shutil.copyfile(local / "checkpoint.npz", folder / "checkpoint.npz")
            save_json(folder / "checkpoint.json", {"step": step, "rows": rows, "cpu": cpu, "real_rng": real_rng.bit_generator.state, "dream_rng": dream_rng.bit_generator.state, "schedule": "base_lr / sqrt(1 + wake_equivalent / decay_scale)", "data_position": "replacement sampling; full generator states above"})
            save_json(folder / "run.json", {**metadata, "rows": rows, "stop_reason": reason})
            if step >= 512:
                window = [r for r in rows if r["step"] >= step / 4]
                key = "train_prediction_nll" if arm == "predictive" else "train_reverse_kl"
                slopes = [float(np.polyfit(np.log([r["step"] for r in window]), np.log(np.maximum([r[m] for r in window], 1e-15)), 1)[0]) for m in (key, "validation_prediction_nll")]
                if min(slopes) > -.02:
                    reason = "stationary_train_and_validation"
                    break
        if step == steps: break
        start = time.process_time()
        index = train[real_rng.integers(len(train), size=b)]
        context_indices, target_indices = index // 2 ** config["target_bits"], index % 2 ** config["target_bits"]
        gr = recognition_gradients(rec, table["context"][context_indices], context_indices, target_indices, table, arm)
        if arm == "sleep":
            dreams, latent = draw_dreams(gen, dream_rng, b, c)
            q, _, hidden = rec.forward(dreams)
            q[np.arange(b), latent] -= 1
            extra = rec.backward(hidden, q / b)
            gr = [a + extra_b for a, extra_b in zip(gr, extra, strict=True)]
        opt.lr = lr / np.sqrt(1 + step * unit / wake_cost / config["decay_scale"])
        opt.update(rec.parameters(), gr)
        cpu += time.process_time() - start
    if reason == "running": reason = "budget"
    save_json(folder / "run.json", {**metadata, "rows": rows, "stop_reason": reason})
    return {"folder": str(folder), "final": rows[-1], "stop_reason": reason}


def audit():
    config = {"latent_bits": 3, "context_bits": 3, "target_bits": 2, "world_seed": 121, "strength": 2., "maximum_data": 128}
    gen = BeliefNetwork([3], 5, np.random.default_rng(122), 1.)
    table = generator_tables(gen, 3)
    xindex = np.arange(8); yindex = np.arange(8) % 4
    errors = {}
    for family, constructor in (("categorical", Recognizer), ("factorized", FactorizedRecognizer)):
        rec = constructor(3, 8, 1, 1, np.random.default_rng(123), 0.)
        for arm in ("wake", "forward", "predictive"):
            gradients = recognition_gradients(rec, table["context"], xindex, yindex, table, arm)
            def objective():
                q, logq, _ = rec.forward(table["context"])
                if arm == "wake": return np.sum(q * (logq - table["logpost"])) / len(xindex)
                if arm == "forward": return -np.sum(table["post"] * logq) / len(xindex)
                return -np.log((q @ table["pyh"])[xindex, yindex]).mean()
            deviations = []
            for p, gradient in zip(rec.parameters(), gradients, strict=True):
                for index in np.ndindex(p.shape):
                    old = p[index]
                    p[index] = old + 1e-5; plus = objective()
                    p[index] = old - 1e-5; minus = objective()
                    p[index] = old
                    deviations.append(abs((plus - minus) / 2e-5 - gradient[index]))
            errors[f"{family}_{arm}"] = max(deviations)
    assert max(errors.values()) < 1e-8
    compiled, _ = softmax(np.column_stack((table["context"], np.ones(8))) @ table["coefficient"])
    compiled_error = float(np.max(np.abs(compiled - table["post"])))
    assert compiled_error < 1e-12
    joint = gen.distribution()[0].sum(axis=1).reshape(8, 4)
    assert np.allclose(joint / joint.sum(axis=1, keepdims=True), table["model_prediction"])
    small, _, _ = data(config, 12, 32)
    large, _, _ = data(config, 12, 64)
    assert np.array_equal(small, large[:32])
    dreams, latent = draw_dreams(gen, np.random.default_rng(124), 200000, 3)
    empirical_cross = gen.h[latent].T @ dreams / len(dreams)
    theoretical_cross = gen.h.T @ (2 * sigmoid(gen.h @ gen.weights[0][:, :3] + gen.biases[0][:3]) - 1) / len(gen.h)
    dream_error = float(np.max(np.abs(empirical_cross - theoretical_cross)))
    assert dream_error < .015
    return {"recognition_gradient_max_errors": errors, "compiled_posterior_error": compiled_error, "dream_cross_moment_max_error": dream_error, "nested_data_verified": True}


def execute(task):
    kind, args = task
    return fitted_generator(*args) if kind == "generator" else run_recognizer(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--generators-only", action="store_true")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(), indent=2)); return
    protocol = json.loads(args.protocol.read_text())
    destination = ROOT / protocol["output"]
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("finite_data_wake_sleep.py", "deep_wake_sleep.py"):
        target = destination / filename
        if args.resume and target.exists(): assert target.read_bytes() == (ROOT / "experiments" / filename).read_bytes()
        shutil.copyfile(ROOT / "experiments" / filename, target)
    if args.resume and (destination / "protocol.json").exists(): assert json.loads((destination / "protocol.json").read_text()) == protocol
    save_json(destination / "protocol.json", protocol)
    tasks = [("generator", (protocol["config"], seed, n, destination / f"s{seed}_n{n}", args.resume)) for seed in protocol["seeds"] for n in protocol["data_sizes"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in tqdm(as_completed([pool.submit(execute, task) for task in tasks]), total=len(tasks), desc="Fit finite-data generators"):
            result = future.result()
            print(json.dumps({k: result[k] for k in ("seed", "n", "compiled_prediction_kl", "total_steps", "training_cpu_seconds")}), flush=True)
    if args.generators_only: return
    tasks = [("recognizer", (protocol["config"], seed, n, arm, lr, destination, args.resume)) for seed in protocol["seeds"] for n in protocol["data_sizes"] for arm in ARMS for lr in protocol["rates"][arm]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in tqdm(as_completed([pool.submit(execute, task) for task in tasks]), total=len(tasks), desc="Train paired recognizers"):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__": main()
