"""Float64 GPU continuation of the fixed-generator, population size experiment.

The fixed joint is cached. Reference FLOPs reproduce the earlier accounting;
cached_dense_flops counts the matrix products actually executed here. Neither
includes elementwise kernels or evaluation. Both axes are retained explicitly.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

from dotenv import load_dotenv
import numpy as np
import torch
from tqdm.auto import tqdm

import deep_wake_sleep as reference


ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def binary_states(bits, device):
    indices = torch.arange(2**bits, device=device)
    shifts = torch.arange(bits - 1, -1, -1, device=device)
    return (((indices[:, None] >> shifts) & 1) * 2 - 1).to(DTYPE)


class PopulationLearner:
    def __init__(self, config, device="cuda"):
        assert not config["coupled"] and not config["sampled"] and config["recognition_depth"] == 1
        assert config["optimizer"] in ("adam", "adam_eps12")
        self.config, self.device = config, device
        rng = np.random.default_rng(config["seed"])
        layers, visible = config["layers"], config["visible"]
        self.x = binary_states(visible, device)
        h = binary_states(sum(layers), device)
        blocks = torch.split(h, layers, dim=1)
        dims = layers + [visible]
        self.gweights = [torch.tensor(rng.normal(size=(a, b)) * config["strength"] / np.sqrt(a), dtype=DTYPE, device=device) for a, b in zip(dims[:-1], dims[1:], strict=True)]
        self.gbiases = [torch.tensor(rng.normal(size=b) * .15, dtype=DTYPE, device=device) for b in dims[1:]]
        logprior = torch.full((len(h),), -layers[0] * np.log(2), dtype=DTYPE, device=device)
        for parent, child, w, b in zip(blocks[:-1], blocks[1:], self.gweights[:-1], self.gbiases[:-1], strict=True):
            logprior -= torch.nn.functional.softplus(-child * (parent @ w + b)).sum(1)
        a = blocks[-1] @ self.gweights[-1] + self.gbiases[-1]
        self.target_w = .5 * a.T
        self.target_b = logprior - torch.logaddexp(-a / 2, a / 2).sum(1)
        logjoint = self.x @ self.target_w + self.target_b
        self.joint = logjoint.exp()
        assert abs(self.joint.sum().item() - 1) < 1e-10
        self.px = self.joint.sum(1)
        self.logpost = logjoint - torch.logsumexp(logjoint, 1, keepdim=True)
        self.post = self.logpost.exp()
        self.w = torch.tensor(rng.normal(size=(visible, len(h))) / np.sqrt(visible) * .05, dtype=DTYPE, device=device)
        self.b = torch.tensor(rng.normal(size=len(h)) * config["neglect"], dtype=DTYPE, device=device)
        self.mw, self.vw = torch.zeros_like(self.w), torch.zeros_like(self.w)
        self.mb, self.vb = torch.zeros_like(self.b), torch.zeros_like(self.b)
        self.counter = torch.zeros((), dtype=DTYPE, device=device)
        self.epsilon = 1e-12 if config["optimizer"] == "adam_eps12" else 1e-8
        self.graph = None
        nx, nz = self.post.shape
        rec = visible * nz
        gen_weights = sum(w.numel() for w in self.gweights)
        wake = 6 * nx * rec + 4 * nx * nz * visible + 6 * nz * gen_weights
        sleep = 6 * nx * rec + 2 * nz * gen_weights
        self.costs = {"wake": wake, "sleep": sleep,
                      "reference_unit": wake + (sleep if config["arm"] == "sleep" else 0),
                      "cached_dense_unit": (6 if config["arm"] == "sleep" else 4) * nx * rec,
                      "accounting": "Reference FLOPs use the original uncached NumPy algorithm; cached dense FLOPs count executed recognition matrix products, excluding elementwise kernels, fixed-generator setup and eval."}

    def state_tensors(self): return {k: getattr(self, k) for k in ("w", "b", "mw", "vw", "mb", "vb", "counter")}

    def state_dict(self): return {k: v.detach().cpu().clone() for k, v in self.state_tensors().items()}

    def load_state_dict(self, state):
        for k, v in self.state_tensors().items(): v.copy_(state[k])

    def gradients(self):
        logq = torch.log_softmax(self.x @ self.w + self.b, 1)
        q = logq.exp()
        residual = logq - self.logpost
        delta = self.px[:, None] * q * (residual - (q * residual).sum(1, keepdim=True))
        if self.config["arm"] == "forward": delta = self.px[:, None] * (q - self.post)
        gw, gb = self.x.T @ delta, delta.sum(0)
        if self.config["arm"] == "sleep":
            extra = self.px[:, None] * (q - self.post)
            gw = gw + self.x.T @ extra
            gb = gb + extra.sum(0)
        return gw, gb

    def update(self):
        gradients = self.gradients()
        self.counter.add_(1)
        correction1, correction2 = 1 - .9**self.counter, 1 - .999**self.counter
        for p, g, m, v in zip((self.w, self.b), gradients, (self.mw, self.mb), (self.vw, self.vb), strict=True):
            m.mul_(.9).add_(g, alpha=.1)
            v.mul_(.999).addcmul_(g, g, value=.001)
            p.sub_(self.config["lr_r"] * (m / correction1) / ((v / correction2).sqrt() + self.epsilon))

    def capture(self):
        initial = self.state_dict()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): self.update()
        torch.cuda.current_stream().wait_stream(stream)
        self.load_state_dict(initial)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self.update()
        self.load_state_dict(initial)
        torch.cuda.synchronize()

    def metrics(self):
        logq = torch.log_softmax(self.x @ self.w + self.b, 1)
        q = logq.exp()
        forward = (self.joint * (self.logpost - logq)).sum()
        reverse = (self.px[:, None] * q * (logq - self.logpost)).sum()
        qw, qb = self.w - self.w.mean(1, keepdim=True), self.b - self.b.mean()
        pw, pb = self.target_w - self.target_w.mean(1, keepdim=True), self.target_b - self.target_b.mean()
        dot = (qw * pw).sum() + (qb * pb).sum()
        norm = ((qw.square().sum() + qb.square().sum()) * (pw.square().sum() + pb.square().sum())).sqrt()
        return {"posterior_kl": forward.item(), "reverse_kl": reverse.item(), "elbo_excess": reverse.item(), "marginal_kl": 0.,
                "posterior_brier": (self.px[:, None] * (q - self.post).square()).sum().item(),
                "neglected_mass": (self.joint * (q < .1 * self.post)).sum().item(),
                "posterior_map_agreement": (self.px * (q.argmax(1) == self.post.argmax(1))).sum().item(),
                "recognition_coefficient_mse": (((qw - pw).square().sum() + (qb - pb).square().sum()) / (qw.numel() + qb.numel())).item(),
                "recognition_weight_overlap": (dot / norm).item(), "weight_mse": 0.}


def save_checkpoint(learner, output, rows, training_seconds, pool, previous):
    if previous is not None: previous.result()
    step = int(learner.counter.item())
    local = Path(os.environ["FAST_CACHE_DIR"]) / "wake-sleep-large-checkpoints" / output.parent.name / output.name
    persistent = Path(os.environ["CACHE_DIR"]) / "generative-learning/wake-sleep-large-checkpoints" / output.parent.name / output.name
    local.mkdir(parents=True, exist_ok=True); persistent.mkdir(parents=True, exist_ok=True)
    name = f"step_{step}.pt"
    state = {"learner": learner.state_dict(), "config": learner.config, "step": step, "rows": rows, "training_seconds": training_seconds,
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
             "scheduler": {"kind": "constant", "lr": learner.config["lr_r"]}, "data_order": "exact population, no sampled data stream"}
    torch.save(state, local / name)
    def transfer():
        shutil.copy2(local / name, persistent / name)
        (output / "resume.json").write_text(json.dumps({"step": step, "local": str(local / name), "persistent": str(persistent / name)}, indent=2))
    return pool.submit(transfer)


def run(config, resume):
    output = ROOT / config["output"] / config["name"]
    resume_run = resume and output.exists()
    if resume_run:
        previous = json.loads((output / "run.json").read_text())
        if previous["stop_reason"] != "running": return {"name": config["name"], "status": "already complete", "stop": previous["stop_reason"]}
    output.mkdir(parents=True, exist_ok=resume)
    learner = PopulationLearner(config)
    rows, training_seconds, first = [], 0., 0
    if resume_run:
        checkpoint = json.loads((output / "resume.json").read_text())
        saved = torch.load(checkpoint["persistent"], map_location="cpu", weights_only=False)
        assert saved["config"] == config
        learner.load_state_dict(saved["learner"])
        rows, training_seconds, first = saved["rows"], saved["training_seconds"], saved["step"]
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
    learner.capture()
    cost = learner.costs
    cap = int(config["budget"] * cost["wake"] / cost["reference_unit"])
    checkpoints = set(np.rint(np.geomspace(1, cap, config["eval_points"])).astype(int)) | {0, cap, first}
    checkpoints |= set(range(config["eval_interval"], cap, config["eval_interval"]))
    checkpoints |= set(config["extra_eval_steps"])
    reason, future = "running", None
    started = time.monotonic()
    last_saved = started
    progress = tqdm(total=cap, initial=first, desc=config["name"], mininterval=15)
    with ThreadPoolExecutor(max_workers=1) as pool:
        for step in range(first, cap + 1):
            if step in checkpoints:
                torch.cuda.synchronize()
                training_seconds += time.monotonic() - started
                metric = learner.metrics()
                assert np.isfinite(list(metric.values())).all() and 0 <= metric["posterior_kl"] < 200, metric
                assert int(learner.counter.item()) == step
                row = {"step": step, "compute": step * cost["reference_unit"], "cached_dense_flops": step * cost["cached_dense_unit"],
                       "wake_equivalent": step * cost["reference_unit"] / cost["wake"], "training_seconds": training_seconds, "dreams": 0, **metric}
                if rows and rows[-1]["step"] == step:
                    assert abs(rows[-1]["posterior_kl"] - metric["posterior_kl"]) < 1e-10
                    rows[-1] = row
                else: rows.append(row)
                if metric["posterior_kl"] <= config["accuracy_stop"]: reason = "accuracy_target"
                elif step >= 256:
                    window = [r for r in rows if step / 4 <= r["step"] <= step]
                    slopes = [np.polyfit(np.log([r["step"] for r in window]), np.log([max(r[key], 1e-15) for r in window]), 1)[0] for key in ("elbo_excess", "posterior_kl")]
                    if min(slopes) > -.02: reason = "stationary_loss_and_eval"
                if reason == "running" and step == cap: reason = "budget"
                if reason == "running" and training_seconds >= config["wall_limit_seconds"]: reason = "time_cap_censored"
                (output / "run.json").write_text(json.dumps({"config": config, "costs": cost, "rows": rows, "stop_reason": reason}, indent=2))
                progress.set_postfix(kl=f'{metric["posterior_kl"]:.4g}', refresh=False)
                if reason != "running" or time.monotonic() - last_saved >= config["checkpoint_seconds"] or step == first:
                    future = save_checkpoint(learner, output, rows, training_seconds, pool, future)
                    last_saved = time.monotonic()
                started = time.monotonic()
                if reason != "running": break
            learner.graph.replay()
            progress.update(1)
        future.result()
    progress.close()
    summary = {"name": config["name"], "stop": reason, "step": rows[-1]["step"], "training_seconds": training_seconds,
               "initial_kl": rows[0]["posterior_kl"], "final_kl": rows[-1]["posterior_kl"]}
    print(json.dumps(summary), flush=True)
    return summary


def audit():
    records = []
    for layers in ([4], [2, 2], [2, 2, 2]):
        for arm in ("elbo", "sleep", "forward"):
            config = {"layers": list(layers), "visible": 6, "seed": 9540, "strength": 2., "coupled": False, "sampled": False,
                      "recognition_depth": 1, "optimizer": "adam", "neglect": 3., "lr_r": .1 if arm == "elbo" else .3, "arm": arm}
            learner = PopulationLearner(config)
            rng = np.random.default_rng(config["seed"])
            gen = reference.BeliefNetwork(list(layers), 6, rng, 2.)
            rec = reference.Recognizer(6, len(gen.h), 1, 32, rng, 3.)
            joint, _ = gen.distribution(); px = joint.sum(1)
            assert np.max(abs(learner.joint.cpu().numpy() - joint)) < 1e-13
            _, gr = reference.components(gen, rec, px, arm, rng, 128, False)
            error = max(np.max(abs(a.cpu().numpy() - b)) for a, b in zip(learner.gradients(), gr, strict=True))
            assert error < 1e-12
            old_metric = reference.metrics(gen, rec, joint, gen.weights)
            metric_error = max(abs(v - old_metric[k]) for k, v in learner.metrics().items())
            assert metric_error < 1e-10
            optimizer = reference.Optimizer(rec.parameters(), "adam", config["lr_r"])
            learner.capture()
            for _ in range(100):
                _, gr = reference.components(gen, rec, px, arm, rng, 128, False)
                optimizer.update(rec.parameters(), gr)
                learner.graph.replay()
            torch.cuda.synchronize()
            trajectory_error = max(np.max(abs(a.cpu().numpy() - b)) for a, b in zip((learner.w, learner.b), rec.parameters(), strict=True))
            assert trajectory_error < 1e-9
            saved = learner.state_dict()
            for _ in range(10): learner.graph.replay()
            torch.cuda.synchronize(); continued = learner.state_dict()
            learner.load_state_dict(saved)
            for _ in range(10): learner.graph.replay()
            torch.cuda.synchronize()
            resume_error = max((v - continued[k]).abs().max().item() for k, v in learner.state_dict().items())
            assert resume_error == 0
            records.append({"layers": layers, "arm": arm, "gradient_error": error, "metric_error": metric_error,
                            "100_step_weight_error": trajectory_error, "resume_error": resume_error})
    config["layers"] = [8, 8]; config["arm"] = "elbo"
    learner = PopulationLearner(config); learner.capture()
    started = time.monotonic()
    for _ in range(1000): learner.graph.replay()
    torch.cuda.synchronize()
    benchmark = {"states": 65536, "steps": 1000, "seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    result = {"device": torch.cuda.get_device_name(), "dtype": str(DTYPE), "cases": records, "benchmark": benchmark}
    target = ROOT / "results/deep_wake_sleep/gpu_audit.json"
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--audit", action="store_true"); parser.add_argument("--protocol"); parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    if args.audit: audit(); return
    protocol = json.loads(Path(args.protocol).read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == protocol["runtime_sha256"]
    cases = protocol["cases"][args.shard::args.shards]
    for case in tqdm(cases, desc=f"Large-size shard {args.shard}"):
        run({**protocol["base"], **case}, args.resume)
        torch.cuda.empty_cache()


if __name__ == "__main__": main()
