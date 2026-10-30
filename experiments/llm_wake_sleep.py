"""Two conditional LLM adapters form a finite, exactly auditable Helmholtz model."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
RULES = ["add 1", "add 2", "add 3", "subtract 1", "subtract 2", "subtract 3", "multiply by 2", "multiply by 3"]
LETTERS = list("ABCDEFGH")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def truth(z, x):
    # Called only by the real-observation provider and held-out evaluator.
    return ([x + 1, x + 2, x + 3, x - 1, x - 2, x - 3, 2 * x, 3 * x][z]) % 10


def data():
    contexts = list(itertools.permutations(range(10), 2))
    np.random.default_rng(9600).shuffle(contexts)
    def rows(contexts):
        return [{"inputs": list(c), "outputs": [truth(z, x) for x in c], "latent": z} for c in contexts for z in range(8)]
    return rows(contexts[:60]), rows(contexts[60:])


def p_prompt(z, x):
    return f"Apply this rule to the input: {RULES[z]}. Keep only the last decimal digit of the result (modulo 10).\nInput: {x}\nOutput (one digit):"


def q_prompt(row):
    choices = "\n".join(f"{letter}: {rule}" for letter, rule in zip(LETTERS, RULES, strict=True))
    observations = "\n".join(f"Input {x} -> Output {y}" for x, y in zip(row["inputs"], row["outputs"], strict=True))
    return f"A hidden rule transforms numbers, keeping only the last decimal digit (modulo 10). Infer the rule from the examples.\nRules:\n{choices}\nExamples:\n{observations}\nWhich rule? Answer with one letter A-H:"


def save(path, value): path.write_text(json.dumps(value, indent=2))


def initialize(seed):
    load_dotenv(Path.home() / ".env")
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed); np.random.seed(seed)
    cache = Path(os.environ["FAST_CACHE_DIR"]) / "hf-wake-sleep"
    model = AutoModelForCausalLM.from_pretrained(MODEL, token=os.environ["HF_TOKEN"], cache_dir=cache, dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=os.environ["HF_TOKEN"], cache_dir=cache)
    tokenizer.padding_side = "left"
    return Model(model, tokenizer)


class Model:
    def __init__(self, model, tokenizer):
        import torch
        self.model, self.tokenizer, self.active = model, tokenizer, "p"
        for p in model.parameters(): p.requires_grad_(False)
        self.adapters = []
        owner = self
        class LoRA(torch.nn.Module):
            def __init__(self, linear):
                super().__init__(); self.linear = linear
                self.a = torch.nn.ParameterDict({key: torch.nn.Parameter(torch.randn(4, linear.in_features, device="cuda", dtype=torch.float32) * .02) for key in ("p", "q")})
                self.b = torch.nn.ParameterDict({key: torch.nn.Parameter(torch.zeros(linear.out_features, 4, device="cuda", dtype=torch.float32)) for key in ("p", "q")})
            def forward(self, x):
                key = owner.active
                extra = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a[key]), self.b[key])
                return self.linear(x) + extra.to(x.dtype)
        for layer in model.model.layers:
            for key in ("q_proj", "v_proj"):
                module = LoRA(getattr(layer.self_attn, key)); setattr(layer.self_attn, key, module); self.adapters.append(module)
        self.ids = {kind: torch.tensor([tokenizer.encode(s, add_special_tokens=False)[0] for s in symbols], device="cuda") for kind, symbols in (("p", list("0123456789")), ("q", [" " + letter for letter in LETTERS]))}
        for s in list("0123456789") + [" " + letter for letter in LETTERS]: assert len(tokenizer.encode(s, add_special_tokens=False)) == 1
        self.base_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        self.usage = {"p_forward_tokens": 0, "q_forward_tokens": 0, "p_backward_tokens": 0, "q_backward_tokens": 0, "generated_digits": 0}

    def parameters(self, key): return [p for layer in self.adapters for p in (layer.a[key], layer.b[key])]

    def state(self): return {key: [p.detach().cpu().clone() for p in self.parameters(key)] for key in ("p", "q")}

    def restore(self, state):
        import torch
        with torch.no_grad():
            for key in ("p", "q"):
                for p, value in zip(self.parameters(key), state[key], strict=True): p.copy_(value)

    def logprobs(self, prompts, key, gradients, account):
        import torch
        self.active = key
        prefix = "" if key == "p" else "Answer:"
        texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) + prefix for p in prompts]
        encoding = self.tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
        with torch.set_grad_enabled(gradients):
            hidden = self.model.model(**encoding, use_cache=False).last_hidden_state
            last = hidden[:, -1]
            logits = torch.nn.functional.linear(last, self.model.lm_head.weight[self.ids[key]]).float()
            result = torch.log_softmax(logits, dim=-1)
        if account:
            self.usage[f"{key}_forward_tokens"] += encoding.input_ids.numel()
            if gradients: self.usage[f"{key}_backward_tokens"] += encoding.input_ids.numel()
        return result

    def generator(self, rows, gradients, account):
        prompts = [p_prompt(z, x) for row in rows for z in range(8) for x in row["inputs"]]
        return self.logprobs(prompts, "p", gradients, account).reshape(len(rows), 8, 2, 10)

    def compute(self):
        forward = self.usage["p_forward_tokens"] + self.usage["q_forward_tokens"]
        backward = self.usage["p_backward_tokens"] + self.usage["q_backward_tokens"]
        # Frozen base still propagates activation gradients; no base weight gradients.
        return 2 * self.base_parameters * forward + 2 * self.base_parameters * backward


def joint_logprob(logp, rows):
    import torch
    target = torch.tensor([row["outputs"] for row in rows], device="cuda")
    return logp.gather(-1, target[:, None, :, None].expand(-1, 8, -1, 1)).squeeze(-1).sum(axis=-1) - np.log(8)


def evaluate(model, rows, size=8):
    import torch
    records = []
    with torch.no_grad():
        table = model.logprobs([p_prompt(z, x) for z in range(8) for x in range(10)], "p", False, False).reshape(8, 10, 10)
        for start in range(0, len(rows), size):
            batch = rows[start:start + size]
            inputs = torch.tensor([row["inputs"] for row in batch], device="cuda")
            logp = table[:, inputs].permute(1, 0, 2, 3)
            logjoint = joint_logprob(logp, batch)
            logpost = torch.log_softmax(logjoint, dim=-1)
            post = logpost.exp()
            logq = model.logprobs([q_prompt(row) for row in batch], "q", False, False)
            q = logq.exp()
            for row, lp, lj, lpost, posterior, lq, prob in zip(batch, logp.cpu().numpy(), logjoint.cpu().numpy(), logpost.cpu().numpy(), post.cpu().numpy(), logq.cpu().numpy(), q.cpu().numpy(), strict=True):
                compatible = np.array([all(truth(z, x) == y for x, y in zip(row["inputs"], row["outputs"], strict=True)) for z in range(8)])
                truepost = compatible / compatible.sum()
                kl = np.sum(truepost[compatible] * (np.log(truepost[compatible]) - lq[compatible]))
                z = row["latent"]
                gen_correct = np.mean([np.exp(lp[z, k, y]) for k, y in enumerate(row["outputs"])])
                records.append({**row, "q": prob.tolist(), "model_posterior": posterior.tolist(), "posterior_kl": float(kl),
                                "compatible_mass": float(prob[compatible].sum()), "accuracy": float(compatible[prob.argmax()]),
                                "generator_correct_probability": float(gen_correct), "marginal_nll": float(-np.logaddexp.reduce(lj)),
                                "recognition_reverse_kl": float(np.sum(prob * (lq - lpost))),
                                "inference_forward_kl": float(np.sum(posterior * (lpost - lq))),
                                "model_posterior_compatible_mass": float(posterior[compatible].sum()),
                                "model_posterior_accuracy": float(compatible[posterior.argmax()]),
                                "model_posterior_true_kl": float(np.sum(truepost[compatible] * (np.log(truepost[compatible]) - lpost[compatible])))})
    metrics = ("posterior_kl", "compatible_mass", "accuracy", "generator_correct_probability", "marginal_nll", "recognition_reverse_kl",
               "inference_forward_kl", "model_posterior_compatible_mass", "model_posterior_accuracy", "model_posterior_true_kl")
    return {"records": records, **{key: float(np.mean([r[key] for r in records])) for key in metrics}}


def checkpoint(model, opts, step, rng, rows, path):
    import torch
    torch.save({"adapters": model.state(), "optimizers": {k: o.state_dict() for k, o in opts.items()}, "step": step,
                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(), "usage": model.usage, "rows": rows, "scheduler": "constant", "data_order": "IID stream; full RNG state saved"}, path)


def train(model, protocol, output, resume):
    import torch
    trainrows, evaluation = data()
    # Training receives no source latent field.
    trainrows = [{k: row[k] for k in ("inputs", "outputs")} for row in trainrows]
    seed, arm = protocol["seed"], protocol["arm"]
    opts = {key: torch.optim.AdamW(model.parameters(key), lr=protocol[f"lr_{key}"], weight_decay=0) for key in ("p", "q")}
    rng = np.random.default_rng(seed + 100000)
    history, first = [], 0
    if resume:
        state = torch.load(resume, map_location="cuda", weights_only=False)
        model.restore(state["adapters"])
        for key in opts: opts[key].load_state_dict(state["optimizers"][key])
        rng.bit_generator.state = state["numpy_rng"]; torch.set_rng_state(state["torch_rng"].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]]); random.setstate(state["python_rng"])
        model.usage, history, first = state["usage"], state["rows"], state["step"]
    initial = evaluate(model, evaluation)
    if not resume: save(output / "eval_0.json", initial)
    checks = set(np.rint(np.geomspace(1, protocol["steps"], protocol["eval_points"])).astype(int)) | {0, first, protocol["steps"]}
    local = Path(os.environ["FAST_CACHE_DIR"]) / "wake-sleep-checkpoints" / output.name; local.mkdir(parents=True, exist_ok=True)
    persistent = Path(os.environ["CACHE_DIR"]) / "generative-learning" / "wake-sleep-checkpoints" / output.name; persistent.mkdir(parents=True, exist_ok=True)
    uploads = ThreadPoolExecutor(max_workers=1); transfers = []
    losses, cpu, reason = [], (history[-1]["training_seconds"] if resume else 0.), "budget"
    for step in tqdm(range(first, protocol["steps"] + 1), desc=output.name):
        if step in checks:
            metric = initial if step == first else evaluate(model, evaluation)
            record = {"step": step, "flops": model.compute(), "usage": dict(model.usage), "training_seconds": cpu,
                      "train_loss": float(np.mean(losses)) if losses else None, **{k: v for k, v in metric.items() if k != "records"}}
            history.append(record); save(output / f"eval_{step}.json", metric)
            save(output / "run.json", {"protocol": protocol, "rows": history, "stop_reason": reason})
            assert np.isfinite(record["posterior_kl"]), "Nonfinite eval; redesign required."
            assert record["posterior_kl"] < 2 * initial["posterior_kl"] + 1., "Clearly degrading posterior; redesign required."
            if step >= 64:
                window = [r for r in history if r["step"] >= step / 4 and r["train_loss"] is not None]
                if len(window) >= 3:
                    slopes = [np.polyfit(np.log([r["step"] for r in window]), np.log([r[key] for r in window]), 1)[0] for key in ("train_loss", "posterior_kl")]
                    if min(slopes) > -.02: reason = "stationary_loss_and_eval"
            losses = []
        if step % 32 == 0 or step == protocol["steps"] or reason != "budget":
            path = local / f"step_{step}.pt"
            checkpoint(model, opts, step, rng, history, path)
            transfers.append(uploads.submit(shutil.copy2, path, persistent / path.name))
            save(output / "resume.json", {"local": str(path), "persistent": str(persistent / path.name), "step": step})
        if step == protocol["steps"] or reason != "budget": break
        batch = [trainrows[i] for i in rng.integers(len(trainrows), size=protocol["batch"])]
        torch.cuda.synchronize(); start = time.monotonic()
        for opt in opts.values(): opt.zero_grad(set_to_none=True)
        logp = model.generator(batch, protocol["coupled"], True)
        logjoint = joint_logprob(logp, batch)
        logq = model.logprobs([q_prompt(row) for row in batch], "q", True, True)
        q = logq.exp()
        loss_q = (q * (logq - logjoint.detach())).sum(axis=-1).mean()
        if arm == "forward": loss_q = -(torch.softmax(logjoint.detach(), dim=-1) * logq).sum(axis=-1).mean()
        if arm == "sleep":
            latent = torch.tensor(rng.integers(8, size=len(batch)), device="cuda")
            probs = logp.detach().exp()[torch.arange(len(batch), device="cuda"), latent]
            generated = torch.multinomial(probs.reshape(-1, 10), 1).reshape(len(batch), 2)
            dreams = [{"inputs": row["inputs"], "outputs": ys} for row, ys in zip(batch, generated.cpu().tolist(), strict=True)]
            dreamlogq = model.logprobs([q_prompt(row) for row in dreams], "q", True, True)
            loss_q = loss_q - dreamlogq[torch.arange(len(batch), device="cuda"), latent].mean()
            model.usage["generated_digits"] += generated.numel()
        assert torch.isfinite(loss_q); loss_q.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters("q"), 1., error_if_nonfinite=True); opts["q"].step()
        if protocol["coupled"]:
            loss_p = -(q.detach() * logjoint).sum(axis=-1).mean()
            assert torch.isfinite(loss_p); loss_p.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters("p"), 1., error_if_nonfinite=True); opts["p"].step()
        torch.cuda.synchronize(); cpu += time.monotonic() - start; losses.append(float(loss_q.detach()))
    for transfer in transfers: transfer.result()
    uploads.shutdown(); save(output / "run.json", {"protocol": protocol, "rows": history, "stop_reason": reason})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--arm", choices=("elbo", "sleep", "forward"), default="elbo"); parser.add_argument("--seed", type=int, default=9601)
    parser.add_argument("--steps", type=int, default=128); parser.add_argument("--batch", type=int, default=8); parser.add_argument("--lr-q", type=float, default=1e-4)
    parser.add_argument("--lr-p", type=float, default=1e-5); parser.add_argument("--coupled", action="store_true"); parser.add_argument("--resume")
    parser.add_argument("--eval-points", type=int, default=12)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=bool(args.resume))
    model = initialize(args.seed)
    protocol = {**vars(args), "model": MODEL, "batch_id": "w8s",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "prior": "uniform over eight rules",
                "policy": "Temperature 1, normalized digit/letter support; no checker, rejection, or true latent in training.",
                "compute": "Forward + activation-backward dense FLOP approximation; counts enumeration, dream training, padded tokens. Evaluation excluded.",
                "scope": "Exploratory adapter training of a finite conditional latent-variable model; no exponent claim."}
    save(output / "protocol.json", protocol); (output / "runtime.py").write_bytes(Path(__file__).read_bytes())
    if args.calibrate:
        training, evaluation = data()
        result = evaluate(model, evaluation)
        save(output / "evaluation.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
        for row in evaluation[::60]:
            print(q_prompt(row)); print(json.dumps(row)); print(json.dumps(result["records"][evaluation.index(row)]))
    else: train(model, protocol, output, args.resume)


if __name__ == "__main__": main()
