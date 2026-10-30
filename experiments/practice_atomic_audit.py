"""Exact screens for the atomic-verification addition to tex/README.md.

These are rejection tests, not a successful practice learner. The first checks
validity against data likelihood. The second checks whether a small
parent-conditioned grammar still admits straightforward exact inference.
"""

import argparse
from datetime import datetime
from functools import lru_cache
import hashlib
from itertools import product
import json
from math import comb
import os
from pathlib import Path
import shutil
from time import perf_counter
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm


def catalan(n): return comb(2*n, n)//(n+1)


def sigmoid(a): return 1/(1+np.exp(-a))


def validity_drift(a):
    p = sigmoid(a)
    return -2*p*p*(1-p)


def validity_curve(dt, end):
    a = -np.log(2.)
    history = []
    for step in range(round(end/dt)+1):
        p = sigmoid(a)
        nll = -(2*np.log(p)+4*np.log1p(-p))/3
        history.append((step*dt, p, 1-p*p, nll, nll-np.log(3)))
        k1 = validity_drift(a)
        k2 = validity_drift(a+dt*k1/2)
        k3 = validity_drift(a+dt*k2/2)
        k4 = validity_drift(a+dt*k3)
        a += dt*(k1+2*k2+2*k3+k4)/6
    return np.asarray(history)


def words(length):
    return np.array(list(product(range(2), repeat=length)), dtype=np.int64)


def grammar_inference(x, pi, rules, counts=True):
    """Exact inside/outside; rules[parent, node, left, right], parent=2 at root."""
    n, length = x.shape
    inside = {}
    for i in range(length):
        inside[i, i+1] = np.broadcast_to((x[:, i, None] == np.arange(2))[:, None, :], (n, 3, 2)).astype(float)
    for span in range(2, length+1):
        for i in range(length-span+1):
            j = i+span
            value = np.zeros((n, 3, 2))
            for k in range(i+1, j):
                weight = catalan(k-i-1)*catalan(j-k-1)/catalan(span-1)
                value += weight*np.einsum("pabc,nab,nac->npa", rules, inside[i, k][:, :2], inside[k, j][:, :2])
            inside[i, j] = value
    likelihood = inside[0, length][:, 2] @ pi
    if not counts: return likelihood
    outside = {key: np.zeros_like(value) for key, value in inside.items()}
    outside[0, length][:, 2] = pi[None, :]/likelihood[:, None]
    root_counts = inside[0, length][:, 2]*pi[None, :]/likelihood[:, None]
    rule_counts = np.zeros((n, 3, 2, 2, 2))
    for span in range(length, 1, -1):
        for i in range(length-span+1):
            j = i+span
            for k in range(i+1, j):
                weight = catalan(k-i-1)*catalan(j-k-1)/catalan(span-1)
                left, right, adjoint = inside[i, k][:, :2], inside[k, j][:, :2], outside[i, j]
                rule_counts += weight*np.einsum("npa,pabc,nab,nac->npabc", adjoint, rules, left, right)
                outside[i, k][:, :2] += weight*np.einsum("npa,pabc,nac->nab", adjoint, rules, right)
                outside[k, j][:, :2] += weight*np.einsum("npa,pabc,nab->nac", adjoint, rules, left)
    return likelihood, root_counts, rule_counts


@lru_cache(None)
def trees(length):
    if length == 1: return (None,)
    return tuple((left, right) for split in range(1, length) for left in trees(split) for right in trees(length-split))


def enumerate_grammar(length, pi, rules):
    """Independent full enumeration of tree shapes and all node categories."""
    result = np.zeros(2**length)
    roots = np.zeros((2**length, 2))
    counts = np.zeros((2**length, 3, 2, 2, 2))
    for tree in trees(length):
        nodes, leaves = [], []

        def flatten(node, parent):
            index = len(nodes)
            nodes.append([parent, -1, -1])
            if node is None: leaves.append(index)
            else:
                nodes[index][1] = flatten(node[0], index)
                nodes[index][2] = flatten(node[1], index)
            return index

        flatten(tree, -1)
        categories = words(2*length-1)
        output = categories[:, leaves] @ (2**np.arange(length-1, -1, -1))
        mass = pi[categories[:, 0]]/catalan(length-1)
        events = []
        for index, (parent, left, right) in enumerate(nodes):
            if left == -1: continue
            p = np.full(len(categories), 2) if parent == -1 else categories[:, parent]
            a, b, c = categories[:, index], categories[:, left], categories[:, right]
            events.append((p, a, b, c))
            mass = mass*rules[p, a, b, c]
        np.add.at(result, output, mass)
        np.add.at(roots, (output, categories[:, 0]), mass)
        for p, a, b, c in events: np.add.at(counts, (output, p, a, b, c), mass)
    return result, roots/result[:, None], counts/result[:, None, None, None, None]


def floor_mstep(counts, floor):
    """Exact categorical M-step with a specified per-entry probability floor."""
    original_shape = counts.shape
    rows = counts.reshape(-1, original_shape[-1])
    result = np.empty_like(rows)
    for index, row in enumerate(rows):
        free = np.ones(len(row), dtype=bool)
        p = np.full(len(row), floor)
        for _ in range(len(row)):
            p[free] = (1-floor*np.sum(~free))*row[free]/row[free].sum()
            violating = free & (p < floor)
            if not violating.any(): break
            p[violating] = floor
            free[violating] = False
        assert np.all(p >= floor) and abs(p.sum()-1) < 1e-12
        result[index] = p
    return result.reshape(original_shape)


def softmax(a, axis=-1):
    result = np.exp(a-np.max(a, axis=axis, keepdims=True))
    return result/result.sum(axis=axis, keepdims=True)


def mathematical_audits():
    rng = np.random.default_rng(20103390)
    errors = {"enumeration_probability": 0., "enumeration_counts": 0., "gradient": 0., "normalization": 0., "score_drift": 0.}
    for _ in tqdm(range(12), desc="Exact grammar audits"):
        pi = rng.dirichlet(np.ones(2))
        rules = rng.dirichlet(np.ones(4), size=(3, 2)).reshape(3, 2, 2, 2)
        exact = enumerate_grammar(5, pi, rules)
        dynamic = grammar_inference(words(5), pi, rules)
        errors["enumeration_probability"] = max(errors["enumeration_probability"], float(np.max(abs(exact[0]-dynamic[0]))))
        errors["enumeration_counts"] = max(errors["enumeration_counts"], float(np.max(abs(exact[2]-dynamic[2]))), float(np.max(abs(exact[1]-dynamic[1]))))
        errors["normalization"] = max(errors["normalization"], abs(float(dynamic[0].sum())-1))
        assert np.max(abs(dynamic[2].sum(axis=(1, 2, 3, 4))-4)) < 1e-10
        direction = rng.normal(size=(3, 2, 4))
        direction /= np.linalg.norm(direction)
        logits = np.log(rules.reshape(3, 2, 4))
        e = 1e-5
        upper = grammar_inference(words(5), pi, softmax(logits+e*direction).reshape(3, 2, 2, 2), False)
        lower = grammar_inference(words(5), pi, softmax(logits-e*direction).reshape(3, 2, 2, 2), False)
        numerical = (np.log(upper).mean()-np.log(lower).mean())/(2*e)
        c = dynamic[2].mean(axis=0).reshape(3, 2, 4)
        analytic = np.sum(direction*(c-c.sum(axis=-1, keepdims=True)*rules.reshape(3, 2, 4)))
        errors["gradient"] = max(errors["gradient"], abs(float(analytic-numerical)))
    states = words(2)
    for a in np.linspace(-6, 3, 37):
        p = sigmoid(a)
        mass = (p**states.sum(axis=1))*((1-p)**(2-states.sum(axis=1)))
        reward = 1-states[:, 0]*states[:, 1]
        direct = np.sum(mass*reward*(states.sum(axis=1)-2*p))
        errors["score_drift"] = max(errors["score_drift"], abs(float(direct-validity_drift(a))))
    coarse, fine = validity_curve(.05, 400), validity_curve(.025, 400)
    errors["rk4_step_halving"] = float(np.max(abs(coarse-fine[::2])))
    assert max(errors.values()) < 1e-8
    assert np.all(np.diff(coarse[:, 2]) > 0) and np.all(np.diff(coarse[:, 3]) > 0)
    return errors, coarse


def fit_one(seed, max_steps):
    """Illustrative direct fitting, not a comparison against sampled practice."""
    rng = np.random.default_rng(seed)
    x = words(8)
    teacher_pi = .02/2+.98*rng.dirichlet(np.full(2, .35))
    teacher_rules = (.02/4+.98*rng.dirichlet(np.full(4, .35), size=(3, 2))).reshape(3, 2, 2, 2)
    truth = grammar_inference(x, teacher_pi, teacher_rules, False)
    assert abs(truth.sum()-1) < 1e-10
    data = rng.choice(len(x), size=128, p=truth)
    ids, frequencies = np.unique(data, return_counts=True)
    frequencies = frequencies/128
    samples = x[ids]
    pi = .02/2+.98*rng.dirichlet(np.full(2, 5.))
    rules = (.02/4+.98*rng.dirichlet(np.full(4, 5.), size=(3, 2))).reshape(3, 2, 2, 2)
    initial_pi, initial_rules = pi.copy(), rules.copy()
    history = []
    start = perf_counter()
    stop = "iteration cap"
    for step in range(max_steps+1):
        likelihood, roots, counts = grammar_inference(samples, pi, rules)
        nll = float(-frequencies @ np.log(likelihood))
        prediction = grammar_inference(x, pi, rules, False)
        risk = float(truth @ (np.log(truth)-np.log(prediction)))
        assert np.isfinite(nll) and np.isfinite(risk)
        history.append((step, nll, risk))
        if step:
            assert history[-1][1] <= history[-2][1]+1e-10
        if step >= 40 and step % 5 == 0:
            recent = np.asarray(history)[-21::2]
            slope = np.polyfit(np.log(recent[:, 0]), np.log(recent[:, 1]), 1)[0]
            # Only the empirical objective selects stopping, never the teacher.
            if abs(slope) < .003:
                stop = "training NLL log-log slope stationary"
                break
        if step == max_steps: break
        pi = floor_mstep(np.einsum("n,na->a", frequencies, roots), .01)
        rule_counts = np.einsum("n,npabc->pabc", frequencies, counts).reshape(3, 2, 4)
        rules = floor_mstep(rule_counts, .005).reshape(3, 2, 2, 2)
    return {"seed": seed, "steps": step, "stop": stop, "seconds": perf_counter()-start}, {
        "history": np.asarray(history), "observed_ids": data, "teacher_pi": teacher_pi, "teacher_rules": teacher_rules,
        "initial_pi": initial_pi, "initial_rules": initial_rules, "final_pi": pi, "final_rules": rules,
        "teacher_distribution": truth, "final_distribution": prediction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    load_dotenv(Path.home()/".env")
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ["FAST_CACHE_DIR"])/"generative_practice_atomic_audit/results" if args.output is None else args.output
    archive = Path(os.environ["CACHE_DIR"])/"generative-learning/practice_research/atomic_audit" if args.archive is None else args.archive
    assert not archive.exists(), f"Choose a new --archive directory: {archive}"
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "uk_created": datetime.now(ZoneInfo("Europe/London")).isoformat(),
        "scope": "Exact rejection diagnostics; no claimed successful joint practice model.",
        "validity": "Uniform population on 00, 01, 10; finite dataset has 40 copies of each; initial shared Bernoulli MLE p=1/3. Integrate exact expected validity-reward gradient flow to t=400 at dt=.05, independently halved.",
        "grammar": "Binary categories; root-parent sentinel; parent-conditioned productions; uniform Catalan tree; 128 length-eight observations; 12 fresh worlds at seed 20103300; one random start per world; exact constrained EM with pi floor .01 and production floor .005.",
        "stopping": "Evaluate every EM update. From 40 updates, check every five; stop if absolute log-NLL/log-step slope < .003 over last 21 iterations sampled every two, or at declared cap. True KL is diagnostic only.",
        "interpretation": "Population risk improvement is not guaranteed or used for selection. Direct-fit curves do not establish equivalence to an unimplemented practice learner. No matched-compute claim.",
        "arguments": {"worlds": args.worlds, "steps": args.steps},
    }
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    shutil.copy2(__file__, output/Path(__file__).name)
    shutil.copy2(root/"tex/README.md", output/"specification.md")
    checks, validity = mathematical_audits()
    np.savez_compressed(output/"validity.npz", history=validity)
    records, curves = [], []
    for index in tqdm(range(args.worlds), desc="Direct grammar fits"):
        record, arrays = fit_one(20103300+index, args.steps)
        records.append(record)
        curves.append(arrays["history"])
        np.savez_compressed(output/f"world_{index}.npz", **arrays)
    summary = {
        "audits": checks, "validity_columns": ["flow_time", "p", "valid_probability", "nll", "population_kl"],
        "validity_initial": validity[0].tolist(), "validity_final": validity[-1].tolist(),
        "grammar": {
            "worlds": records, "initial_mean_nll": float(np.mean([h[0, 1] for h in curves])),
            "final_mean_nll": float(np.mean([h[-1, 1] for h in curves])),
            "initial_mean_kl": float(np.mean([h[0, 2] for h in curves])),
            "final_mean_kl": float(np.mean([h[-1, 2] for h in curves])),
            "worlds_improving_kl": int(sum(h[-1, 2] < h[0, 2] for h in curves)),
        },
    }
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    figure_dir = root/"tex/archive/figures"
    fig, axes = plt.subplots(2, 1, layout="constrained")
    axes[0].plot(validity[:, 0], validity[:, 2])
    axes[0].set(ylabel="Valid sample probability", title="Exact validity-reward learning")
    axes[1].plot(validity[:, 0], validity[:, 4])
    axes[1].set(xlabel="Gradient-flow time", ylabel="Population KL (nats)")
    fig.savefig(figure_dir/"practice-atomic-validity-audit.png")
    plt.close(fig)
    steps = np.arange(max(len(h) for h in curves))
    padded = np.array([np.pad(h[:, 1:], ((0, len(steps)-len(h)), (0, 0)), mode="edge") for h in curves])
    fig, axes = plt.subplots(2, 1, layout="constrained")
    for metric, ax in enumerate(axes):
        for h in curves: ax.plot(h[:, 0], h[:, metric+1], alpha=.15)
        ax.plot(steps, padded[:, :, metric].mean(axis=0), color="black", label="Mean; stopped fits held fixed")
    axes[0].set(ylabel="Training NLL (nats)", title="Parent-conditioned grammar: exact EM")
    axes[0].legend()
    axes[1].set(xlabel="Direct EM updates", ylabel="Population KL (nats)")
    fig.savefig(figure_dir/"practice-atomic-grammar-fit.png")
    plt.close(fig)
    archive.mkdir(parents=True, exist_ok=False)
    manifest = []
    sources = list(output.iterdir())+[figure_dir/"practice-atomic-validity-audit.png", figure_dir/"practice-atomic-grammar-fit.png"]
    for source in sources:
        target = archive/source.name
        shutil.copy2(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append({"source": str(source), "archive": str(target), "sha256": digest})
    (archive/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
