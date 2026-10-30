"""Audit the revised specification: one posterior fit versus iterative causal learning."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m
from practice_drawing_fixed_posterior import complete_q, counts, mstep


def weighted_program_counts(cat, a, b, codes, weights):
    """Update motif recognition while keeping the program posterior fixed."""
    component = np.prod(b[:, np.arange(3)[None, :], m.MICRO], axis=-1)
    acount, bcount = np.zeros_like(a), np.zeros_like(b)
    for chunk in range(3):
        frequency = np.bincount(cat.chunks[codes, chunk], weights=weights, minlength=64)
        responsibility = a[chunk, :, None] * component
        responsibility /= responsibility.sum(axis=0, keepdims=True)
        joint = responsibility * frequency
        acount[chunk] = joint.sum(axis=-1)
        for position in range(3):
            for action in range(4):
                bcount[:, position, action] += joint[:, m.MICRO[:, position] == action].sum(axis=-1)
    assert np.max(abs(acount.sum(axis=-1)-1)) < 1e-12
    assert abs(bcount.sum()-9) < 1e-12
    return acount, bcount


def score(cat, a, b, observed, teacher):
    program, image = cat.probabilities(a, b)
    nll = -np.log(image[observed]).mean()
    risk = np.sum(teacher * np.log(teacher/image))
    return float(nll), float(risk), program


def audit_world(cat, source, exact, output, index):
    world = np.load(source/f"run_{index}/world.npz")
    observed = np.searchsorted(cat.images, world["observed"])
    assert np.array_equal(cat.images[observed], world["observed"])
    a, b = world["initial_a"], world["initial_b"]
    acount, bcount, initial_p = counts(cat, a, b, observed)
    initial_code, _ = cat.probabilities(a, b)
    fixed_a, fixed_b = mstep(acount, bcount)
    repeat_a, repeat_b = mstep(acount, bcount)
    assert np.array_equal(repeat_a, fixed_a) and np.array_equal(repeat_b, fixed_b)
    one_nll, one_kl, _ = score(cat, fixed_a, fixed_b, observed, world["teacher_p"])
    initial_nll, initial_kl, _ = score(cat, a, b, observed, world["teacher_p"])
    q_gain = complete_q(fixed_a, fixed_b, acount, bcount)-complete_q(a, b, acount, bcount)
    assert initial_nll-one_nll >= q_gain-1e-10
    assert q_gain >= -1e-12
    saved = np.load(exact/f"world_{index}.npz")
    assert abs(saved["records"][1, 1]-one_nll) < 1e-12
    assert abs(saved["records"][1, 2]-one_kl) < 1e-12
    assert np.max(np.abs(saved["weights"][1]-np.r_[fixed_a.ravel(), fixed_b.ravel()])) < 1e-12

    unique, frequencies = np.unique(observed, return_counts=True)
    aliases = [cat.aliases(image) for image in unique]
    codes = np.concatenate(aliases)
    weights = np.concatenate([n*initial_code[z]/initial_p[image] for image, n, z in zip(unique, frequencies, aliases)])/len(observed)
    assert abs(weights.sum()-1) < 1e-12
    assert np.array_equal(cat.canvas[codes], np.repeat(cat.images[unique], [len(z) for z in aliases]))
    checked_a, checked_b = weighted_program_counts(cat, a, b, codes, weights)
    assert np.max(abs(checked_a-acount)) < 1e-12
    assert np.max(abs(checked_b-bcount)) < 1e-12

    # The posterior over programs is now fixed. Only the latent motif assignments change.
    frozen_program_history, frozen_program_weights = [], []
    status = "update limit"
    for step in range(401):
        nll, risk, program = score(cat, a, b, observed, world["teacher_p"])
        latent_nll = -np.dot(weights, np.log(program[codes]))
        frozen_program_history.append([step, nll, risk, latent_nll])
        frozen_program_weights.append(np.r_[a.ravel(), b.ravel()])
        assert np.isfinite(frozen_program_history[-1]).all()
        if step > 0: assert frozen_program_history[-1][3] <= frozen_program_history[-2][3]+1e-10
        if step >= 40:
            recent = np.array(frozen_program_history[-20:])
            if abs(np.polyfit(np.log(recent[:, 0]), np.log(recent[:, 3]), 1)[0]) < .003:
                status = "stationary fixed-program complete-data likelihood"
                break
        if step == 400: break
        acount_now, bcount_now = weighted_program_counts(cat, a, b, codes, weights)
        a, b = mstep(acount_now, bcount_now)
    frozen_program_history = np.array(frozen_program_history)
    assert abs(frozen_program_history[1, 2]-one_kl) < 1e-12
    joint = np.load(source/f"run_{index}/sleep.npz")
    cache = np.load(source/f"run_{index}/fixed_cache.npz")
    np.savez_compressed(output/f"world_{index}.npz", one_a=fixed_a, one_b=fixed_b, acount=acount, bcount=bcount,
                        frozen_program_history=frozen_program_history, frozen_program_weights=np.array(frozen_program_weights),
                        exact_history=saved["records"], joint_history=joint["records"], fixed_cache_history=cache["records"],
                        fixed_program_status=status)
    return dict(index=index, seed=int(world["seed"]), initial_kl=initial_kl, one_posterior_kl=one_kl,
                fixed_program_kl=float(frozen_program_history[-1, 2]), joint_kl=float(joint["records"][-1, 2]),
                exact_em_kl=float(saved["records"][-1, 2]), fixed_cache_kl=float(cache["records"][-1, 2]),
                initial_nll=initial_nll, one_posterior_nll=one_nll, one_posterior_q_gain=float(q_gain),
                exact_em_steps=int(saved["records"][-1, 0]), fixed_program_steps=int(frozen_program_history[-1, 0]),
                exact_program_reconstruction_pairs=len(codes), reconstruction_error=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=m.CACHE/"unmasked_confirmation")
    parser.add_argument("--exact", type=Path, default=m.CACHE/"unmasked_direct_em")
    parser.add_argument("--output", type=Path, default=Path(os.environ["FAST_CACHE_DIR"])/"generative_practice_revision_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(m.__file__), Path(__file__).with_name("practice_drawing_fixed_posterior.py")]
    for source in sources: shutil.copy2(source, args.output/source.name)
    n = json.loads((args.source/"protocol.json").read_text())["arguments"]["worlds"]
    protocol = dict(uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(), retrospective=True,
                    source=str(args.source), exact=str(args.exact), worlds=n,
                    main_control="Freeze the full exact initial posterior over programs and motifs; take its globally optimal categorical M-step. Every repeat is identical.",
                    secondary_control="Freeze the initial program posterior but iteratively learn motif responsibilities and generator rows. This does not freeze all recognition.",
                    secondary_stopping="After 40 updates stop when log fixed-program NLL against log update slope has magnitude below .003 over 20 points; cap 400.",
                    evaluation="Teacher used only for population KL, never choosing updates or stopping. Exact reconstruction verified for every positive-weight original-image/program pair.",
                    interpretation="One-shot posterior fit is not all possible analytic estimators. Iterative exact EM is a likelihood learner, not a trained reconstruction map.",
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    cat = m.Catalogue()
    rows = [audit_world(cat, args.source, args.exact, args.output, index) for index in tqdm(range(n), desc="Revised-spec one-shot audit")]
    bootstrap = np.random.default_rng(20103420).integers(n, size=(50000, n))

    def estimate(values):
        values = np.asarray(values)
        return dict(mean=float(values.mean()), median=float(np.median(values)), per_world=values.tolist(),
                    bootstrap95=np.quantile(values[bootstrap].mean(axis=-1), [.025, .975]).tolist())

    fields = ["initial_kl", "one_posterior_kl", "fixed_program_kl", "joint_kl", "exact_em_kl", "fixed_cache_kl"]
    summary = dict(rows=rows, estimates={field: estimate([r[field] for r in rows]) for field in fields}, contrasts={})
    for control in ["one_posterior_kl", "fixed_program_kl", "exact_em_kl", "fixed_cache_kl"]:
        differences = np.array([r["joint_kl"]-r[control] for r in rows])
        summary["contrasts"][f"joint_minus_{control}"] = estimate(differences) | dict(worlds_improved=int((differences < 0).sum()))
    summary["checks"] = dict(saved_exact_first_step_match=True, every_supported_reconstruction_exact=True,
                             full_posterior_repeat_identical=True, max_full_em_updates=max(r["exact_em_steps"] for r in rows),
                             min_full_em_updates=min(r["exact_em_steps"] for r in rows))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    results = [np.load(args.output/f"world_{index}.npz") for index in range(n)]
    grid = np.arange(max(max(r["exact_history"][-1, 0], r["frozen_program_history"][-1, 0]) for r in results)+1)
    fig, axes = plt.subplots(2, 1, layout="constrained")
    for key, label in [("exact_history", "Update programs and motifs"), ("frozen_program_history", "Fix programs; learn motifs")]:
        curves = np.array([np.interp(grid, r[key][:, 0], r[key][:, 2]) for r in results])
        axes[0].plot(grid, curves.mean(axis=0), label=label)
    initial = summary["estimates"]["initial_kl"]["mean"]
    one = summary["estimates"]["one_posterior_kl"]["mean"]
    axes[0].plot(grid, np.where(grid == 0, initial, one), linestyle="--", label="Fix full initial posterior")
    axes[0].axhline(summary["estimates"]["joint_kl"]["mean"], linestyle=":", color="black", label="Joint practice endpoint")
    axes[0].set(xlabel="Exact fitting updates; stopped models held fixed", ylabel="Mean population KL (nats)", xscale="symlog", yscale="log")
    axes[0].set_xlim(0, grid[-1])
    axes[0].legend()
    shown = ["one_posterior_kl", "fixed_program_kl", "joint_kl", "exact_em_kl"]
    values = np.array([[r[key] for key in shown] for r in rows])
    for index, value in enumerate(values): axes[1].plot(range(len(shown)), value, alpha=.4)
    axes[1].plot(range(len(shown)), values.mean(axis=0), color="black", marker="o", label="Mean of sixteen worlds")
    axes[1].set_xticks(range(len(shown)), ["One posterior\nfit", "Fix programs;\nlearn motifs", "Joint\npractice", "Iterative\nexact EM"])
    axes[1].set(ylabel="Population KL (nats)", yscale="log")
    axes[1].legend()
    figure = m.ROOT/"tex/archive/figures/practice-revision-oneshot.png"
    fig.savefig(figure)
    plt.close(fig)
    shutil.copy2(figure, args.output/figure.name)
    print(json.dumps({key: summary[key] for key in ["estimates", "contrasts", "checks"]}, indent=2))
    print(str(figure))
    print(str(args.output/"summary.json"))


if __name__ == "__main__": main()
