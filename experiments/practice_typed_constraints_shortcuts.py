"""Pilot-selected direct controls, evaluated on the saved fresh confirmation tests."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

import practice_typed_constraints as m


def direct(inputs, relations, rule, rng):
    batch, vertices, _ = inputs.shape
    extended = np.concatenate([relations, np.ones((1, 3, 3), dtype=bool)])
    allowed, previous = np.ones((batch, vertices, 3), dtype=bool), np.full((batch, vertices), -1)
    alive = np.ones(batch, dtype=bool)
    for step in range(vertices):
        phi = m.feature_block(inputs, allowed, previous, step, extended)
        current = allowed[:, step] & (phi[:, :, 0] == 0)
        alive &= current.any(axis=-1)
        current[~alive] = True
        if rule == "forced":
            forced = phi[:, :, 1:4].sum(axis=-1)
            current &= forced == np.where(current, forced, np.inf).min(axis=-1, keepdims=True)
        elif rule == "volume":
            compatible = extended[inputs[:, step, step+1:]]
            remaining = np.einsum("bucv,buv->buc", compatible.astype(int), allowed[:, step+1:].astype(int))
            volume = np.log(np.maximum(remaining, 1)).sum(axis=1)
            current &= volume == np.where(current, volume, -np.inf).max(axis=-1, keepdims=True)
        else: raise ValueError(rule)
        probability = current/current.sum(axis=-1, keepdims=True)
        action = (rng.random((batch, 1)) > probability.cumsum(axis=-1)).sum(axis=-1)
        previous[:, step] = action
        atom = extended[inputs[:, step, step+1:]]
        permitted = atom[np.arange(batch)[:, None], np.arange(vertices-step-1), action[:, None]]
        allowed[:, step+1:] &= permitted
    left, right = np.triu_indices(vertices, 1)
    correct = extended[inputs[:, left, right], previous[:, left], previous[:, right]].all(axis=-1)
    assert np.array_equal(correct, alive)
    return correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/Path(__file__).name)
    protocol = json.loads((args.source/"protocol.json").read_text())["config"]
    (args.output/"protocol.json").write_text(json.dumps(dict(uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(),
        status="Additional controls selected from the separate pilot, after confirmation began; no confirmation outcome inspected to select them.",
        choices=["Same-radius static dead:forced coefficient ratio3, others zero",
                 "Hard rule: eliminate locally illegal or immediately future-dead choices, then minimize forced-neighbor count with random ties",
                 "Same hard exclusion, then maximize product of remaining future domain sizes with random ties"],
        caveat="Hard controls are deterministic decision rules with random ties; they do not obey the actor's finite-norm softmax parameterization."), indent=2)+"\n")
    outcomes = []
    for index in tqdm(range(protocol["worlds"]), desc="Pilot-selected shortcut controls", mininterval=2):
        directory = args.source/f"world_{index}"
        world = np.load(directory/"world.npz")
        samples = np.load(directory/"final_samples.npz")
        summary = json.loads((directory/"summary.json").read_text())
        inputs = np.repeat(world["final_graphs"], protocol["test_attempts"], axis=0)
        stream = np.random.SeedSequence(summary["seed"]).spawn(7)[6]
        evaluation_rng = np.random.default_rng(stream)
        evaluation_rng.integers(2**32)
        evaluation_seed = int(evaluation_rng.integers(2**32))
        fixed = np.r_[-3., -np.ones(3), np.zeros(10)]
        fixed *= protocol["radius"]/np.linalg.norm(fixed)
        values = {"static_ratio3": m.generate(inputs, fixed, world["learned"], np.random.default_rng(evaluation_seed))[0]}
        for rule in ["forced", "volume"]:
            values[f"hard_forward_{rule}"] = direct(inputs, world["learned"], rule, np.random.default_rng(evaluation_seed))
        values = {key: value.reshape(protocol["test_graphs"], protocol["test_attempts"]) for key, value in values.items()}
        np.savez_compressed(args.output/f"world_{index}.npz", **values)
        outcomes.append(dict(index=index, practice=float(samples["practice"][world["final_feasible"]].mean()),
                             controls={key: float(value[world["final_feasible"]].mean()) for key, value in values.items()}))
    boot = np.random.default_rng(20103521).integers(len(outcomes), size=(50000, len(outcomes)))
    result = dict(rows=outcomes, controls={})
    for key in outcomes[0]["controls"]:
        control = np.array([row["controls"][key] for row in outcomes])
        difference = np.array([row["practice"]-row["controls"][key] for row in outcomes])
        result["controls"][key] = dict(mean=float(control.mean()), practice_minus_control=float(difference.mean()),
            bootstrap95=np.quantile(difference[boot].mean(axis=-1), [.025, .975]).tolist(),
            worlds_practice_better=int((difference > 0).sum()))
    (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
