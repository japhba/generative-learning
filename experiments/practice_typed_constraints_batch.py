"""Prospective fixed-design confirmation with permuted atoms and untouched final graphs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import practice_typed_constraints as m


def data_objective(phi, allowed, action, weights):
    probability = m.policy(phi, allowed, weights)
    nll = -np.log(probability[np.arange(len(action)), action]).mean()
    gradient = (np.einsum("bc,bcf->bf", probability, phi)-phi[np.arange(len(action)), action]).mean(axis=0)
    return float(nll), gradient


def project(weights, radius): return weights*min(1., radius/np.linalg.norm(weights))


def static_menu(radius):
    menu = {"uniform_legal": np.zeros(14)}
    for label, indices, sign in [("dead_only", [0], -1), ("forced_only", [1, 2, 3], -1),
                               ("pressure_only", [4, 5, 6], -1), ("elimination_only", [7, 8, 9], -1),
                               ("reuse", [13], 1), ("value0", [10], 1), ("value1", [11], 1), ("value2", [12], 1)]:
        value = np.zeros(14); value[indices] = sign
        menu[label] = radius*value/np.linalg.norm(value)
    value = np.r_[-10., -np.ones(3), np.zeros(10)]
    menu["dead_then_forced"] = radius*value/np.linalg.norm(value)
    value = np.r_[-np.ones(10), np.zeros(3), 1.]
    menu["equal_signed"] = radius*value/np.linalg.norm(value)
    return menu


def run(job):
    output, index, config = job
    directory = output/f"world_{index}"
    directory.mkdir()
    seed = config["seed"]+index
    streams = np.random.SeedSequence(seed).spawn(7)
    atom_rng, exposure_rng, practice_rng, shuffle_rng, validation_rng, final_rng, evaluation_rng = [np.random.default_rng(s) for s in streams]
    palette, edge_types = atom_rng.permutation(3), atom_rng.permutation(3)
    teacher = m.teacher_relations()[edge_types][:, palette][:, :, palette]
    exact = m.Exact(config["vertices"], teacher)
    examples, solutions, attempts = exact.exposure(exposure_rng, config["examples"], config["edge_probability"])
    counts, learned = m.acquire(examples, solutions)
    assert np.array_equal(learned, teacher)
    phi, allowed, action = m.generate(examples, np.zeros(14), learned, exposure_rng, target=solutions, record_features=True)
    initial, exposure_history, exposure_fit = m.constrained_exposure(phi, allowed, action, config["radius"])
    validation = m.context(validation_rng, config["validation_graphs"], config["vertices"], config["edge_probability"])
    validation_feasible = np.array([exact.valid(graph).any() for graph in validation])
    validation_inputs = np.repeat(validation, config["validation_attempts"], axis=0)
    validation_mask = np.repeat(validation_feasible, config["validation_attempts"])
    validation_seed = int(evaluation_rng.integers(2**32))
    weights, shuffled = initial.copy(), initial.copy()
    records, weight_history, shuffled_history, recent = [], [], [], []
    status = "maximum declared updates"
    for step in range(config["max_steps"]+1):
        if step % config["eval_every"] == 0:
            regular_reward = m.generate(validation_inputs, weights, learned, np.random.default_rng(validation_seed))[0]
            shuffled_reward = m.generate(validation_inputs, shuffled, learned, np.random.default_rng(validation_seed))[0]
            train = float(np.mean(recent)) if step else np.nan
            records.append([step, regular_reward.mean(), regular_reward[validation_mask].mean(),
                            shuffled_reward.mean(), shuffled_reward[validation_mask].mean(), train])
            weight_history.append(weights.copy()); shuffled_history.append(shuffled.copy())
            np.savez_compressed(directory/"progress.npz", records=np.array(records), weights=np.array(weight_history),
                                shuffled_weights=np.array(shuffled_history), practice_rng=json.dumps(practice_rng.bit_generator.state),
                                shuffle_rng=json.dumps(shuffle_rng.bit_generator.state))
            recent = []
            if step >= 1000:
                window = np.array(records[-10:])
                train_slope = np.polyfit(np.log(window[:, 0]), np.log(1-window[:, 5]), 1)[0]
                validation_slope = np.polyfit(np.log(window[:, 0]), np.log(1-window[:, 2]), 1)[0]
                if abs(train_slope) < .003 and abs(validation_slope) < .003:
                    status = "stationary training and validation risk"
                    break
        if step == config["max_steps"]: break
        graphs = m.context(practice_rng, config["batch"], config["vertices"], config["edge_probability"])
        reward, score = m.generate(graphs, weights, learned, practice_rng)
        advantage = config["batch"]/(config["batch"]-1)*(reward-reward.mean())
        weights = project(weights+config["rate"]*np.mean(advantage[:, None]*score, axis=0), config["radius"])
        noise_reward, noise_score = m.generate(graphs, shuffled, learned, shuffle_rng)
        shift = int(shuffle_rng.integers(1, config["batch"]))
        # Every assigned reward and its baseline depend only on other independent trajectories.
        false_advantage = np.roll(noise_reward, shift)-(noise_reward.sum()-noise_reward)/(config["batch"]-1)
        shuffled = project(shuffled+config["rate"]*np.mean(false_advantage[:, None]*noise_score, axis=0), config["radius"])
        assert np.isfinite(weights).all() and np.isfinite(shuffled).all()
        recent.append(float(reward.mean()))
    datafit = initial.copy()
    datafit_history = []
    for data_step in range(step+1):
        nll, gradient = data_objective(phi, allowed, action, datafit)
        if data_step % config["eval_every"] == 0: datafit_history.append([data_step, nll, np.linalg.norm(datafit-initial)])
        if data_step == step: break
        datafit = project(datafit-gradient, config["radius"])
    final_graphs = m.context(final_rng, config["test_graphs"], config["vertices"], config["edge_probability"])
    final_feasible = np.array([exact.valid(graph).any() for graph in final_graphs])
    assert not ({g.tobytes() for g in examples} & {g.tobytes() for g in final_graphs})
    assert not ({g.tobytes() for g in validation} & {g.tobytes() for g in final_graphs})
    final_inputs = np.repeat(final_graphs, config["test_attempts"], axis=0)
    final_seed = int(evaluation_rng.integers(2**32))
    methods = dict(initial=initial, practice=weights, shuffled_reward=shuffled, continued_datafit=datafit) | static_menu(config["radius"])
    samples = {}
    for label, vector in methods.items():
        reward = m.generate(final_inputs, vector, learned, np.random.default_rng(final_seed))[0]
        samples[label] = reward.reshape(config["test_graphs"], config["test_attempts"])
    samples["lexical_legal"] = m.generate(final_inputs, np.zeros(14), learned, np.random.default_rng(final_seed),
                                         deterministic=True)[0].reshape(config["test_graphs"], config["test_attempts"])
    samples["mrv_lcv"] = np.repeat(np.array([m.greedy_mrv(g, learned) for g in final_graphs])[:, None], config["test_attempts"], axis=1)
    endpoints = {key: dict(unconditional=float(value.mean()), conditional=float(value[final_feasible].mean())) for key, value in samples.items()}
    static_keys = list(static_menu(config["radius"]))
    best_static = max(static_keys, key=lambda k: endpoints[k]["conditional"])
    result = dict(index=index, seed=seed, final_step=step, status=status, endpoints=endpoints,
                  best_static=best_static, palette=palette.tolist(), edge_types=edge_types.tolist(),
                  relation_counts=counts.tolist(), exposure_fit=exposure_fit, exposure_attempts=attempts,
                  final_feasible_graphs=int(final_feasible.sum()), initial_weights=initial.tolist(),
                  final_weights=weights.tolist(), shuffled_weights=shuffled.tolist(),
                  datafit_nll_change=float(datafit_history[-1][1]-datafit_history[0][1]),
                  datafit_weight_change=float(np.linalg.norm(datafit-initial)),
                  initial_norm=float(np.linalg.norm(initial)), final_norm=float(np.linalg.norm(weights)))
    np.savez_compressed(directory/"world.npz", examples=examples, solutions=solutions, teacher=teacher, learned=learned,
        relation_counts=counts, validation_graphs=validation, validation_feasible=validation_feasible,
        final_graphs=final_graphs, final_feasible=final_feasible, exposure_history=exposure_history,
        datafit_history=np.array(datafit_history), initial=initial, final=weights, shuffled=shuffled, datafit=datafit)
    np.savez_compressed(directory/"final_samples.npz", **samples)
    (directory/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def analyze(output, rows, config):
    ordered = [rows[str(i)] for i in range(config["worlds"])]
    keys = list(ordered[0]["endpoints"])
    bootstrap = np.random.default_rng(20103520).integers(len(ordered), size=(50000, len(ordered)))

    def estimate(values):
        values = np.asarray(values)
        return dict(mean=float(values.mean()), per_world=values.tolist(),
                    bootstrap95=np.quantile(values[bootstrap].mean(axis=-1), [.025, .975]).tolist())

    endpoint = {key: np.array([r["endpoints"][key]["conditional"] for r in ordered]) for key in keys}
    best = np.array([r["endpoints"][r["best_static"]]["conditional"] for r in ordered])
    result = dict(worlds=len(ordered), conditional={key: estimate(value) for key, value in endpoint.items()},
                  best_static=estimate(best), contrasts={})
    for label, value in {key: endpoint[key] for key in ["initial", "shuffled_reward", "continued_datafit", "mrv_lcv"]}.items():
        difference = endpoint["practice"]-value
        result["contrasts"][label] = estimate(difference) | dict(worlds_improved=int((difference > 0).sum()))
    difference = endpoint["practice"]-best
    result["contrasts"]["best_static"] = estimate(difference) | dict(worlds_improved=int((difference > 0).sum()))
    result["datafit_max_abs_nll_change"] = max(abs(r["datafit_nll_change"]) for r in ordered)
    result["datafit_max_weight_change"] = max(r["datafit_weight_change"] for r in ordered)
    result["max_initial_kkt_residual"] = max(r["exposure_fit"]["kkt_residual"] for r in ordered)
    result["all_relations_recovered"] = True
    (output/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    trajectories = [np.load(output/f"world_{i}/progress.npz")["records"] for i in range(config["worlds"])]
    grid = np.arange(0, config["max_steps"]+1, config["eval_every"])
    fig, axes = plt.subplots(2, 1, layout="constrained")
    for column, label in [(2, "Learned correctness feedback"), (4, "Shuffled feedback")]:
        curves = np.array([np.interp(grid, r[:, 0], r[:, column]) for r in trajectories])
        axes[0].plot(grid, curves.mean(axis=0), label=label)
    initial_validation = np.mean([r[0, 2] for r in trajectories])
    axes[0].axhline(initial_validation, color="black", linestyle=":", label="Converged original-data fit")
    axes[0].set(xlabel="Offline updates; stopped runs held fixed", ylabel="Validation single-attempt\nsuccess probability")
    axes[0].legend()
    shown = ["initial", "shuffled_reward", "best_static", "practice", "mrv_lcv"]
    values = np.stack([best if key == "best_static" else endpoint[key] for key in shown], axis=-1)
    for row in values: axes[1].plot(range(len(shown)), row, alpha=.4)
    axes[1].plot(range(len(shown)), values.mean(axis=0), color="black", marker="o", label="Mean of twelve datasets")
    axes[1].set_xticks(range(len(shown)), ["Original\nfit", "Shuffled\nfeedback", "Best static\nweights", "Offline\npractice", "MRV + LCV\nheuristic"])
    axes[1].set(ylabel="Untouched test single-attempt\nsuccess probability")
    axes[1].legend()
    figure = Path(__file__).resolve().parents[1]/"tex/archive/figures/practice-typed-constraints-confirmation.png"
    fig.savefig(figure)
    plt.close(fig)
    shutil.copy2(figure, output/figure.name)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(worlds=12, seed=20103480, vertices=10, edge_probability=.55, examples=256,
                  validation_graphs=512, validation_attempts=8, test_graphs=1024, test_attempts=16,
                  batch=512, max_steps=3000, eval_every=100, radius=12., rate=1.)
    sources = [Path(__file__), Path(m.__file__), Path(__file__).with_name("practice_graph_constraints.py")]
    for source in sources: shutil.copy2(source, args.output/source.name)
    protocol = dict(uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(), config=config,
        prospective=True, pilot="Seed20103460 informed candidate choice; confirmation seeds are new and all reported.",
        teachers="One fixed balanced relation family, independently permuted palette and edge-type labels in each dataset; not independent unconstrained relation draws.",
        primary="Untouched test conditional validity: practice minus converged original-data fit.",
        controls="Matched-update shuffled feedback, matched-update full-gradient original-data fit, declared static coefficient menu, MRV+LCV without backtracking.",
        shuffle="Cyclic nonzero shift of rewards; baseline excludes the receiver's original reward. Expected unprojected update is zero.",
        best_static="Optimistic upper envelope selects highest final-test success among the predeclared fixed vectors separately for each dataset.",
        stopping="After1000updates, absolute log-loss/log-step slopes below.003 over10validation evaluations for BOTH generated-training failure and validation failure; cap3000.",
        final_test="Independent new1024contexts×16raw attempts, sampled only after stopping. Exact feasibility used only for evaluation.",
        uncertainty="Paired bootstrap over12 independent datasets, shared relational family with random labels; not theorem about finite sampled steps.",
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    rows = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run, (args.output, i, config)): i for i in range(config["worlds"])}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Independent typed confirmations"):
            index = futures[future]
            rows[str(index)] = future.result()
            (args.output/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    analyze(args.output, rows, config)


if __name__ == "__main__": main()
