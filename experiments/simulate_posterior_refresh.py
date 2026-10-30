"""Linear Gaussian posterior rollouts initialized by current or frozen recognition."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads(Path("results/elicitation_toy/protocol.json").read_text())
    A, sigma = np.array(previous["A"]), previous["noise_sd"]
    dimension, eta, batch, repetitions = len(A), .08, 256, 32
    C = A @ A.T + sigma**2*np.eye(dimension)
    H = np.eye(dimension) + A.T @ A / sigma**2
    delta = .8/np.linalg.eigvalsh(H).max()
    R = np.eye(dimension)-delta*H
    G = delta*A.T/sigma**2
    innovation = 2*delta*np.eye(dimension)-delta**2*H
    chol = np.linalg.cholesky(innovation)
    posterior_covariance = np.linalg.inv(H)
    np.testing.assert_allclose(R @ posterior_covariance @ R.T + innovation, posterior_covariance, atol=1e-12)
    Bstar = np.linalg.solve(C,A).T
    np.testing.assert_allclose(H @ Bstar, A.T/sigma**2, atol=1e-12)
    assert eta*delta*np.linalg.eigvalsh(H).max()*np.linalg.eigvalsh(C).max() < 1
    assert eta*np.linalg.eigvalsh(C).max() < 1
    checkpoints = (0,1,2,4,8,16,32,64,128,256,512,1024)
    config = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "seed": 703000, "dimension": dimension, "batch_size": batch, "repetitions": repetitions, "learning_rate": eta,
              "posterior_step": delta, "checkpoints": checkpoints, "A": A.tolist(), "H": H.tolist(), "C": C.tolist(),
              "Bstar": Bstar.tolist(), "frozen_limit": G.tolist(), "noise_sd": sigma,
              "mechanism": "Initialize a latent particle with current or frozen Bx plus unit Gaussian noise, then take one exact-invariant linear Gaussian posterior transition using fixed A. Fit B to this actual sampled latent. Raw imitation omits posterior refinement.",
              "primary": "Current posterior initialization versus frozen initialization at the same number of internal pairs and one posterior transition per pair. Analytically distinct fixed points are predicted before simulation; no power-law fit.",
              "limits": "The posterior-refinement operator is given by the assumed useful generative model. This is an explicit teacher/student computation asymmetry, not pure same-policy self-imitation or a matched LLM arithmetic model."}
    (output/"protocol.json").write_text(json.dumps(config,indent=2)+"\n")
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rng = np.random.default_rng(config["seed"])
    arms = ("current_refined", "frozen_refined", "raw_self_imitation")
    weights = {arm: np.zeros((repetitions,dimension,dimension)) for arm in arms}
    population = {arm: np.zeros((dimension,dimension)) for arm in arms}
    records = []
    bayes = float(np.trace(posterior_covariance))
    def metrics(B):
        error = B-Bstar
        excess = np.einsum("...ij,jk,...ik->...",error,C,error)
        overlap = np.einsum("...ij,jk,ik->...",B,C,Bstar)/np.einsum("ij,jk,ik",Bstar,C,Bstar)
        return {"mse":(bayes+excess).tolist(), "excess_mse":excess.tolist(), "normalized_overlap":overlap.tolist()}
    for step in tqdm(range(checkpoints[-1]+1),desc="Posterior initialization"):
        if step in checkpoints:
            records.append({"steps":step,"generated_pairs":step*batch,
                            "arms":{arm:{**metrics(weights[arm]),"population":metrics(population[arm]),
                                         "first_seed_weights":weights[arm][0].tolist()} for arm in arms}})
        if step == checkpoints[-1]: break
        # The latent used to draw x is discarded: it is never a teaching target.
        x = rng.normal(size=(repetitions,batch,dimension)) @ A.T + sigma*rng.normal(size=(repetitions,batch,dimension))
        initial_noise = rng.normal(size=x.shape)
        added_noise = rng.normal(size=x.shape) @ chol.T
        first_targets = []
        for arm in arms:
            prediction = x @ weights[arm].transpose(0,2,1)
            initial = initial_noise if arm == "frozen_refined" else prediction+initial_noise
            target = initial if arm == "raw_self_imitation" else initial @ R.T + x @ G.T + added_noise
            if step == 0 and arm != "raw_self_imitation": first_targets.append(target)
            weights[arm] -= eta*(prediction-target).transpose(0,2,1) @ x / batch
            if arm == "current_refined": population[arm] -= eta*delta*H @ (population[arm]-Bstar) @ C
            elif arm == "frozen_refined": population[arm] -= eta*(population[arm]-G) @ C
        if step == 0: np.testing.assert_array_equal(*first_targets)
    final = records[-1]["arms"]
    assert np.mean(final["current_refined"]["mse"]) < np.mean(final["frozen_refined"]["mse"])
    assert np.mean(final["current_refined"]["mse"]) < dimension
    result = {"protocol":config,"bayes_mse":bayes,"frozen_limit_metrics":metrics(G),"records":records,
              "checks":{"posterior_covariance_invariant":True,"posterior_normal_equations":True,
                        "population_stability":True,"initial_refined_targets_paired":True,"current_improves_over_frozen":True}}
    (output/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    labels = {"current_refined":"Posterior sample: current initialization", "frozen_refined":"Posterior sample: frozen initialization",
              "raw_self_imitation":"Unfiltered recognition self-imitation"}
    for metric,ylabel in (("mse","Expected squared prediction error"),("normalized_overlap","Normalized overlap with optimal inverse")):
        fig,ax = plt.subplots()
        for arm in arms:
            rows = records[1:]
            x = [r["generated_pairs"] for r in rows]
            values = np.array([r["arms"][arm][metric] for r in rows])
            line, = ax.plot(x,values.mean(axis=1),label=labels[arm])
            se = values.std(axis=1,ddof=1)/np.sqrt(repetitions)
            ax.fill_between(x,values.mean(axis=1)-1.96*se,values.mean(axis=1)+1.96*se,color=line.get_color(),alpha=.2)
            ax.plot(x,[r["arms"][arm]["population"][metric] for r in rows],linestyle="--",color=line.get_color())
        if metric == "mse": ax.axhline(bayes,color="black",linestyle=":",label="Optimal posterior mean")
        ax.set(xscale="log",xlabel="Internal training pairs (one posterior step per pair)",ylabel=ylabel,title="Does refreshing posterior initialization help?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/f"posterior-refresh-{metric}.png")
        plt.close(fig)
    print(json.dumps({"bayes_mse":bayes,"frozen_limit":result["frozen_limit_metrics"],
                      "final":{arm:{key:float(np.mean(final[arm][key])) for key in ("mse","excess_mse","normalized_overlap")} for arm in arms}},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("results/posterior_refresh"))
    run(parser.parse_args().output)
