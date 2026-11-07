"""Control for practice_sbn_lead.py: is the reading lead a property of learning, or of how reading and writing suffer
from the same parameter error? Each world's generator is perturbed by relative Gaussian noise of size eps on every
weight matrix; the perturbed model reads world samples with its own exact posterior (an ideal recognizer) and writes
its own samples. Same scores and chance levels as the learner runs; results keyed by KL(world || perturbed model).
Usage: python practice_sbn_lead_static.py RUN_DIR (uses that run's arguments and conditions)"""

import json
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

from practice_sbn_lead import SBN, logpdf, make_world, matched, posterior_stats, probe, strict_parse


def perturb(G, eps, rng):
    noisy = lambda M: M + eps * np.linalg.norm(M) / np.sqrt(M.size) * rng.standard_normal(M.shape)
    return SBN(G.K, G.D, G.sigma, [noisy(V) for V in G.V], [a.copy() for a in G.a], noisy(G.W), G.c.copy())


def main():
    run = Path(sys.argv[1]); d = json.load(open(run / "results.json")); args = d["args"]; out = []
    for r in tqdm(d["results"], desc="worlds"):
        K, D, w = r["K"], r["D"], r["world"]; rng = np.random.default_rng([args["seed"], w, D, *K])
        world = make_world(K, D, args["sigma"], args["beta"], rng); ht, Xt = world.sample(args["n_eval"], rng); lp_true = logpdf(world, Xt)
        for eps in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
            prng = np.random.default_rng([w, int(eps * 1000)]); G = perturb(world, eps, prng)
            m = posterior_stats(G, Xt)[0]; _, Xo = G.sample(args["n_eval"], prng); ov, ob = strict_parse(world, Xo)
            out.append(dict(name=r["name"], world=w, eps=eps, kl=float(np.mean(lp_true - logpdf(G, Xt))), read=[matched(m[l], ht[l]) for l in range(len(K))], read_probe=[probe(m[l], ht[l]) for l in range(len(K))], write=list(zip(ob, ov))))
    json.dump(out, open(run / "static.json", "w")); print(run / "static.json")


if __name__ == "__main__": main()
