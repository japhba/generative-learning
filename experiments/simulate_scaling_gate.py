"""Finite-sample confirmation for the separable one-logit Boltzmann gate."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from bestn_theory import sigmoid
from run_elicitation_pilot import ROOT, digest, save, setup
from train_bestn import stamp


OUTPUT = ROOT / "results/scaling_s4g/gate_finite"
STEPS, REPLICAS, ETA = 262144, 512, .2
CHECKS = sorted({0,*np.unique(np.rint(np.geomspace(1,STEPS,65)).astype(int)).tolist()})


def simulate(batch):
    rng=np.random.default_rng(99400+batch)
    weight=np.full(REPLICAS,np.log(.6/.4)); snapshots=[]; records=[]
    for step in tqdm(range(STEPS+1),desc=f"Boltzmann gate batch{batch}"):
        if step in CHECKS:
            errors=sigmoid(-weight)
            records.append({"step":step,"errors":errors.tolist(),"mean_error":float(errors.mean()),"weight_mean":float(weight.mean())})
            snapshots.append(weight.copy())
        if step==STEPS:break
        probability=sigmoid(weight)
        draws=rng.random((REPLICAS,batch,2))<probability[:,None,None]
        chosen=draws.any(-1).mean(-1)
        weight+=ETA*(chosen-probability)
    raw=setup()/"scaling_s4g/gate_finite";raw.mkdir(parents=True,exist_ok=True)
    path=raw/f"batch{batch}.npz";np.savez_compressed(path,weights=np.stack(snapshots),steps=CHECKS)
    value={"batch":batch,"eta":ETA,"replicas":REPLICAS,"seed":99400+batch,"records":records,"raw":str(path),"finished_at_uk":stamp()}
    save(OUTPUT/f"batch{batch}.json",value)
    return value


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    (OUTPUT/"source").mkdir()
    for name in (Path(__file__).name,"bestn_theory.py"):(OUTPUT/"source"/name).write_bytes((ROOT/"experiments"/name).read_bytes())
    save(OUTPUT/"protocol.json",{"created_at_uk":stamp(),"source_sha256":digest(Path(__file__)),"batches":[1,8],"eta":ETA,"replicas":REPLICAS,"steps":STEPS,
         "prediction":"epsilon~1/(eta*updates), exponent1 and amplitude1 in gradient-flow time, from the one-logit Boltzmann gate. No irreducible label noise or decision-boundary inputs. Actual two-draw best-of-two samples are used.",
         "noise":"Near correctness1, selected failure probability is epsilon², and scalar score variance is O(epsilon²). This differs from the Gaussian-RBM boundary noise of order1/R.",
         "fit":"Final factor8 in updates;2000 whole-replica bootstrap resamples. Prediction frozen before samples. This follows the existing gate population derivation, not the LLM curve fit.",
         "scope":"Separate toy check; does not establish that a fixed one-weight model predicts the full LLM's changing representation."})
    with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(simulate,[1,8]))
    fig,ax=plt.subplots();summary=[];rng=np.random.default_rng(99499)
    for row in results:
        steps=np.array([r["step"] for r in row["records"]]);times=ETA*steps
        errors=np.array([r["errors"] for r in row["records"]]);means=errors.mean(-1)
        keep=steps>=STEPS/8;x=np.log(times[keep]);x-=x.mean()
        beta=float(-x@np.log(means[keep])/(x@x))
        boot=[float(-x@np.log(errors[keep][:,rng.integers(REPLICAS,size=REPLICAS)].mean(-1))/(x@x)) for _ in range(2000)]
        summary.append({"batch":row["batch"],"beta":beta,"ci95":np.quantile(boot,[.025,.975]).tolist(),"last_error":float(means[-1]),"last_error_over_prediction":float(means[-1]*times[-1])})
        ax.loglog(times[1:],means[1:],label=f"Batch {row['batch']}")
    ax.loglog(times[1:],1/times[1:],"--",label="Unfitted inverse-time prediction")
    ax.set(xlabel="Learning rate × SGD updates",ylabel="Sampled-policy error probability",ylim=(1e-5,1))
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-gate-finite.png");plt.close(fig)
    save(OUTPUT/"analysis.json",{"finished_at_uk":stamp(),"cases":summary});print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
