"""Norm and teacher-overlap predictions for the finite-sample RBM."""

import json

import matplotlib.pyplot as plt
import numpy as np

from run_elicitation_pilot import ROOT
from scaling_theory import asymptotic_constant,normal_cdf,phi
from run_elicitation_pilot import save


def main():
    root=ROOT/"results/scaling_s4g/finite_confirmation"
    fig,ax=plt.subplots()
    for path in sorted(root.glob("d*.json")):
        row=json.loads(path.read_text());records=row["records"][1:]
        times=np.array([r["time"] for r in records]);norm=np.array([r["norm_mean"] for r in records])
        line,=ax.loglog(times,norm,label=f"{row['dimension']} weights, batch {row['batch']}")
        lam=(row["dimension"]-1)*row["eta"]*float(phi(1.5))/row["batch"]
        prediction=(8*asymptotic_constant()*times)**.25 if row["dimension"]==1 else (3*lam*times)**(1/3)
        ax.loglog(times,prediction,"--",color=line.get_color())
    ax.set(xlabel="Learning rate × SGD updates",ylabel="Weight norm",title="Solid: sampled RBM weights; dashed: predictions")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-rbm-weights.png");plt.close(fig)
    fig,ax=plt.subplots()
    for path in sorted(root.glob("d8*.json")):
        row=json.loads(path.read_text());records=row["records"][1:]
        times=np.array([r["time"] for r in records]);norm=np.array([r["norm_mean"] for r in records]);angle=1-np.array([r["overlap_mean"] for r in records])
        line,=ax.loglog(times,angle,label=f"Batch {row['batch']}")
        lam=(row["dimension"]-1)*row["eta"]*float(phi(1.5))/row["batch"]
        prediction=(row["dimension"]-1)*row["eta"]/(2*row["batch"]*1.5*(3*lam*times)**(1/3))
        ax.loglog(times,prediction,"--",color=line.get_color())
    ax.set(xlabel="Learning rate × SGD updates",ylabel="1 − cosine with true generative weights",title="Finite-sample angular error and its prediction")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-rbm-alignment.png");plt.close(fig)
    fig,ax=plt.subplots();comparison=[]
    for path in sorted(root.glob("d8*.json")):
        row=json.loads(path.read_text());records=row["records"][1:]
        weights=np.load(row["raw"])["weights"][1:]
        overlaps=weights[:,:,0]/np.linalg.norm(weights,axis=-1)
        greedy=np.array([[normal_cdf(-1.5*rho) for rho in values] for values in overlaps]).mean(-1)-normal_cdf(-1.5)
        sampled=np.array([r["excess_mean"] for r in records]);times=np.array([r["time"] for r in records])
        line,=ax.loglog(times,greedy,label=f"Batch {row['batch']}: greedy")
        ax.loglog(times,sampled,"--",color=line.get_color(),label=f"Batch {row['batch']}: sampled")
        comparison.append({"batch":row["batch"],"last_greedy_excess":float(greedy[-1]),"last_sampled_excess":float(sampled[-1]),
                           "fraction_due_to_angular_error":float(greedy[-1]/sampled[-1])})
    ax.set(xlabel="Learning rate × SGD updates",ylabel="Error above the known Bayes floor",title="Noisy RBM scaling also improves greedy decisions")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-rbm-greedy.png");plt.close(fig)
    save(root/"greedy_comparison.json",{"cases":comparison,"population_limit":"With rho1 fixed, greedy error is already the Bayes floor; the remaining population1/2 sampled-error tail is confidence sharpening."})


if __name__=="__main__":main()
