"""Validate the local learning rules, bundle world-level curves, and rebuild the Hebbian MLP figure."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
import numpy as np

from repeated_branch import M, RHO, coins
from repeated_branch_review import FIGS, canvas
from strict_hebb_branch import LABELS, MARKS, STORE, SQRT_M, Net, parity, ramp, render, run

SUITES = ("main","probes","tied","budgets","stack","confirm","depth")


def pack(stage, destination):
    items = {}
    for suite in SUITES:
        with np.load(stage/f"{suite}.npz") as raw:
            for key in raw.files:
                if suite=="budgets" and key=="marks":
                    continue  # Each budget key contains its actual watch and practice counts.
                items[f"{suite}/{key}"] = raw[key]
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**items)


def check_local_rule():
    rng = np.random.default_rng(701)
    n = 7
    y = coins(rng,np.zeros((n,M,M)))
    raw = render(y,rng)
    np.testing.assert_array_equal(parity(raw[...,0],raw[...,1]),y)
    for kind in LABELS:
        for variant in ("base","tied"):
            net = Net(rng,n,kind,variant)
            # Nonzero motif so that a forward occurrence at the upper level contributes to its update.
            net.b += rng.normal(0,.8,net.b.shape)
            old = {k:getattr(net,k).copy() for k in ("r","w","U","V","c","b","bottom","mix_h","mix_t")}
            h = coins(np.random.default_rng(31),net.shallow_rate(y))
            q,v,p,t,a = (x[:,0] for x in net.read(h[:,None]))
            expected = {k: val.copy() for k,val in old.items()}
            expected["b"] += .16*(h[:,:,None]*y).sum(1)/(M+1)
            if net.transfer:
                expected["b"] += .16*p[:,None]*t/(M+1)
            if kind!="dense":
                expected["c"] += .025*p[:,None]*h
            if kind=="learn_mix":
                expected["mix_h"] += .025*h*v
                expected["mix_t"] += .025*t*v
            expected["r"] += net.eta*q[:,None]*a
            if net.dense:
                expected["U"] += net.eta*np.einsum("ni,nj->nij",a,v)/SQRT_M
            if not net.tied:
                expected["w"] += net.eta*q[:,None]*a
                if net.dense:
                    expected["V"] += net.eta*np.einsum("ni,nj->nij",v,a)/SQRT_M
                expected["bottom"][:,1:] += .18*(v[:,:,None]*y[:,:,1:]).mean(1)
            net.update(raw,np.random.default_rng(31))
            for k in old:
                np.testing.assert_allclose(getattr(net,k),expected[k],atol=1e-13)
    for q in (-1,1):
        for sensed in (-1,1):
            for motif in (-1,1):
                cell = (sensed+ramp(np.array(q*motif),0.,1.))/2
                assert q*cell*motif == float(sensed==q*motif)


def validate(data):
    check_local_rule()
    for key in data.files:
        x = data[key]
        if x.dtype.kind in "fi":
            assert np.isfinite(x).all(),key
    for kind in LABELS:
        watched = data[f"main/{kind}/base/watch/32"][-1]
        for arm in ("own","teacher","classic","sleep","matched","twin","modal","free","frozen","random"):
            np.testing.assert_array_equal(watched,data[f"main/{kind}/base/{arm}/32"][0])
    # Repeat representative whole trajectories with the delivered code.
    for kind,variant,suite in (("add","tied","tied"),("dense_matched","base","main")):
        before,curve = run((261129,192,kind,"own",variant,32,MARKS))
        np.testing.assert_allclose(before,data[f"{suite}/{kind}/{variant}/watch/32"],atol=1e-10)
        np.testing.assert_allclose(curve,data[f"{suite}/{kind}/{variant}/own/32"],atol=1e-10)
    # Test a true zero-drive teacher configuration in the noiseless row-read limit.
    net=Net(np.random.default_rng(510),1,"add")
    s=np.ones((1,M))
    neutral=np.tile((-.9,.9),M//2)[None]
    net.r,net.c=neutral.copy(),neutral.copy()
    net.b=s.copy()
    q,_,p,_,_=net.read(np.stack((s,-s),axis=1))
    assert not np.any(q) and not np.any(p)


def plot(data):
    fig,axes=canvas(2,2)
    for kind,color,name in (("add","C0","Additive branches"), ("dense_matched","C1","Structured dense"),
                            ("dense","C2","Ordinary dense")):
        for source,style in (("own","-"),("teacher","--")):
            x = data[f"tied/{kind}/tied/{source}/32"]
            for ax,col in zip(axes[0],(0,1)):
                ax.plot(MARKS,x[:,:,col].mean(1),ls=style,color=color,
                        label=name if source=="own" else "_nolegend_")
    axes[0,0].set(title="Generation with transpose-tied weights",ylabel="Population KL (nats per string)")
    axes[0,1].set(title="Recognition with transpose-tied weights",ylabel="Teacher-root accuracy",ylim=(.47,1.02))
    axes[0,1].legend(handles=[Line2D([],[],color=".25",ls=ls,label=source)
                              for source,ls in (("Own","-"),("Fresh teacher","--"))])
    for ax in axes[0]:
        ax.set_xscale("symlog",linthresh=16)
        ax.set_xticks([0,16,32,64,128,256,512],labels=["0","16","32","64","128","256","512"])
        ax.set(xlabel="Additional strings after Watching",xlim=(0,512))
    axes[0,0].legend()
    ax=axes[1,0]
    t=data["stack/marks"]
    x=data["stack/add/own"]
    for col,label in ((2,"Shallow"),(3,"Middle"),(4,"Root")):
        ax.plot(t,x[:,:,col].mean(1),label=label)
    for a,style,label in (("freeze_middle",":","Root, middle frozen"),("teacher","--","Root, fresh teacher")):
        ax.plot(t,data[f"stack/add/{a}"][:,:,4].mean(1),style,color="C2",label=label)
    ax.axvline(32,ls=":",color=".5")
    ax.set_xscale("symlog",linthresh=32)
    ax.set_xticks([0,32,64,160,544],labels=["0","32","64","160","544"])
    ax.set(title="Learning through three levels",xlabel="Total strings",ylabel="Teacher-cause accuracy",xlim=(0,544),ylim=(.47,1.02))
    ax.legend()
    ax=axes[1,1]
    totals,shares=(32,64,100,128,256,512),(0.,.25,.5,.75,1.)
    grid=np.array([[data[f"budgets/add/base/own/{b-round(b*f)}/{round(b*f)}"][0,:,0].mean()
                    for b in totals] for f in shares])
    norm=SymLogNorm(linthresh=.1,vmin=0,vmax=grid.max())
    im=ax.imshow(grid,aspect="auto",origin="lower",cmap="viridis",norm=norm)
    ax.set(xticks=range(len(totals)),xticklabels=totals,yticks=range(len(shares)),yticklabels=shares,
           xlabel="Total strings",ylabel="Fraction spent on Own",title="Sample allocation with separate weights")
    for (i,j),value in np.ndenumerate(grid):
        ax.text(j,i,f"{value:.1f}",ha="center",va="center",color="white" if norm(value)<.55 else "black")
    fig.colorbar(im,ax=ax,label="Population KL (nats)",ticks=[0,.1,1,10,100])
    svg=FIGS/"strict_hebb_branch.svg"
    fig.savefig(svg,transparent=True,metadata={"Date":None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    fig.savefig(FIGS/"strict_hebb_branch.png",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack",type=Path)
    parser.add_argument("--data",type=Path,default=STORE/"study.npz")
    args=parser.parse_args()
    if args.pack:
        pack(args.pack,args.data)
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"strict-hebb-branch"})
    with np.load(args.data) as data:
        validate(data)
        plot(data)
