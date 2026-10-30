"""Exact weight-space identities for affine inference in a belief network."""

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from deep_wake_sleep import BeliefNetwork, Recognizer, components, softmax


def audit(layers, seed, neglect):
    rng=np.random.default_rng(seed)
    generator=BeliefNetwork(layers,6,rng,2.)
    joint,logjoint=generator.distribution();px=joint.sum(1);p,lp=softmax(logjoint)
    rec=Recognizer(6,len(generator.h),1,32,rng,neglect)
    design=np.column_stack((generator.x,np.ones(len(generator.x))))
    target=np.linalg.lstsq(design,lp-lp.mean(1,keepdims=True),rcond=None)[0]
    weights=np.vstack((rec.weights[0],rec.biases[0]));weights-=weights.mean(1,keepdims=True)
    error=weights-target;displacement=design@error
    q,lq,_=rec.forward(generator.x)
    def fisher_product(prob):
        centered=displacement-np.sum(prob*displacement,axis=1,keepdims=True)
        return design.T@(px[:,None]*prob*centered)
    reverse=fisher_product(q)
    nodes,quadrature=np.polynomial.legendre.leggauss(64)
    forward=np.zeros_like(reverse)
    for s,w in zip((nodes+1)/2,quadrature/2,strict=True):
        path_q=softmax(design@(target+s*error))[0]
        forward+=w*fisher_product(path_q)
    actual={arm:np.vstack(components(generator,rec,px,arm,rng,128,False)[1]) for arm in ("elbo","forward")}
    covariance_error=float(np.max(np.abs(reverse-actual["elbo"])))
    integrated_error=float(np.max(np.abs(forward-actual["forward"])))
    compilation_error=float(np.max(np.abs(softmax(design@target)[0]-p)))
    wake_contraction=float(np.sum(error*actual["elbo"]))
    sleep_contraction=float(np.sum(error*actual["forward"]))
    jeffreys=float(np.sum(px[:,None]*(q-p)*(lq-lp)))
    assert covariance_error<1e-9 and integrated_error<1e-9 and compilation_error<1e-12
    assert wake_contraction>=0 and sleep_contraction>=0
    assert abs(sleep_contraction-jeffreys)<1e-9
    return {"layers":layers,"seed":seed,"neglect":neglect,"posterior_compilation_error":compilation_error,
            "current_fisher_weight_identity_error":covariance_error,"integrated_fisher_weight_identity_error":integrated_error,
            "wake_radial_contraction":wake_contraction,"forward_radial_contraction":sleep_contraction,
            "forward_over_wake_radial_contraction":sleep_contraction/wake_contraction}


if __name__=="__main__":
    cases=[(layers,seed,neglect) for layers in ([6],[3,3],[2,2,2],[4,4]) for seed in (9880,9881,9882) for neglect in (0,3,6)]
    rows=[audit(*case) for case in tqdm(cases,desc="Weight-space mechanism audit")]
    result={"cases":rows,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope":"Exact affine-recognizer gradient identities and Euclidean gradient-flow contractions. These do not describe Adam’s parameter preconditioner or prove a scaling exponent in the nonlinear LLM."}
    root=Path(__file__).resolve().parents[1]
    (root/"results/deep_wake_sleep/weight_mechanism_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
