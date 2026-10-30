"""Independent finite-state and finite-binomial checks for sampled practice."""

import argparse
from itertools import product
import json
from math import comb
from pathlib import Path
import shutil

import numpy as np

import practice_drawing_sampled as s
import practice_drawing_library as m


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cat=m.Catalogue(); rng=np.random.default_rng(20103131)
    data=s.world(cat,20103120)
    a,b=data['initial_a'],data['initial_b']
    codes=rng.integers(len(cat.actions),size=128)
    weights=rng.dirichlet(np.ones(len(codes)))
    actual_a,actual_b=s.expected_counts(codes,weights,a,b)
    expected_a,expected_b=np.zeros((3,3)),np.zeros((3,3,4))
    expected_p=[]
    for code,weight in zip(codes,weights):
        terms=[]
        for labels in product(range(3),repeat=3):
            p=np.prod([a[c,labels[c]]*np.prod(b[labels[c],np.arange(3),cat.actions[code,3*c:3*c+3]]) for c in range(3)])
            terms.append((labels,p))
        total=sum(p for _,p in terms); expected_p.append(total)
        for labels,p in terms:
            for c in range(3):
                expected_a[c,labels[c]]+=weight*p/total
                for t in range(3): expected_b[labels[c],t,cat.actions[code,3*c+t]]+=weight*p/total
    probability_error=float(np.max(abs(s.program_probability(codes,a,b)-expected_p)))
    count_error=float(max(np.max(abs(actual_a-expected_a)),np.max(abs(actual_b-expected_b))))
    geometry=s.Geometry()
    assert np.array_equal(geometry.indices,cat.indices)
    minimum_coverage=1.
    for n in [1,2,10,64,256]:
        k=np.arange(n+1)
        upper=np.r_[s.upper_binomial(k[:-1]/n,n,.01),1.]
        for p in np.linspace(.001,.999,1001):
            probabilities=np.array([comb(n,int(i))*p**i*(1-p)**(n-i) for i in k])
            coverage=probabilities[upper>=p].sum()
            minimum_coverage=min(minimum_coverage,coverage)
            assert coverage>=.99-1e-12
    targets,counts=np.unique(data['observed'],return_counts=True)
    mass=counts/counts.sum()
    caches=[cat.aliases(int(np.searchsorted(cat.images,target)))[-2:] for target in targets]
    cached,posterior,old=s.posterior(caches,mass,a,b)
    ac,bc=s.expected_counts(cached,posterior,a,b)
    aa,bb=s.mstep(ac,bc)
    gain=s.complete_q(aa,bb,ac,bc)-s.complete_q(a,b,ac,bc)
    new=s.posterior(caches,mass,aa,bb)[2]
    _,image_p=cat.probabilities(aa,bb)
    ll=np.mean(np.log(image_p[np.searchsorted(cat.images,data['observed'])]))
    assert new>=old+gain-1e-10 and ll>=new-1e-10
    assert probability_error<1e-14 and count_error<1e-12
    result=dict(probability_error=probability_error,coactivity_error=count_error,
        binomial_cases=5005,minimum_coverage_at_99_percent=minimum_coverage,
        generator_q_gain=float(gain),elbo_gain=float(new-old),likelihood_minus_elbo=float(ll-new),
        geometry_matches_enumeration=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(__file__,args.output/'source.py')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
