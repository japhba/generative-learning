"""Independent checks of block balancing, cause counts, and the tight anchor."""

import json
from pathlib import Path
import shutil

import numpy as np

import practice_drawing_library as m
import practice_drawing_balanced as b


def scalar_counts(cat, states, weights):
    acount, bcount = np.zeros((3,3)), np.zeros((3,3,4))
    for state, weight in zip(states, weights):
        code, labels = state//27, b.LABELS[state%27]
        for chunk in range(3):
            acount[chunk,labels[chunk]] += weight
            for position in range(3): bcount[labels[chunk],position,cat.actions[code,3*chunk+position]] += weight
    return acount,bcount


def main():
    output=m.CACHE/'balanced_audit'
    output.mkdir(parents=True,exist_ok=False)
    cat,rng=m.Catalogue(),np.random.default_rng(20103100)
    world=m.world(cat,20103020)
    recognition=b.Recognition(cat,world)
    mass=np.bincount(world['observed'],minlength=len(cat.images))/len(world['observed'])
    initial_a,initial_b,initial_p=b.counts(cat,world['initial_a'],world['initial_b'],world['observed'])
    a,new_b=b.mstep(initial_a,initial_b)
    codes=rng.integers(len(cat.actions),size=512)
    joint=b.joint_probability(cat,a,new_b,np.repeat(codes,27),np.tile(b.LABELS,(len(codes),1))).reshape(-1,27).sum(axis=-1)
    code_p,_=cat.probabilities(a,new_b)
    marginal_error=float(np.max(abs(joint-code_p[codes])))
    states=rng.integers(len(cat.actions)*27,size=512)
    weights=rng.normal(size=512)
    fast=b.coactivity(cat,states,weights)
    slow=scalar_counts(cat,states,weights)
    coactivity_error=float(max(np.max(abs(x-y)) for x,y in zip(fast,slow)))
    images=rng.choice(len(cat.images),200000,p=mass)
    sampled=recognition.samples(images,rng)
    empirical=b.coactivity(cat,sampled,np.full(len(sampled),1/len(sampled)))
    # Three mutually exclusive motif labels per chunk give binomial SEs.
    sampling_z=float(np.max(abs(empirical[0]-initial_a)/np.sqrt(initial_a*(1-initial_a)/len(sampled))))
    states=np.unique(sampled[:4096])
    state_images=cat.ids[states//27]
    old=recognition.probability[states].copy()
    joint=b.joint_probability(cat,a,new_b,states//27,b.LABELS[states%27])
    before=np.sum(mass[state_images]*old*np.log(joint/old))
    ca,cb,gain,_=recognition.balance(states,a,new_b,mass)
    new=recognition.probability[states]
    leaf=recognition.tree[recognition.base+recognition.rank[states//27]*27+states % 27]
    assert np.array_equal(leaf,new)
    rows=recognition.prefix_mass(27*cat.offset[1:])-recognition.prefix_mass(27*cat.offset[:-1])
    assert np.max(abs(rows-1)) < 1e-9
    after=np.sum(mass[state_images]*new*np.log(joint/new))
    gain_error=float(abs(after-before-gain))
    expected=scalar_counts(cat,states,mass[state_images]*(new-old))
    transfer_error=float(max(np.max(abs(x-y)) for x,y in zip((ca,cb),expected)))
    initial_ll=np.log(initial_p[world['observed']]).mean()
    initial_q=b.complete_q(world['initial_a'],world['initial_b'],initial_a,initial_b)
    first_q=b.complete_q(a,new_b,initial_a,initial_b)
    bound_error=float(abs(recognition.exact_bound(a,new_b,mass)-(initial_ll+first_q-initial_q+gain)))
    next_a,next_b=b.mstep(initial_a+ca,initial_b+cb)
    generator_gain=b.complete_q(next_a,next_b,initial_a+ca,initial_b+cb)-b.complete_q(a,new_b,initial_a+ca,initial_b+cb)
    final_bound=initial_ll+first_q-initial_q+gain+generator_gain
    _,final_p=cat.probabilities(next_a,next_b)
    ll=np.log(final_p[world['observed']]).mean()
    result=dict(marginalization_error=marginal_error,coactivity_error=coactivity_error,initial_sampler_max_z=sampling_z,
                recognition_gain=float(gain),recognition_gain_error=gain_error,transfer_count_error=transfer_error,
                tight_anchor_error=bound_error,generator_gain=float(generator_gain),likelihood_minus_certified_bound=float(ll-final_bound))
    assert max(marginal_error,coactivity_error,gain_error,transfer_error,bound_error) < 1e-10
    assert sampling_z<6 and gain>=0 and generator_gain>=0 and ll>=final_bound-1e-10
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(__file__,output/'source.py')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
