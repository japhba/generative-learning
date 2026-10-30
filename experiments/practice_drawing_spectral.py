"""Noniterative mixture control from initial-posterior command moments.

This is a rectified simultaneous-diagonalization estimator, not a claim that
posterior moments computed under an imperfect initial model are true moments.
"""

import itertools

import numpy as np

import practice_drawing_library as m


PERMUTATIONS = list(itertools.permutations(range(3)))
PROJECTIONS = np.random.default_rng(20103105).normal(size=(8,4))


def component_tensor(b): return np.einsum('ka,kb,kc->kabc', b[:,0],b[:,1],b[:,2])


def decompose(tensor, permutation, projection):
    tensor=tensor.transpose(permutation)
    pair=tensor.sum(axis=-1)
    u,s,vh=np.linalg.svd(pair,full_matrices=False)
    rank=min(3,int((s>s[0]*1e-8).sum()))
    u,v=u[:,:rank],vh[:rank].T
    contracted=np.einsum('abc,c->ab',tensor,projection)
    operator=(u.T @ contracted @ v) @ np.linalg.pinv(u.T @ pair @ v)
    _,vectors=np.linalg.eig(operator)
    first=np.abs(u @ vectors)
    first/=first.sum(axis=0,keepdims=True)
    second=np.abs(np.linalg.pinv(first) @ pair).T
    second/=second.sum(axis=0,keepdims=True)
    products=np.einsum('ak,bk->abk',first,second).reshape(16,rank)
    third=np.abs(np.linalg.pinv(products) @ tensor.reshape(16,4)).T
    third/=third.sum(axis=0,keepdims=True)
    b=np.stack([first.T,second.T,third.T],axis=1)[:,np.argsort(permutation)]
    assert np.isfinite(b).all() and np.allclose(b.sum(axis=-1),1)
    return b


def canonical_map(cat,world):
    result=np.zeros(len(cat.images),dtype=int)
    result[np.unique(world['observed'])]=world['canonical']
    return result


def exposure_frequency(cat,observed,mapping):
    actions=cat.actions[mapping[observed]].reshape(-1,3,3)
    return m.row_normalize(np.stack([np.bincount(actions[:,:,t].ravel(),minlength=4) for t in range(3)]))


def refit_initial(cat,observed,world,mapping):
    full=exposure_frequency(cat,world['observed'],mapping)
    subset=exposure_frequency(cat,observed,mapping)
    perturbation=(world['initial_b']-.02/4)/full[None]
    b=m.row_normalize(subset[None]*perturbation)
    if np.array_equal(observed,world['observed']): assert np.max(abs(b-world['initial_b']))<1e-12
    return world['initial_a'],b


def fits(cat,observed,world,mapping):
    initial_a,initial_b=refit_initial(cat,observed,world,mapping)
    code,image=cat.probabilities(initial_a,initial_b)
    frequency=np.bincount(observed,minlength=len(cat.images))/len(observed)
    posterior=code*frequency[cat.ids]/image[cat.ids]
    tensors=np.stack([np.bincount(cat.chunks[:,c],weights=posterior,minlength=64).reshape(4,4,4) for c in range(3)])
    candidates=[]
    for permutation in PERMUTATIONS:
        for projection in PROJECTIONS:
            b=decompose(tensors.mean(axis=0),permutation,projection)
            rank=len(b)
            indices=np.arange(3)%rank
            component=component_tensor(b).reshape(rank,64)
            mixing=np.maximum(tensors.reshape(3,64) @ np.linalg.pinv(component),0)+1e-6
            mixing/=mixing.sum(axis=-1,keepdims=True)
            a=m.row_normalize(mixing[:,indices]/np.bincount(indices,minlength=rank)[indices])
            b=m.row_normalize(b[indices])
            fitted=np.einsum('ck,kijl->cijl',a,component_tensor(b))
            moment_error=np.sum((tensors-fitted)**2)
            _,p=cat.probabilities(a,b)
            nll=-np.log(p[observed]).mean()
            candidates.append((moment_error,nll,p,a,b))
    moment=min(candidates,key=lambda c:c[0])
    likelihood=min(candidates,key=lambda c:c[1])
    return dict(spectral_moments=moment[2],spectral_likelihood=likelihood[2])


def audit():
    rng=np.random.default_rng(20103106)
    errors=[]
    for rank in [1,2,3]:
        for _ in range(20):
            b=rng.dirichlet(np.full(4,2.),size=(rank,3))
            pi=rng.dirichlet(np.ones(rank))
            tensor=np.einsum('k,kabc->abc',pi,component_tensor(b))
            recovered=decompose(tensor,(0,1,2),rng.normal(size=4))
            component=component_tensor(recovered).reshape(len(recovered),64)
            mixing=tensor.ravel() @ np.linalg.pinv(component)
            errors.append(np.max(abs(mixing @ component-tensor.ravel())))
    assert max(errors)<1e-9
    return dict(exact_population_tensors=60,maximum_reconstruction_error=float(max(errors)))


if __name__=='__main__': print(audit())
