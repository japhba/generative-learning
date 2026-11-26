"""Three-level sign tree with transpose-tied weights and forward-pass Hebb at every plastic synapse."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from repeated_branch import M, PATTERNS, RHO, WATCH, coins, mass, world
from repeated_branch_stack import TIMES, broadcast, log_prob
from strict_hebb_branch import Net, SQRT_M, parity, render

METRICS = ("kl_mc", "panel_se", "shallow", "middle", "root", "middle_writer_overlap", "top_writer_overlap")


def deep_hebb(net, h, state):
    q, v, p, _, a = state
    net.c += .025*(h*p[...,None]).mean(1)
    net.r += net.eta*(a*q[...,None]).mean(1)
    if net.dense:
        net.U += net.eta*np.einsum("nki,nkj->nij",a,v)/(h.shape[1]*SQRT_M)


def stack(job):
    seed, n, panel, kind, arm = job
    motif = world(np.random.default_rng([seed,0]), n)
    target = RHO*motif
    rng, read_rng = (np.random.default_rng([seed,i]) for i in (1,2))
    middle = Net(np.random.default_rng([seed,3]),n,kind,"tied")
    top = Net(np.random.default_rng([seed,4]),n,kind,"tied")
    top.b = middle.b
    test = np.random.default_rng([seed,5])
    z = coins(test,np.zeros((n,panel)))
    k = broadcast(test,z,target)
    h = broadcast(test,k,target)
    random_read = test.random(h.shape)
    log_teacher = log_prob(h,target,target)
    rows = PATTERNS[PATTERNS[:,0]==1]
    teacher_rows = 2*mass(np.column_stack((np.zeros(n),target[:,1:])),rows)
    curve = []
    for t in range(TIMES[-1]+1):
        mid_g, shallow_g = middle.generator()
        top_g, _ = top.generator()
        if t in TIMES:
            shallow = ((1+middle.shallow_rate(rows[None]))*teacher_rows).sum(1)/2
            test_h = h*np.where(random_read<shallow[:,None,None,None],1.,-1.)
            # Chunk the held-out panel to keep the largest dense-layer temporary small.
            read_mid, read_top = [], []
            for part in np.array_split(test_h,8,axis=1):
                m = middle.read(part.reshape(n,-1,M))[0].reshape(n,-1,M)
                read_mid.append(m)
                read_top.append(top.read(m)[0])
            mid_corr = (k*np.concatenate(read_mid,axis=1)).mean((1,2))
            top_corr = (z*np.concatenate(read_top,axis=1)).mean(1)
            p, q = (1+target[:,1:])/2,(1+shallow_g[:,1:])/2
            bottom_kl = M*M*(p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))).sum(1)
            ratio = log_teacher-log_prob(h,mid_g,top_g)
            curve.append(np.stack((bottom_kl+ratio.mean(1),ratio.std(1,ddof=1)/np.sqrt(panel),shallow,
                                   (1+abs(mid_corr))/2,(1+abs(top_corr))/2,(mid_g*motif).mean(1),
                                   (top_g*motif).mean(1)),axis=-1))
        if t == TIMES[-1]:
            break
        foreign = t < WATCH or arm=="teacher"
        written_z = coins(rng,np.zeros(n))
        written_k = broadcast(rng,written_z,target if foreign else top_g)
        written_h = broadcast(rng,written_k,target if foreign else mid_g)
        y = broadcast(rng,written_h,target if foreign else shallow_g)
        y[...,0] = written_h
        raw = render(y,rng)
        y = parity(raw[...,0],raw[...,1]).reshape(n,M*M,M)
        read_h = coins(read_rng,middle.shallow_rate(y)).reshape(n,M,M)
        mids = middle.read(read_h)
        tops = top.read(mids[0][:,None])
        if t < WATCH or arm!="freeze_middle":
            deep_hebb(middle,read_h,mids)
        deep_hebb(top,mids[0][:,None],tops)
        # The shallow weight has one pooled local increment from each site where it is used.
        shallow_corr = (read_h.reshape(n,M*M,1)*y).sum(1)
        reuse = (mids[2][...,None]*mids[3]).sum(1) + (tops[2][...,None]*tops[3]).sum(1)
        middle.b += .16*(shallow_corr+reuse)/(M*M+M+1)
    return np.stack(curve)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    n,panel,seed=96,512,261130
    jobs=[(seed,n,panel,m,a) for m in ("add","dense_matched") for a in ("teacher","own","freeze_middle")]
    with ProcessPoolExecutor(max_workers=3) as pool:
        result={f"{m}/{a}":x for (_,_,_,m,a),x in zip(jobs,tqdm(pool.map(stack,jobs),total=len(jobs)))}
    np.savez_compressed(args.out,seed=seed,worlds=n,panel=panel,marks=TIMES,metrics=METRICS,**result)
