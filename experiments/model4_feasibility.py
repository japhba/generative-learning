"""Exact and sampled local learning in a two-level latent tree with learned parity evidence."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
import json
import hashlib
from pathlib import Path
import time

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

load_dotenv(Path.home() / ".env")
OUT = Path("/nfs/gatsbystor/jbauer/generative-learning/model4_feasibility")
X = np.array(list(product([-1., 1.], repeat=6)))
XB = (X + 1) / 2
F = np.concatenate([X.reshape(64, 2, 3), np.ones((64, 2, 1))], -1)
ST = np.array(list(product(range(2), range(4), range(4))))
Z, HL, HR = ST.T
OH = [np.eye(4)[HL], np.eye(4)[HR]]
FEATURES = [c for k in [1, 2, 3] for c in combinations(range(6), k)]
CHI = np.stack([X[:, c].prod(1) for c in FEATURES], 1)
ARMS = ["symmetric", "forward", "practice", "watch", "fresh", "ae", "mle", "no_evidence", "no_entropy", "frozen", "exact_posterior", "unweighted", "forward_frozen", "forward_no_evidence"]


def lse(a, axis=-1):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def lsm(a): return a - lse(a)[..., None]
def sigmoid(a): return np.exp(-np.logaddexp(0, -a))


def world(seed, n, evidence="basis"):
    rng = np.random.default_rng(seed)
    mask = rng.choice([-1., 1.], 6)
    latent = np.array(list(product([-1., 1.], repeat=3)))
    z, u, v = latent.T
    values = np.stack([u, z*u, z, v, z*v, z], 1) * mask
    ix = ((values + 1) / 2 @ (2 ** np.arange(5, -1, -1))).astype(int)
    pop = np.zeros(64); pop[ix] = 1/8
    draws = rng.choice(ix, n)
    empirical = np.bincount(draws, minlength=64) / n
    moments = empirical @ CHI
    selected, basis = [], {}
    for j, c in enumerate(FEATURES):
        if abs(moments[j]) < 1 - 1e-10: continue
        if evidence=="all": selected.append(j); continue
        row = sum(1 << k for k in c)
        for pivot, other in sorted(basis.items(), reverse=True):
            if row & (1 << pivot): row ^= other
        if row:
            basis[row.bit_length()-1] = row
            selected.append(j)
    beta = .5 * np.log(2*n + 1)
    energy = beta * CHI[:, selected] @ np.sign(moments[selected])
    logpi = energy - lse(energy)
    return dict(mask=mask, latent=latent, ix=ix, pop=pop, empirical=empirical, energy=energy, logpi=logpi,
                selected=selected, signs=np.sign(moments[selected]), beta=beta, draws=draws)


def init(seed, coupling=.8):
    rng = np.random.default_rng(100000 + seed)
    t = .5 * rng.standard_normal((2, 2, 4))
    d = .7 * rng.standard_normal((2, 4, 3))
    r = np.empty((2, 4, 4))
    r[:, :3] = d.transpose(0, 2, 1) / 2
    r[:, 3] = -np.logaddexp(d/2, -d/2).sum(-1)
    h = lsm(t).transpose(0, 2, 1)
    r = coupling*r + np.sqrt(1-coupling**2)*.35*rng.standard_normal(r.shape)
    h = coupling*h + np.sqrt(1-coupling**2)*.35*rng.standard_normal(h.shape)
    return [t, d, r, h]


def tables(params):
    t, d, r, h = params
    lt = lsm(t); pt = np.exp(lt); pd = sigmoid(d)
    le = np.einsum("xbj,bkj->xbk", XB.reshape(64, 2, 3), d) - np.logaddexp(0, d).sum(-1)[None]
    lp = -np.log(2) + lt[0, Z, HL] + lt[1, Z, HR] + le[:, 0, HL] + le[:, 1, HR]
    lr = lsm(np.einsum("xbj,bjk->xbk", F, r)); pr = np.exp(lr)
    lh = lsm(h[0, :, None, :] + h[1, None, :, :]); ph = np.exp(lh)
    lq = lr[:, 0, HL] + lr[:, 1, HR] + lh[HL, HR, Z]
    return lp, lq, pt, pd, pr, ph


def decoder_grad(w, pt, pd):
    gt, gd = np.zeros((2, 2, 4)), np.zeros((2, 4, 3))
    wh = w.sum(0)
    for b, hb in enumerate([HL, HR]):
        count = np.zeros((2, 4)); np.add.at(count, (Z, hb), wh)
        gt[b] = count - count.sum(1)[:, None]*pt[b]
        wxh = w @ OH[b]
        gd[b] = wxh.T @ XB[:, 3*b:3*b+3] - wxh.sum(0)[:, None]*pd[b]
    return gt, gd


def encoder_grad(w, pr, ph):
    gr, gh = np.zeros((2, 4, 4)), np.zeros((2, 4, 2))
    wx = w.sum(1)
    wh = w.sum(0).reshape(2, 4, 4).transpose(1, 2, 0)
    diff = wh - wh.sum(-1)[..., None]*ph
    gh[0], gh[1] = diff.sum(1), diff.sum(0)
    for b in range(2): gr[b] = F[:, b].T @ (w @ OH[b] - wx[:, None]*pr[:, b])
    return gr, gh


def own_counts(params, rng, batch):
    t,d,_,_=params; pt=np.exp(lsm(t)); pd=sigmoid(d)
    z=rng.integers(2,size=batch)
    hl=(rng.random((batch,1))>pt[0,z].cumsum(1)).sum(1)
    hr=(rng.random((batch,1))>pt[1,z].cumsum(1)).sum(1)
    xb=rng.random((batch,6))<np.concatenate([pd[0,hl],pd[1,hr]],1)
    xi=(xb@(2**np.arange(5,-1,-1))).astype(int); hi=16*z+4*hl+hr
    counts=np.zeros((64,32)); np.add.at(counts,(xi,hi),1)
    return counts


def grads(params, data, mode, rng=None, batch=None, counts=None):
    if mode in ["symmetric", "symmetric_frozen", "symmetric_no_evidence"]:
        evidence = data if mode!="symmetric_no_evidence" else data | {"energy":np.zeros(64)}
        shared=None if batch is None else own_counts(params,rng,batch)
        gf=grads(params,evidence,"forward",rng,batch,shared); gr=grads(params,evidence,"practice",rng,batch,shared)
        combined=[(a+b)/2 for a,b in zip(gf,gr)]
        if mode=="symmetric_frozen": combined[2]*=0; combined[3]*=0
        return combined
    lp, lq, pt, pd, pr, ph = tables(params)
    p, q = np.exp(lp), np.exp(lq)
    if mode in ["forward", "forward_frozen", "forward_no_evidence", "forward_explore", "forward_explore_frozen", "forward_explore_no_evidence"]:
        energy = np.zeros(64) if mode.endswith("no_evidence") else data["energy"]
        weights = np.exp(energy[:,None] - lse(energy) + lq)
        if batch is not None:
            if mode.startswith("forward_explore"):
                nown=rng.binomial(batch,.8)
                counts=own_counts(params,rng,nown)
                xi=rng.integers(64,size=batch-nown)
                hi=(rng.random((batch-nown,1))>q[xi].cumsum(1)).sum(1)
                np.add.at(counts,(xi,hi),1)
                logproposal=np.logaddexp(np.log(.8)+lp,np.log(.2/64)+lq)
            else:
                counts = own_counts(params,rng,batch) if counts is None else counts
                logproposal=lp
            active = counts>0
            ratio = energy[:,None]+lq-logproposal
            weights = np.zeros_like(p)
            weights[active] = counts[active]*np.exp(ratio[active]-ratio[active].max())
            weights /= weights.sum()
        gt, gd = decoder_grad(weights, pt, pd)
        reward = lp-lq
        baseline = (q*reward).sum(1)[:,None] if batch is None else -6*np.log(2)
        gr, gh = encoder_grad(weights*(reward-baseline), pr, ph)
        if mode.endswith("frozen"): gr*=0; gh*=0
    elif mode=="unweighted":
        weights = p if batch is None else own_counts(params,rng,batch)/batch
        gt,gd=decoder_grad(weights,pt,pd); gr,gh=encoder_grad(weights,pr,ph)
    elif mode in ["practice", "no_evidence", "no_entropy", "frozen", "exact_posterior"]:
        energy = np.zeros(64) if mode == "no_evidence" else data["energy"]
        reward = energy[:, None] + lq - lp
        if mode == "no_entropy": reward = energy[:, None] + lq
        if mode == "exact_posterior": reward = np.broadcast_to((energy-lse(lp, 1))[:, None], lp.shape)
        weights = p
        if batch is not None: weights = (own_counts(params,rng,batch) if counts is None else counts) / batch
        # The baseline is the exact model expectation; it is independent of each sampled trajectory.
        wg = weights * (reward - ((p*reward).sum() if batch is None else 0))
        wr = weights
        gt, gd = decoder_grad(wg, pt, pd); gr, gh = encoder_grad(wr, pr, ph)
        if mode == "frozen": gr *= 0; gh *= 0
    else:
        dist = data["pop"] if mode == "fresh" else data["empirical"]
        post = np.exp(lp - lse(lp, 1)[:, None]) if mode == "mle" else q
        weights = dist[:, None]*post
        if batch is not None: weights = rng.multinomial(batch, weights.ravel()).reshape(weights.shape) / batch
        gt, gd = decoder_grad(weights, pt, pd)
        if mode == "mle": wr = weights
        else:
            reward = lp if mode == "ae" else lp-lq
            wr = weights * (reward - (post*reward).sum(1)[:, None])
        gr, gh = encoder_grad(wr, pr, ph)
    return [gt, gd, gr, gh]


def information(joint):
    positive = joint > 0
    independent = joint.sum(0)[None, :]*joint.sum(1)[:, None]
    return (joint[positive]*np.log(joint[positive]/independent[positive])).sum()


def metrics(params, data):
    lp, lq, _, _, pr, _ = tables(params)
    p, q = np.exp(lp), np.exp(lq); px = p.sum(1); logpx = lse(lp, 1)
    ix = data["ix"]; pop = data["pop"]
    nmi = []
    for b in range(2):
        true = ((data["latent"][:, 0]+1) + (data["latent"][:, b+1]+1)/2).astype(int)
        joint = np.zeros((4,4)); np.add.at(joint, true, pr[ix, b]/8)
        nmi.append(information(joint)/np.log(4))
    qz = q.reshape(64, 2, 4, 4).sum((2,3))
    joint = np.zeros((2,2)); np.add.at(joint, ((data["latent"][:,0]+1)/2).astype(int), qz[ix]/8)
    shallow = np.prod((X*data["mask"]).reshape(64,2,3), -1) == 1
    mismatch = (p*(lp-logpx[:,None]-lq)).sum()
    reverse = (px*(logpx-data["logpi"])).sum()
    target=np.exp(data["logpi"])
    forward=(target[:,None]*q*(data["logpi"][:,None]+lq-lp)).sum()
    logw=data["energy"][:,None]+lq-lp
    ess=np.exp(2*lse(data["energy"])-lse((lp+2*logw).ravel()))
    valid = px[ix].sum(); validconditional = px[ix]/valid
    return dict(kl=float(-np.log(8)-logpx[ix].mean()), validity=float(valid), shallow=float(px@shallow.all(1)),
                entropy=float(-(px*logpx).sum()), effective_modes=float(np.exp(-(validconditional*np.log(validconditional)).sum())),
                root_mi=float(information(joint)/np.log(2)), branch_mi=float(np.mean(nmi)),
                objective=float(reverse+mismatch), reverse=float(reverse), inference_gap=float(mismatch),
                forward_objective=float(forward), importance_ess=float(ess), train_nll=float(-data["empirical"]@logpx))


def check():
    data=world(3,16); par=init(5); lp,lq,*_=tables(par)
    assert np.isclose(np.exp(lp).sum(),1) and np.allclose(np.exp(lq).sum(1),1)
    for mode in ["practice","watch","ae","forward","symmetric"]:
        g=grads(par,data,mode)
        def objective(ps):
            p,q,*_=tables(ps)
            if mode=="practice": return (np.exp(p)*(data["energy"][:,None]+q-p)).sum()
            if mode=="forward": return (np.exp(data["logpi"][:,None]+q)*(p-q)).sum()
            if mode=="symmetric":
                return .5*((np.exp(data["logpi"][:,None]+q)*(p-q)).sum()+(np.exp(p)*(data["energy"][:,None]+q-p)).sum())
            return (data["empirical"][:,None]*np.exp(q)*(p if mode=="ae" else p-q)).sum()
        errors=[]
        for j in range(4):
            for index in [tuple(0 for _ in par[j].shape), tuple(s-1 for s in par[j].shape)]:
                eps=1e-5; a=[x.copy() for x in par]; b=[x.copy() for x in par]
                a[j][index]+=eps; b[j][index]-=eps
                numerical=(objective(a)-objective(b))/(2*eps)
                errors.append(abs(numerical-g[j][index]))
        assert max(errors)<1e-8, (mode,errors)
        print(mode,"maximum finite-difference error",max(errors))
    m=metrics(par,data)
    jointkl=(np.exp(lp)*(lp-data["logpi"][:,None]-lq)).sum()
    assert abs(m["objective"]-jointkl)<1e-10
    assert all(np.allclose(a,b) for a,b in zip(grads(par,data,"forward"),grads(par,data | {"pop":np.roll(data["pop"],1)},"forward")))
    print("All normalization, gradient and KL-decomposition checks passed.")


def run(spec):
    seed,n,coupling,steps,watchsteps,lr,batch,arms,evidence = spec
    data=world(seed,n,evidence); params=init(seed,coupling); rng=np.random.default_rng(seed+200000)
    timeline=np.unique(np.r_[0,np.round(np.geomspace(1,max(steps,1),70))]).astype(int)
    watchtime=np.unique(np.r_[0,np.round(np.geomspace(1,max(watchsteps,1),30))]).astype(int)
    before=[]
    for t in range(watchsteps+1):
        if t in watchtime: before.append(dict(step=t,**metrics(params,data)))
        if t<watchsteps:
            gg=grads(params,data,"watch",rng,batch)
            params=[a+lr*b for a,b in zip(params,gg)]
    records={}; states={}
    for arm in arms:
        par=[a.copy() for a in params]; rows=[]
        armrng=np.random.default_rng(seed+300000)
        for t in range(steps+1):
            if t in timeline:
                row=metrics(par,data); assert np.isfinite(list(row.values())).all(),(seed,arm,t,row)
                rows.append(dict(step=t,**row))
            if t<steps:
                gg=grads(par,data,arm,armrng,batch)
                par=[a+lr*b for a,b in zip(par,gg)]
        records[arm]=rows
        states[arm]=[a.tolist() for a in par]
    pop=data["pop"]; energy=data["energy"]
    return dict(seed=seed,n=n,coupling=coupling,steps=steps,watchsteps=watchsteps,lr=lr,batch=batch,evidence=evidence,
                selected=[list(FEATURES[j]) for j in data["selected"]], signs=data["signs"].tolist(), mask=data["mask"].tolist(),
                critic_kl=float(-np.log(8)-data["logpi"][data["ix"]].mean()),
                critic_validity=float(np.exp(data["logpi"])[data["ix"]].sum()),
                false_constraints=int(sum(abs(pop@CHI[:,j])<1-1e-10 for j in data["selected"])),
                energy=energy.tolist(), valid_indices=data["ix"].tolist(), observed_indices=data["draws"].tolist(),
                watching=before,curves=records,parameters=states)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--check",action="store_true"); p.add_argument("--name",default="pilot")
    p.add_argument("--seeds",type=int,default=3); p.add_argument("--start",type=int,default=0); p.add_argument("--n",type=int,nargs="+",default=[16])
    p.add_argument("--coupling",type=float,nargs="+",default=[.8]); p.add_argument("--steps",type=int,default=2500)
    p.add_argument("--watchsteps",type=int,default=500); p.add_argument("--lr",type=float,default=.5); p.add_argument("--batch",type=int)
    p.add_argument("--arms",nargs="+",default=ARMS); p.add_argument("--workers",type=int,default=4)
    p.add_argument("--evidence",choices=["basis","all"],default="basis"); p.add_argument("--allocation",action="store_true")
    args=p.parse_args()
    if args.check: check(); return
    specs=[(s,n,c,args.steps,args.watchsteps,args.lr,args.batch,args.arms,args.evidence) for n in args.n for c in args.coupling for s in range(args.start,args.start+args.seeds)]
    if args.allocation:
        specs=[(s,16,.8,round(t*f),t-round(t*f),args.lr,128,[args.arms[0]],args.evidence) for t in [100,250,500,1000,2000] for f in [0,.25,.5,.75,.9,1] for s in range(args.start,args.start+args.seeds)]
    start=time.time()
    with ProcessPoolExecutor(args.workers) as pool: result=list(tqdm(pool.map(run,specs),total=len(specs),desc=args.name))
    OUT.mkdir(parents=True,exist_ok=True)
    payload=dict(configuration=vars(args),elapsed_seconds=time.time()-start,created_at=datetime.now().astimezone().isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=result)
    (OUT/f"{args.name}.json").write_text(json.dumps(payload,separators=(",",":")))
    for arm in result[0]["curves"]:
        print(arm, {k:round(float(np.mean([r["curves"][arm][-1][k] for r in result])),4) for k in ["kl","validity","root_mi","branch_mi","objective"]})
    print("Initial",{k:round(float(np.mean([r["curves"][next(iter(r["curves"]))][0][k] for r in result])),4) for k in ["kl","validity","root_mi","branch_mi"]})
    print("Learned critic KL",np.mean([r["critic_kl"] for r in result]),"seconds",time.time()-start)


if __name__=="__main__": main()
