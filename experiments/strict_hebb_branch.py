"""Signed two-factor Hebb in feedforward readers and stochastic writers of the repeated-branch tree."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from repeated_branch import M, PATTERNS, RHO, coins, draw, mass, world

WATCH = 32
MARKS = (0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)
METRICS = ("kl", "root", "oriented_root", "shallow", "prototype_mass", "deep_kl", "reader_cos", "writer_cos", "private_teacher", "private_own", "reader_norm", "writer_norm")
LABELS = ("plain", "add", "coincide", "dense", "dense_add", "dense_sparse", "dense_matched", "learn_mix")
STORE = Path("/nfs/gatsbystor/jbauer/generative-learning/results/strict_hebb_branch")
SQRT_M = np.sqrt(M)


def relu(x):
    return np.maximum(x, 0.)


def ramp(x, threshold, gain):
    """Odd, bounded piecewise-linear activation with a central dead zone."""
    pos, neg = gain * (x-threshold), gain * (-x-threshold)
    return np.minimum(relu(pos), 1.) - np.minimum(relu(neg), 1.)


def parity(a, b):
    return relu(a+b) + relu(-a-b) - 1.


def render(leaves, rng):
    mask = coins(rng, np.zeros_like(leaves))
    return np.stack((mask, parity(mask, leaves)), axis=-1)


class Net:
    def __init__(self, rng, n, kind, variant="base"):
        self.kind, self.variant = kind, variant
        self.flat = variant in ("flat", "flat_held")
        self.hold_motif = False
        self.tied = variant == "tied" or self.flat
        self.fidelity = .85 if variant == "noise85" else .98 if variant == "noise98" else RHO
        self.b = rng.normal(0, .08, (n, M))
        self.bottom = np.zeros_like(self.b)
        self.w = .90 * coins(rng, np.zeros_like(self.b))
        self.c = self.w + rng.normal(0, .04, self.w.shape)
        self.r = self.w + rng.normal(0, .04, self.w.shape)
        self.dense = kind in ("dense", "dense_add", "dense_sparse", "dense_matched")
        self.transfer = kind in ("add", "coincide", "dense_add", "dense_sparse", "dense_matched", "learn_mix")
        self.U = rng.normal(0, 1, (n, M, M))
        self.V = self.U.transpose(0, 2, 1).copy() + rng.normal(0, .04, self.U.shape)
        if kind in ("dense_add", "dense_sparse"):
            logits = np.einsum("nij,nj->ni", self.V, np.tanh(self.w))/SQRT_M
            self.c = .9*logits/np.abs(logits).mean(1,keepdims=True)
        if kind == "dense_matched":
            desired = self.w.copy()
            self.w = .90*coins(rng,np.zeros_like(self.w))
            hidden = np.tanh(self.w)
            residual = SQRT_M*desired - np.einsum("nij,nj->ni", self.V, hidden)
            self.V += residual[:,:,None]*hidden[:,None,:]/(hidden**2).sum(1)[:,None,None]
            self.U = self.V.transpose(0,2,1).copy() + rng.normal(0,.04,self.U.shape)
            self.r = self.w + rng.normal(0,.04,self.w.shape)
        self.mix_h = rng.uniform(.35,.65,size=self.w.shape)
        self.mix_t = rng.uniform(.35,.65,size=self.w.shape)
        self.lag = False
        self.past_p = np.zeros((n,1))
        self.eps, self.eta = .02, .16
        self.sensory = .50
        if variant.startswith("sensory_"):
            self.sensory = float(variant.split("_")[1])
        if variant == "uncoupled":
            self.c = .90 * coins(rng, np.zeros_like(self.b))
            self.r = self.c + rng.normal(0, .04, self.w.shape)
        if variant == "slow":
            self.eta /= 4
        if variant == "fast":
            self.eta *= 4
        self.scramble = np.ones_like(self.b)

    def generator(self):
        weights = self.r if self.tied else self.w
        shallow = self.b if self.tied else self.bottom
        if self.dense:
            basis = self.U.transpose(0,2,1) if self.tied else self.V
            logits = np.einsum("nij,nj->ni", basis, np.tanh(weights))/SQRT_M
        else:
            logits = weights
        return self.fidelity*np.tanh(logits), self.fidelity*np.tanh(shallow)

    def sample(self, rng):
        return draw(rng, *self.generator(), flat=self.flat)

    def shallow_rate(self, leaves):
        field = .45*leaves[..., 0] + (leaves*self.b[:, None, :]).sum(-1)/SQRT_M
        return ramp(field, 0., 1.)

    def read(self, h, training=False):
        # h has shape (world, string, branch). This is the forward graph used at test and when awake.
        p = ramp((self.c[:, None]*h).sum(-1)/M, .48, 6.)
        if self.variant == "no_private":
            p = h[..., 0]
        prediction = self.past_p if training and self.lag else p
        t = ramp(prediction[..., None]*self.b[:, None]*self.scramble[:, None], 0., 1.)
        if self.kind == "coincide":
            v = self.eps*h + relu(h+t-1.) - relu(-h-t-1.)
        elif self.kind in ("add","dense_add","dense_sparse","dense_matched"):
            v = self.sensory*h + (1-self.sensory)*t
        elif self.kind == "learn_mix":
            v = ramp(self.mix_h[:,None]*h + self.mix_t[:,None]*t, 0., 1.)
        elif self.kind in ("plain","dense"):
            v = h
        else:
            raise ValueError(self.kind)
        if self.dense:
            a = np.tanh(np.einsum("nij,nkj->nki", self.U, v)/SQRT_M)
            bias = .7*p if self.transfer else 0.
            denominator = M if self.kind in ("dense_sparse","dense_matched") else SQRT_M
            q = ramp((self.r[:, None]*a).sum(-1)/denominator + bias, .35, 2.)
        else:
            a = v
            q = ramp((self.r[:, None]*v).sum(-1)/M + .7*p, .29, 4.)
        return q, v, p, t, a

    def update(self, raw, rng, mode="both", trace=None):
        y = parity(raw[..., 0], raw[..., 1])
        if self.flat:
            h = y
            lateral = coins(rng,self.shallow_rate(h[:,None]))
            shallow_corr = h*lateral
            sites = 2
        else:
            h = y[:, :, 0] if mode == "sleep" else coins(rng, self.shallow_rate(y))
            shallow_corr = (h[:, :, None]*y).sum(1)
            sites = M+1
        q, v, p, t, a = (x[:,0] for x in self.read(h[:, None], training=True))
        pre_transfer = self.past_p[:,0].copy() if self.lag else p
        if self.lag:
            self.past_p = p[:,None].copy()
        if mode == "sleep":
            q = trace[:, 0]
            if self.dense:
                a = np.tanh((self.r if self.tied else self.w)*q[:,None])
        if mode in ("both", "sleep", "free"):
            if not self.hold_motif:
                self.b += .16*shallow_corr/sites
                if self.transfer:
                    self.b += .16*pre_transfer[:,None]*t*self.scramble/sites
            if self.kind != "dense" and self.variant != "no_private":
                self.c += .025*p[:, None]*h
            if self.kind == "learn_mix":
                self.mix_h += .025*h*v
                self.mix_t += .025*t*v
            if self.dense:
                self.U += self.eta*a[:, :, None]*v[:, None, :]/SQRT_M
            self.r += self.eta*q[:, None]*a
        if mode == "free":
            q, v = trace[:,0], y[:,:,0]
            a = np.tanh(self.w*q[:,None]) if self.dense else v
        if mode == "wake" and self.tied:
            self.r += self.eta*q[:,None]*a
            if self.dense:
                self.U += self.eta*a[:,:,None]*v[:,None,:]/SQRT_M
            self.b[:,1:] += .18*(v[:,:,None]*y[:,:,1:]).mean(1)
        if mode in ("both", "wake", "free") and not self.tied:
            self.w += self.eta*q[:, None]*a
            if self.dense:
                self.V += self.eta*v[:, :, None]*a[:, None, :]/SQRT_M
            self.bottom[:,1:] += .18*(v[:, :, None]*y[:,:,1:]).mean(1)


def metric(net, motif, target):
    top, bot = net.generator()
    teacher, own = mass(target), mass(top)
    pg, pt = (own+own[:, ::-1])/2, (teacher+teacher[:, ::-1])/2
    deep = (pt*np.log(pt/pg)).sum(1)
    a, b = (1+RHO*motif[:, 1:])/2, (1+bot[:, 1:])/2
    lower = M*(a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))).sum(1)
    patterns = PATTERNS[PATTERNS[:, 0] == 1]
    rates = net.shallow_rate(patterns[None])
    def fidelity(bottom):
        row_mass = 2*mass(np.column_stack((np.zeros(len(bot)), bottom[:, 1:])), patterns)
        return (row_mass*(1+rates)/2).sum(1)
    f, fo = fidelity(RHO*motif), fidelity(bot)
    if net.flat:
        f = fo = np.ones(len(bot))
        lower = np.zeros(len(bot))
    read = mass(target*(2*f[:, None]-1))
    read_own = mass(top*(2*fo[:, None]-1))
    h = np.broadcast_to(PATTERNS, (len(bot), len(PATTERNS), M))
    q, _, p, _, _ = net.read(h)
    corr = (read*q).sum(1)
    if net.dense:
        # The effective preference on single-sign coordinate probes includes both dense layers.
        reader = (np.tanh(net.U/SQRT_M)*net.r[:, :, None]).sum(1)/SQRT_M
        basis = net.U.transpose(0,2,1) if net.tied else net.V
        writer = np.einsum("nij,nj->ni", basis, np.tanh(net.r if net.tied else net.w))/SQRT_M
    else:
        reader, writer = net.r, net.r if net.tied else net.w
    def cosine(values):
        return (values*np.sign(target)).sum(1)/(SQRT_M*np.linalg.norm(values, axis=1))
    coverage = (pg*(np.abs(np.sign(target)@PATTERNS.T)==M)).sum(1)
    return np.stack((deep+lower, (1+abs(corr))/2, (1+corr)/2, f, coverage, deep, cosine(reader), cosine(writer),
                     (read*np.abs(p)).sum(1), (read_own*np.abs(p)).sum(1),
                     np.linalg.norm(reader, axis=1), np.linalg.norm(writer, axis=1)), axis=-1)


def run(job):
    seed, n, kind, arm, variant, watching, marks = job
    motif = world(np.random.default_rng([seed, 0]), n)
    upper = world(np.random.default_rng([seed, 9]), n) if variant == "untied" else motif
    target = RHO*upper
    rng, read_rng, twins, twin_read, control = (np.random.default_rng([seed, i]) for i in range(2, 7))
    net = Net(np.random.default_rng([seed, 1]), n, kind, variant)
    twin = Net(np.random.default_rng([seed, 7]), n, kind, variant)
    permutation = control.permuted(np.broadcast_to(np.arange(M), (n, M)), axis=1)
    if variant == "orthogonal":
        polarity = np.ones_like(motif)
        polarity[:, :M//2] = -1
        neutral = .9*upper*control.permuted(polarity,axis=1)
        net.c, net.r, net.w = neutral.copy(), neutral.copy(), neutral.copy()
    before = []
    for t in range(watching+1):
        if t in (0, 1, 2, 4, 8, 16, 32) or t == watching:
            before.append(metric(net, motif, target))
        if t < watching:
            y, _ = draw(rng, target, RHO*motif, flat=net.flat)
            raw = render(y, rng)
            net.update(raw, read_rng)
            twin.update(raw, twin_read)
    net.hold_motif = variant == "flat_held"
    if variant == "knockout":
        net.scramble.fill(0.)
    if variant == "lag":
        net.lag = True
    if variant == "scramble":
        net.scramble = coins(control, np.zeros_like(motif))
    if arm == "matched":
        donor = copy.deepcopy(net)
        donor_read = copy.deepcopy(read_rng)
    out = []
    for t in range(max(marks)+1):
        if t in marks:
            out.append(metric(net, motif, target))
        if t == max(marks):
            break
        if arm == "teacher" or (arm == "classic" and t%2==0):
            y, _ = draw(rng, target, RHO*motif, flat=net.flat)
            net.update(render(y, rng), read_rng, mode="wake" if arm == "classic" else "both")
        elif arm == "twin":
            y, _ = twin.sample(twins)
            raw = render(y, twins)
            twin.update(raw, twin_read)
            net.update(raw, read_rng)
        elif arm == "matched":
            y, _ = donor.sample(rng)
            raw = render(y, rng)
            donor.update(raw, donor_read)
            y = np.take_along_axis(y*upper[:, :, None], permutation[:, :, None], axis=1)*upper[:, :, None]
            net.update(render(y, control), read_rng)
        elif arm in ("own", "frozen", "classic", "sleep", "free"):
            y, z = net.sample(rng)
            mode = "sleep" if arm in ("classic", "sleep") else "wake" if arm == "frozen" else "free" if arm == "free" else "both"
            net.update(render(y, rng), read_rng, mode=mode, trace=z if mode in ("sleep","free") else None)
        elif arm == "modal":
            # Keep ordinary shallow channel noise but remove exploration of deep branch signs.
            top, bottom = net.generator()
            y, _ = draw(rng, np.sign(top), bottom)
            net.update(render(y, rng), read_rng)
        elif arm == "random":
            y, _ = draw(rng, np.zeros_like(target), net.generator()[1])
            net.update(render(y, rng), read_rng)
        else:
            raise ValueError(arm)
    return np.stack(before), np.stack(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("pilot", "main", "probes", "budgets", "tied", "confirm", "depth"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worlds", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.suite == "pilot":
        jobs = [(m,a,"base",WATCH,MARKS) for m in LABELS for a in ("teacher","own","matched")]
    elif args.suite == "main":
        jobs = [(m,a,"base",WATCH,MARKS) for m in LABELS
                for a in ("teacher","own","twin","matched","classic","sleep","frozen","random","modal","free")]
    elif args.suite == "depth":
        jobs = [("add",a,v,WATCH,(0,32,128,512)) for a in ("teacher","own")
                for v in ("flat","flat_held","orthogonal")]
    elif args.suite == "confirm":
        jobs = [(m,a,v,WATCH,(0,32,128,512)) for m in ("add","dense","dense_matched") for v in ("base","tied")
                for a in ("own","teacher","classic","matched","twin","random")]
    elif args.suite == "probes":
        jobs = [(m,a,v,0 if v=="no_watch" else WATCH, (0,128,512)) for m in ("add","dense_matched")
                for v in ("no_watch","uncoupled","no_private","untied","scramble","knockout","lag","slow","fast",
                          "noise85","noise98","sensory_0.35","sensory_0.45","sensory_0.55","sensory_0.65")
                for a in ("own","teacher")]
        jobs += [("dense",a,v,WATCH,(0,128,512)) for v in ("slow","fast") for a in ("own","teacher")]
    elif args.suite == "tied":
        jobs = [(m,a,"tied",WATCH,MARKS) for m in ("add","dense","dense_matched")
                for a in ("own","teacher","matched","twin","random","modal")]
    else:
        jobs = [(m,"own","base",b-round(b*f),(round(b*f),)) for m in ("add","dense","dense_matched")
                for b in (32,64,100,128,256,512) for f in (0.,.25,.5,.75,1.)]
    arrays = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [(args.seed,args.worlds,*job) for job in jobs]
        for (m,a,v,watch,marks),(before,curve) in zip(jobs,tqdm(pool.map(run,tasks),total=len(jobs))):
            arrays[f"{m}/{v}/watch/{watch}"] = before
            suffix = f"/{marks[-1]}" if args.suite == "budgets" else ""
            arrays[f"{m}/{v}/{a}/{watch}{suffix}"] = curve
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out, metrics=METRICS, marks=() if args.suite=="budgets" else marks, seed=args.seed, worlds=args.worlds, **arrays)


if __name__ == "__main__":
    main()
