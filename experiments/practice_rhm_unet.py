"""Masked-diffusion U-Net on the random hierarchy model: does the denoiser read a level's latent structure in world samples
before it writes samples that are valid at that level?

World: RHM (practice_rhm.RHM), v symbols per level, nc root classes, m synonymous productions per symbol, branching s,
depth L, d = s^L leaves. Height h = 1 (just above the leaves) .. L (root).
Learner: convolutional U-Net as in Favero et al. (ICML 2025): encoder block h merges s children (kernel = stride = s), so
its output at position j is the representation of the height-h node j; decoder mirrors it with skip connections. Trained
as an absorbing-state (masked) diffusion model: k ~ U{1..d} random positions masked, loss (d/k) * sum of cross-entropies on
the masked positions (the any-order autoregressive bound on -log p(x), nats per string). Online: a fresh batch from the
world at every step (--P 0), or a fixed training set of P strings. Several worlds train at once as independent groups of
one grouped network (groups never mix; Adam is per parameter), only to share kernel launches.

Measures per height h, at log-spaced steps:
  writes       share of the learner's own strings (unmasked one position at a time, random order, from all-masked) that
               parse under the world's grammar up to height h.
  writes_node  share of height-h nodes in own strings whose subtree parses up to h (no conjunction over nodes).
  fills        a world string with one random leaf masked: the learner's probability mass on leaf values that keep the
               string valid up to height h (the exact posterior puts all its mass there). One step of generation in a
               valid context: inference of a missing part.
  abstracts    synonym invariance of the encoder's height-h node representation on clean world strings: one height-h
               node's subtree is redrawn from the same symbol (D_same) or a different one (D_diff); 1 - D_same/D_diff
               (1: the node codes only its symbol; Favero et al.'s relative sensitivity, inverted).
  reads        linear read-out (ridge) of the height-h symbol from the encoder's height-h node, fitted on nodes produced by
               half of each symbol's synonymous productions and tested on the other half (chance 1/v; 1/nc at the root).
Usage: python practice_rhm_unet.py --tag NAME --L 3 --steps 60000 -> /var/tmp/jbauer/generative_practice_rhm/unet/NAME/L3_P0.json"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from practice_rhm import RHM


class Dense(nn.Module):
    """W independent linear maps applied as one batched matmul: (W, ..., i) -> (W, ..., o). PyTorch's default init."""
    def __init__(self, W, i, o):
        super().__init__(); b = i ** -0.5
        self.w = nn.Parameter(torch.empty(W, i, o).uniform_(-b, b)); self.b = nn.Parameter(torch.empty(W, 1, o).uniform_(-b, b))

    def forward(self, x): return (torch.bmm(x.reshape(len(x), -1, x.shape[-1]), self.w) + self.b).reshape(*x.shape[:-1], -1)


class UNet(nn.Module):
    """W independent U-Nets, world-first layout. Input (W, n, d) tokens (v = mask); output (W, n, d, v) logits. A convolution
    with kernel = stride = s is a dense map of each s-tuple of positions; the transposed one maps a position to s children."""
    def __init__(self, v, s, L, C, W):
        super().__init__()
        self.v, self.s, self.L, self.C = v, s, L, C
        self.embed = Dense(W, v + 1, C)
        self.down = nn.ModuleList([nn.ModuleList([Dense(W, s * C, C), Dense(W, C, C)]) for _ in range(L)])
        self.up = nn.ModuleList([Dense(W, C, s * C) for _ in range(L)])
        self.merge = nn.ModuleList([nn.ModuleList([Dense(W, 2 * C, C), Dense(W, C, C)]) for _ in range(L)])
        self.out = Dense(W, C, v)

    def encode(self, x):
        """Encoder features per height 0..L, each (W, n, s^(L-h), C)."""
        W, n, d = x.shape; h = self.embed(F.one_hot(x, self.v + 1).float()); feats = [h]
        for a, b in self.down: h = F.gelu(b(F.gelu(a(h.reshape(W, n, -1, self.s * self.C))))); feats.append(h)
        return feats

    def forward(self, x):
        W, n, d = x.shape; feats = self.encode(x); h = feats[-1]
        for up, (a, b), sk in zip(self.up, self.merge, reversed(feats[:-1])):
            h = F.gelu(up(h)).reshape(W, n, -1, self.C); h = F.gelu(b(F.gelu(a(torch.cat([h, sk], -1)))))
        return self.out(h)


def masked_loss(net, X, v, gen):
    """Any-order autoregressive bound on -log p(x), nats per string, per world: (W,). One random mask per string."""
    W, n, d = X.shape; k = torch.randint(1, d + 1, (W, n), device=X.device, generator=gen)
    M = torch.rand(W, n, d, device=X.device, generator=gen).argsort(-1).argsort(-1) < k[..., None]
    ce = F.cross_entropy(net(torch.where(M, v, X)).reshape(-1, v), X.reshape(-1), reduction="none").reshape(W, n, d)
    return ((ce * M).sum(-1) * d / k).mean(1)


@torch.no_grad()
def generate(net, W, n, d, v, gen, device):
    X = torch.full((W, n, d), v, device=device); order = torch.rand(W, n, d, device=device, generator=gen).argsort(-1)
    for i in range(d):
        pos = order[..., i:i + 1]; p = torch.gather(net(X), 2, pos[..., None].expand(W, n, 1, v))[:, :, 0].softmax(-1)
        X.scatter_(2, pos, torch.multinomial(p.reshape(-1, v), 1, generator=gen).reshape(W, n, 1))
    return X.cpu().numpy()


def node_valid(rhm, x):
    """Share of height-h nodes whose subtree parses up to height h, per h = 1..L."""
    ok, a, out = np.ones(x.shape, bool), x, []
    for k in reversed(range(rhm.L)):
        p = rhm.parent[k][np.ravel_multi_index(a.reshape(len(a), -1, rhm.s).transpose(2, 0, 1), (rhm.v,) * rhm.s)]
        ok = ok.reshape(len(a), -1, rhm.s).all(2) & (p >= 0); out.append(float(ok.mean())); a = np.maximum(p, 0)
    return out


def subtree(rhm, a, k, rng):
    """Leaves of fresh subtrees grown from symbols a (n,) at tree level k (0 = root): (n, s^(L-k))."""
    a = a[:, None]
    for kk in range(k, rhm.L): a = rhm.rules[kk][a, rng.integers(rhm.m, size=a.shape)].reshape(len(a), -1)
    return a


def subtree_split(rhm, a, k, first, rng):
    """Like subtree, but the production at the top node is drawn from the index set `first`."""
    c = rhm.rules[k][a, np.asarray(first)[rng.integers(len(first), size=len(a))]]
    for kk in range(k + 1, rhm.L): c = rhm.rules[kk][c, rng.integers(rhm.m, size=c.shape)].reshape(len(a), -1)
    return c


@torch.no_grad()
def reads(net, rhms, X, rng, device, ridge=1e-2):
    """Linear read-out of the height-h symbol from the encoder's height-h node, tested on synonyms the read-out never saw:
    fitted (ridge, one-hot targets) on nodes whose own production is one of the first m/2, tested on the other m/2.
    Random features of an unseen tuple carry no trace of its symbol; only a code shared by synonyms transfers.
    X: (W, n, d) world strings into which the probed subtrees are written. Per world: test accuracy for h = 1..L."""
    W, n, d = X.shape; ar = np.arange(n); L, m = rhms[0].L, rhms[0].m; out = [[] for _ in range(W)]
    for h in range(1, L + 1):
        k = L - h; w_ = rhms[0].s ** h; nsym = rhms[0].nc if k == 0 else rhms[0].v; feats, labels = [], []
        for first in [range(m // 2), range(m // 2, m)]:
            Xp = X.copy(); J = np.zeros((W, n), int); A = np.zeros((W, n), int)
            for w, rhm in enumerate(rhms):
                j = rng.integers(rhm.s ** k, size=n); a = rng.integers(nsym, size=n); cols = j[:, None] * w_ + np.arange(w_)
                Xp[w, ar[:, None], cols] = subtree_split(rhm, a, k, list(first), rng); J[w], A[w] = j, a
            Fh = net.encode(torch.as_tensor(Xp, device=device))[h]; Jt = torch.as_tensor(J, device=device)
            feats.append(torch.gather(Fh, 2, Jt[:, :, None, None].expand(W, n, 1, Fh.shape[-1]))[:, :, 0].double()); labels.append(torch.as_tensor(A, device=device))
        mu = feats[0].mean(1, keepdim=True); Ftr, Fte = feats[0] - mu, feats[1] - mu; G = Ftr.transpose(1, 2) @ Ftr
        lam = ridge * G.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(G.shape[-1], device=device, dtype=G.dtype)
        B = torch.linalg.solve(G + lam, Ftr.transpose(1, 2) @ F.one_hot(labels[0], nsym).double())
        acc = ((Fte @ B).argmax(-1) == labels[1]).double().mean(1)
        for w in range(W): out[w].append(float(acc[w]))
    return out


@torch.no_grad()
def fills(net, rhms, X, rng, device):
    """X: (W, n, d) world strings. Per world: (mass on values valid up to h, the same for a uniform guess), h = 1..L."""
    W, n, d = X.shape; v = rhms[0].v; ar = np.arange(n); pos = rng.integers(d, size=(W, n)); Xm = X.copy()
    Xm[np.arange(W)[:, None], ar[None], pos] = v
    P = net(torch.as_tensor(Xm, device=device)).softmax(-1).cpu().numpy()[np.arange(W)[:, None], ar[None], pos]  # (W, n, v)
    out = []
    for w, rhm in enumerate(rhms):
        Xc = np.repeat(X[w, :, None], v, 1); Xc[ar[:, None], np.arange(v)[None], pos[w, :, None]] = np.arange(v)[None]
        ok = rhm.parses(Xc.reshape(-1, d)).reshape(n, v, rhm.L)
        out.append(([float(z) for z in (P[w, :, :, None] * ok).sum(1).mean(0)], [float(z) for z in ok.mean(1).mean(0)]))
    return out


@torch.no_grad()
def abstracts(net, rhms, trees, rng, device):
    """trees[w]: the world-w sample tree of the evaluation strings. Per world: 1 - D_same / D_diff for h = 1..L."""
    X = np.stack([t[-1] for t in trees]); W, n, d = X.shape; ar = np.arange(n); L = rhms[0].L
    enc = lambda Z: net.encode(torch.as_tensor(Z, device=device)); F0 = enc(X); out = [[] for _ in range(W)]
    for h in range(1, L + 1):
        k = L - h; w_ = rhms[0].s ** h; Xs, Xd = X.copy(), X.copy(); J = np.zeros((W, n), int)
        for w, (rhm, tree) in enumerate(zip(rhms, trees)):
            nsym = rhm.nc if k == 0 else rhm.v; j = rng.integers(rhm.s ** k, size=n); a = tree[k][ar, j]; a2 = (a + rng.integers(1, nsym, size=n)) % nsym
            cols = j[:, None] * w_ + np.arange(w_); Xs[w, ar[:, None], cols] = subtree(rhm, a, k, rng); Xd[w, ar[:, None], cols] = subtree(rhm, a2, k, rng); J[w] = j
        Jt = torch.as_tensor(J, device=device); node = lambda Fh: torch.gather(Fh, 2, Jt[:, :, None, None].expand(W, n, 1, Fh.shape[-1]))[:, :, 0]
        f0, fs, fd = node(F0[h]), node(enc(Xs)[h]), node(enc(Xd)[h])
        r = 1 - ((fs - f0) ** 2).sum(-1).mean(1) / ((fd - f0) ** 2).sum(-1).mean(1)
        for w in range(W): out[w].append(float(r[w]))
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--v", type=int, default=8); pa.add_argument("--nc", type=int, default=8); pa.add_argument("--m", type=int, default=4)
    pa.add_argument("--s", type=int, default=2); pa.add_argument("--L", type=int, default=3); pa.add_argument("--C", type=int, default=512)
    pa.add_argument("--P", type=int, default=0, help="training set size; 0 = fresh world samples at every step")
    pa.add_argument("--worlds", type=int, default=4); pa.add_argument("--steps", type=int, default=60000); pa.add_argument("--batch", type=int, default=128)
    pa.add_argument("--lr", type=float, default=1e-3); pa.add_argument("--n_points", type=int, default=30); pa.add_argument("--n_eval", type=int, default=2000)
    pa.add_argument("--seed", type=int, default=0); pa.add_argument("--out", type=Path, default=Path("/var/tmp/jbauer/generative_practice_rhm/unet")); pa.add_argument("--tag", required=True)
    args = pa.parse_args(); device = "cuda"; W, d = args.worlds, args.s ** args.L
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    f = args.out / args.tag / f"L{args.L}_P{args.P}.json"; f.parent.mkdir(parents=True, exist_ok=True); assert not f.exists(), f
    rhms = [RHM(args.v, args.nc, args.m, args.s, args.L, np.random.default_rng([args.seed, w, args.L])) for w in range(W)]
    rng = np.random.default_rng([args.seed, 99, args.L]); torch.manual_seed(args.seed); gen = torch.Generator(device=device).manual_seed(args.seed)
    sample = lambda n: np.stack([r.sample(n, rng)[-1] for r in rhms])
    net = UNet(args.v, args.s, args.L, args.C, W).to(device); opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    train = torch.as_tensor(sample(args.P), device=device) if args.P else None
    train_sets = [{r.tobytes() for r in train[w].cpu().numpy()} for w in range(W)] if args.P else None
    ev_trees = [r.sample(args.n_eval, rng) for r in rhms]; ev_np = np.stack([t[-1] for t in ev_trees]); ev_X = torch.as_tensor(ev_np, device=device)
    U = rng.integers(args.v, size=(100000, d))
    res = [dict(world=w, L=args.L, entropy=float(r.entropy()), chance_write=[float(z) for z in r.valid(U)], chance_node=node_valid(r, U), curve=[]) for w, r in enumerate(rhms)]
    evals = set(np.round(np.geomspace(1, args.steps, args.n_points)).astype(int).tolist()) | {0}; loss_sum, n_loss, buf = torch.zeros(W, device=device), 0, []
    for step in tqdm(range(args.steps + 1), desc=f"L{args.L} P{args.P}", mininterval=10):
        if step in evals:
            net.eval()
            with torch.no_grad(): nll = torch.stack([masked_loss(net, ev_X, args.v, gen) for _ in range(4)]).mean(0).cpu().numpy()
            Xg = generate(net, W, args.n_eval, d, args.v, gen, device); fl = fills(net, rhms, ev_np, rng, device); ab = abstracts(net, rhms, ev_trees, rng, device); rd = reads(net, rhms, ev_np, rng, device)
            tl = (loss_sum / n_loss).cpu().numpy() if n_loss else None
            for w, r in enumerate(rhms):
                rec = dict(step=step, nll=float(nll[w]), train_loss=None if tl is None else float(tl[w]), writes=[float(z) for z in r.valid(Xg[w])],
                           writes_node=node_valid(r, Xg[w]), fills=fl[w][0], fills_chance=fl[w][1], abstracts=ab[w], reads=rd[w])
                if args.P: rec["copies"] = float(np.mean([x.tobytes() in train_sets[w] for x in Xg[w]]))
                res[w]["curve"].append(rec)
            loss_sum, n_loss = torch.zeros(W, device=device), 0; net.train()
            json.dump(dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, results=res), open(f.with_suffix(".partial"), "w"))
        if step == args.steps: break
        if args.P: X = train[:, torch.randint(args.P, (args.batch,), device=device, generator=gen)]
        else:
            if not buf: buf = list(torch.as_tensor(sample(256 * args.batch), device=device).reshape(W, 256, args.batch, d).unbind(1))
            X = buf.pop()
        lw = masked_loss(net, X, args.v, gen); opt.zero_grad(); lw.sum().backward(); opt.step(); loss_sum += lw.detach(); n_loss += 1
    f.with_suffix(".partial").rename(f); print(f)


if __name__ == "__main__": main()
