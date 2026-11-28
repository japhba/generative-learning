"""Are the states that watching with the ideal reader leaves behind traps for teacher strings that practice can leave?
Tied world of tied_binding.py, level 2 frozen. Phase 1: 1M samples of watching with the writer's exact posterior as credit. Phase 2, from the
same net: another 1M samples of the same watching, or of practice (half own samples: sleep with the writing state + accepted samples move the
level-1 writer), or of sleep only. Reports KL and binding of the level-1 reader at the fork and at the end, and the expected drives at the fork:
the exact-posterior teacher field on the level-1 writer and the practice field (accepted own samples, writing state), by enumeration.

    python tied_binding_trap.py --out /var/tmp/jbauer/tied_binding/trap.json
"""
import argparse, copy, json
from multiprocessing import Pool
import numpy as np
import tied_binding as tb


def step(net, w, rng, mode, batch, eps):
    if mode == 'exact':
        x = w.teacher(rng, batch)
        net.seen[np.arange(w.C)[None, :].repeat(batch, 0), x[:, w.checks[:, 0]], x[:, w.checks[:, 1]], x[:, w.checks[:, 2]]] = True
        k2, k1 = net.exact_post(x, rng)
        net.R1, net.G1 = tb.hebb(net.R1, k1, x, eps), tb.hebb(net.G1, k1, x, eps)
    elif mode == 'teacher':
        x = w.teacher(rng, batch)
        k1, _ = net.read1(x, rng)
        net.R1, net.G1 = tb.hebb(net.R1, k1, x, eps), tb.hebb(net.G1, k1, x, eps)
    else:                                                                                       # own sample
        k2, k1, x = net.write(rng, batch)
        V = (net.seen[np.arange(w.C)[None, :], x[:, w.checks[:, 0]], x[:, w.checks[:, 1]], x[:, w.checks[:, 2]]]).all(1)
        net.R1 = tb.hebb(net.R1, k1, x, eps)
        if mode == 'own+verdict' and V.any():
            net.G1 = tb.hebb(net.G1, k1[V], x[V], eps)


def fields(net, w, X, valid):
    """expected updates of the level-1 writer per sample: exact-posterior teacher field and accepted-own-sample field (writing state)."""
    l1, l2 = net.lift(net.G1), net.lift(net.G2)
    import itertools
    J1 = np.array(list(itertools.product(range(net.K1), repeat=2)))
    lp1 = np.log(tb.sm(sum(l2[g, np.zeros(1, int)] for g in range(1))))                          # level 2 frozen and diffuse: any state
    p1 = np.exp(lp1[0][0][J1[:, 0]] + lp1[0][1][J1[:, 1]])                                    # [J1]
    lp0 = np.log(tb.sm(sum(l1[g, J1[:, g]] for g in range(2))))
    pxk = np.exp(lp0[:, np.arange(w.B)[None, :], X].sum(2))                                     # [J1, N]
    joint = p1[:, None] * pxk                                                                  # p(k1, x)
    px = joint.sum(0)
    pT = valid / valid.sum()
    post = joint / np.maximum(px, 1e-300)
    H = np.eye(w.q)[X]
    out = {}
    for name, wt in (('teacher_exact', post * pT[None]), ('practice', joint * valid[None])):
        F = np.zeros_like(net.G1)
        for g in range(2):
            W = np.zeros((net.K1, len(X)))
            np.add.at(W, J1[:, g], wt)
            F[g] = np.einsum('kn,nbs->kbs', W, H) - W.sum(1)[:, None, None] * net.G1[g]
        out[name] = F
    return out, float(px[valid].sum())


def run(job):
    seed, n1, n2, eps = job
    rng = np.random.default_rng([seed, 9])
    w = tb.World(2, 3, True); X = w.all_strings(); valid = w.valid(X)
    net = tb.Net(np.random.default_rng([seed, 1]), w, 3, .1)
    for i in range(int(n1 / 64)):
        step(net, w, rng, 'exact', 64, eps)
    ev = lambda m: (lambda e: dict(kl=e['kl'], bind=e['bind'], valid=e['valid']))(tb.evaluate(m, w, X, valid, rng))
    fork = ev(net)
    F, pS = fields(net, w, X, valid)
    n = lambda a: float(np.linalg.norm(a))
    drive = dict(teacher_exact=n(F['teacher_exact']), practice=n(F['practice']), pS=pS,
                 cos=float((F['teacher_exact'] * F['practice']).sum() / (n(F['teacher_exact']) * n(F['practice']) + 1e-300)))
    out = dict(seed=seed, fork=fork, drive=drive)
    for cont, schedule in (('more exact watching', ['exact']), ('practice (half own, verdict)', ['exact', 'own+verdict']), ('sleep only (half own)', ['exact', 'own']),
                           ('practice, teacher read by the reader', ['teacher', 'own+verdict'])):
        m, r = copy.deepcopy(net), np.random.default_rng([seed, 10])
        for i in range(int(n2 / 64)):
            step(m, w, r, schedule[i % len(schedule)], 64, eps)
        out[cont] = ev(m)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='/var/tmp/jbauer/tied_binding/trap.json'); ap.add_argument('--seeds', type=int, default=16)
    a = ap.parse_args()
    with Pool(16) as pool:
        res = pool.map(run, [(s, 1e6, 1e6, .003) for s in range(a.seeds)])
    json.dump(res, open(a.out, 'w'), indent=1)
    print('fork = after 1M samples of watching with the exact posterior (level 2 frozen; best possible KL without level 2 = log 3 = 1.099)')
    for r in res:
        f, d = r['fork'], r['drive']
        print(f"seed {r['seed']:2d} fork KL {f['kl']:.3f} bind {f['bind']:.2f} | drives teacher {d['teacher_exact']:.2e} practice {d['practice']:.2e} (p(S) {d['pS']:.3f}, cos {d['cos']:.2f}) | " +
              '  '.join(f"{k.split(' (')[0]}: {r[k]['kl']:.3f}/{r[k]['bind']:.2f}" for k in r if k not in ('seed', 'fork', 'drive')))
