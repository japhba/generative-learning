"""Exact check of the practice identity on the order-m world.

World: m blocks of q symbols, valid iff the symbols sum to 0 mod q (S, |S| = q^(m-1)); teacher uniform on S; every m - 1 blocks uniform.
Learner: one winner-take-all group, K states, uniform prior, writer table G[k, b, s] = p(symbol s in block b | state k); rule
P <- P + eps * credit * (onehot - P) at the credited state.

Expected update per sample (exact, by enumeration of all q^m strings):
  practice  : state k drawn from the prior, string from G[k]; credit = the writing state; accepted (valid) samples only, or signed
              (+1 valid, -1 refused, refused samples move P away from onehot).
  teacher   : string from the teacher; credit = posterior of a reader table R over states, R = G (exact posterior) or another table.
  foreign   : string from another writer table F, credit = posterior of R, accepted only.
Identity: practice = sum over S of grad p(x) = p(S) * E_{p(.|S)}[score]; teacher with the exact posterior = E_{U_S}[score]; equal up to
the factor p(S) when p is flat on S.

    python practice_identity.py --out /var/tmp/jbauer/practice_identity/exact.json
"""
import argparse, itertools, json
import numpy as np

sm = lambda u: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(u - u.max(-1, keepdims=True)))


def world(q, m):
    S = np.array(list(itertools.product(range(q), repeat=m)))
    return S, S.sum(1) % q == 0


def p_given_state(T, S):
    """p(x | k) for every state and string [K, N]."""
    m = S.shape[1]
    return np.exp(np.log(T)[:, np.arange(m)[None, :], S].sum(2))


def hebb_field(T, S, weight):
    """expected update of table T given weight[k, n] = probability of (credited state k, string n): sum_n weight * (onehot - T)."""
    q = T.shape[-1]
    H = np.eye(q)[S]                                                                              # [N, m, q]
    return np.einsum('kn,nbs->kbs', weight, H) - weight.sum(1)[:, None, None] * T


def updates(G, S, valid, R=None, F=None):
    K = G.shape[0]
    pk = p_given_state(G, S) / K                                                                   # joint p(k, x) of the writer
    px = pk.sum(0)
    out = dict(pS=float(px[valid].sum()))
    out['practice'] = hebb_field(G, S, pk * valid)                                                 # accepted own samples, writing state
    out['practice_signed'] = hebb_field(G, S, pk * np.where(valid, 1., -1.))
    post = lambda T: sm(np.log(p_given_state(T, S) + 1e-300).T).T                                 # [K, N] posterior of a reader table T
    pT = valid / valid.sum()
    out['teacher_exact'] = hebb_field(G, S, post(G) * pT)
    if R is not None:
        out['teacher_reader'] = hebb_field(G, S, post(R) * pT)
    if F is not None:
        pf = (p_given_state(F, S) / K).sum(0)
        out['foreign_exact'] = hebb_field(G, S, post(G) * pf * valid)
        if R is not None:
            out['foreign_reader'] = hebb_field(G, S, post(R) * pf * valid)
    out['flatness'] = float(np.std(px[valid]) / np.mean(px[valid]))                              # coefficient of variation of p on S
    return out


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def flat_on_S(rng, q, m, K, a):
    """a writer that is flat on S by construction: K = |S| states, each state a tilted product around its own valid string; the
    assignment of states to strings is a bijection, so every valid string has the same probability."""
    S, valid = world(q, m)
    Sv = S[valid]
    base = np.full((K, m, q), 1.)
    for k in range(K):
        base[k, np.arange(m), Sv[k]] += a
    return base / base.sum(-1, keepdims=True)


def main(out):
    rng = np.random.default_rng(0)
    res = dict(flat=[], random=[], reader=[])
    for q, m in ((2, 3), (2, 4), (2, 5), (3, 3), (3, 4)):
        S, valid = world(q, m)
        K = q ** (m - 1)
        for a in (0.01, 0.1, 0.5, 1., 2., 4.):                                                     # flat writers of growing sharpness
            G = flat_on_S(rng, q, m, K, a)
            u = updates(G, S, valid)
            ratio = np.linalg.norm(u['practice']) / np.linalg.norm(u['teacher_exact'])
            res['flat'].append(dict(q=q, m=m, a=a, pS=u['pS'], flatness=u['flatness'], ratio=float(ratio), cos=cos(u['practice'], u['teacher_exact'])))
        for a in (0.01, 0.1, 0.3, 1., 3.):                                                         # random tilted writers (not flat on S)
            for s in range(8):
                G = sm(a * rng.normal(size=(K, m, q)))
                R_diffuse = sm(1e-3 * rng.normal(size=(K, m, q)))
                F = sm(a * rng.normal(size=(K, m, q)))                                              # a twin's writer
                u = updates(G, S, valid, R=R_diffuse, F=F)
                n = lambda x: float(np.linalg.norm(x))
                res['random'].append(dict(q=q, m=m, a=a, seed=s, pS=u['pS'], flatness=u['flatness'],
                                          ratio=n(u['practice']) / n(u['teacher_exact']), cos=cos(u['practice'], u['teacher_exact']),
                                          cos_signed=cos(u['practice_signed'], u['teacher_exact']),
                                          reader_diffuse=n(u['teacher_reader']) / n(u['teacher_exact']),
                                          foreign_exact=n(u['foreign_exact']) / n(u['teacher_exact']), cos_foreign=cos(u['foreign_exact'], u['teacher_exact'])))
    json.dump(res, open(out, 'w'), indent=1)
    print('flat writers: |practice| / |teacher, exact posterior| against p(S), and their cosine')
    for r in res['flat']:
        print(f"  q {r['q']} m {r['m']} a {r['a']:5.2f}  p(S) {r['pS']:.4f}  ratio {r['ratio']:.4f}  cos {r['cos']:.6f}  flatness {r['flatness']:.1e}")
    print('random writers (median over 8): ratio / p(S), cosine practice vs teacher, flatness on S, diffuse reader / exact')
    for q, m in ((2, 3), (2, 4), (2, 5), (3, 3), (3, 4)):
        for a in (0.01, 0.1, 0.3, 1., 3.):
            rs = [r for r in res['random'] if (r['q'], r['m'], r['a']) == (q, m, a)]
            med = lambda k: np.median([r[k] for r in rs])
            print(f"  q {q} m {m} a {a:4.2f}  ratio/p(S) {np.median([r['ratio'] / r['pS'] for r in rs]):6.3f}  cos {med('cos'):.4f}  cos signed {med('cos_signed'):.4f}  flatness {med('flatness'):.2e}  diffuse reader {med('reader_diffuse'):.2e}  twin(exact)/teacher {med('foreign_exact'):.3f} cos {med('cos_foreign'):.3f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='/var/tmp/jbauer/practice_identity/exact.json')
    main(ap.parse_args().out)
