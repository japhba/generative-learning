"""Does an own sample still show who wrote it after the first layer has read it back? Ring world of shared_visibles_pilot.py at several widths.

Slot i of S holds one of V = 8 units, each a token pair at ring positions (i, i + 1); neighboring slots share a token, the string shows one of
the two tokens written there. K deep units start on sharp random templates (one unit per slot, written with probability p). For own samples
the script counts the slots that are read back as the writer's template and whether the writer is the best-matching of the K units; for
teacher strings (uniform over the consistent strings, drawn with transfer matrices) the match with the best of the K units.

    python shared_visibles_width_check.py --out /nfs/nhome/live/jbauer/generative-learning/reports/shared-visibles-width-check.json
"""
import argparse
import json

import numpy as np
from tqdm.auto import tqdm

from shared_visibles_pilot import DEG, Q, V, regular01


def ring(rng, s):
    allowed = np.stack([regular01(rng) for _ in range(s)])
    left, right = (np.stack([np.nonzero(allowed[i])[k] for i in range(s)]) for k in (0, 1))
    unit_of = -np.ones((s, Q, Q), int)
    for i in range(s):
        unit_of[i, left[i], right[i]] = np.arange(V)
    return allowed, left, right, unit_of


def teacher_codes(rng, allowed, unit_of, n):
    """n uniform draws from the strings in which every neighboring token pair around the ring is allowed."""
    s = len(allowed)
    suf = [np.eye(Q)]
    for i in reversed(range(s)):
        suf.insert(0, allowed[i] @ suf[0])                                # suf[i] = A_i ... A_{s-1}
    draw = lambda w: (w.cumsum(1) < rng.random((n, 1)) * w.sum(1, keepdims=True)).sum(1)
    x = np.zeros((n, s), int)
    x[:, 0] = draw(np.tile(np.diag(suf[0]), (n, 1)))
    for i in range(s - 1):
        x[:, i + 1] = draw(allowed[i][x[:, i]] * suf[i + 1][:, x[:, 0]].T)
    codes = unit_of[np.arange(s), x, np.roll(x, -1, 1)]
    assert (codes >= 0).all()
    return codes


def readback(rng, left, right, unit_of, h):
    idx = np.arange(h.shape[1])
    lt, rt = left[idx, h], right[idx, h]
    x = np.where(rng.random(h.shape) < .5, np.roll(rt, 1, 1), lt)
    return unit_of[idx, x, np.roll(x, -1, 1)]


def matches(code, tmpl):
    """[B, K] number of slots in which the code equals each template (a silent slot, -1, matches nothing)."""
    return np.concatenate([(c[:, None] == tmpl[None]).sum(-1) for c in np.array_split(code, max(1, len(code) // 128))])


def one(seed, s, k, p, repeats):
    rng = np.random.default_rng([seed, s, k])
    allowed, left, right, unit_of = ring(rng, s)
    tmpl = rng.integers(V, size=(k, s))
    writer = np.tile(np.arange(k), repeats)
    h = np.where(rng.random((len(writer), s)) < p, tmpl[writer], (tmpl[writer] + rng.integers(1, V, (len(writer), s))) % V)
    code = readback(rng, left, right, unit_of, h)
    m = matches(code, tmpl)
    own = m[np.arange(len(writer)), writer]
    m[np.arange(len(writer)), writer] = -1
    accepted, as_written = (code >= 0).all(1), (code == h).all(1)
    teach_best = matches(teacher_codes(rng, allowed, unit_of, 2048), tmpl).max(1)
    grid = np.arange(s + 1)
    own_pass, teach_pass = (own[:, None] >= grid).mean(0), (teach_best[:, None] >= grid).mean(0)
    best = int(np.argmax(own_pass - teach_pass))
    return dict(slots=s, units=k, own_match=float(own.mean() / s), own_read=float((code >= 0).mean()), own_as_written=float((code == h).mean()),
                writer_is_best=float((own > m.max(1)).mean()), teacher_best=float(teach_best.mean() / s), own_above_teacher=float((own[:, None] > teach_best[None, :256]).mean()),
                threshold=best / s, own_pass=float(own_pass[best]), teacher_pass=float(teach_pass[best]),
                accepted=float(accepted.mean()), accepted_match=float(own[accepted].mean() / s) if accepted.any() else None, accepted_writer_is_best=float((own > m.max(1))[accepted].mean()) if accepted.any() else None,
                consistent=float(as_written.mean()), consistent_match=float(own[as_written].mean() / s) if as_written.any() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--worlds', type=int, default=16)
    ap.add_argument('--scale', type=float, default=4.)
    args = ap.parse_args()
    p = float(np.exp(args.scale) / (np.exp(args.scale) + V - 1))
    rows = []
    for s, k, repeats in tqdm([(8, 32, 2048), (8, 1024, 256), (16, 1024, 16), (32, 1024, 4), (64, 1024, 2), (128, 1024, 1)]):
        runs = [one(seed, s, k, p, repeats) for seed in tqdm(range(args.worlds), leave=False)]
        rows.append({key: (None if any(r[key] is None for r in runs) else float(np.median([r[key] for r in runs]))) for key in runs[0]})
        print(json.dumps(rows[-1]), flush=True)
    json.dump(dict(p=p, values=Q, successors=DEG, units_per_slot=V, worlds=args.worlds, rows=rows), open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
