"""Build toy_model_2_WS.html: the two-layer wake-sleep toy on the two-layer ('deep') world. How practice turns a rotated
latent space, how recognition follows generation layer by layer, what a learned second layer buys, and, at a fixed budget
of examples, how much of it to spend on dreams: against the autoencoder baseline, with scarce data, over the size of the
latent space, and with dreams from other sources.

Every number in the page is computed here from the saved runs (practice_ws_layers.py, practice_ws_budget.py,
practice_disentangle_dreams.py, practice_disentangle_prior.py) and the two-cause simulation (toy_model_2_ws_sim.js, run
under node, the same code the page animates); nothing is typed in by hand. The page goes to the world-readable ceph reports
folder with copies of the code it describes, and is symlinked into reports/ for the local 127.0.0.1:8095 server.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from zoneinfo import ZoneInfo

from report_katex import stylesheet, typeset

SHARED = Path('/ceph/scratch/jbauer/reports/generative-learning')
PROJECT = 'tex/practice-loop'  # in the clone of the OverLyX project jan/generative-learning
NAME = 'toy_model_2_WS'
PRIOR = 'deep'  # the world: two top causes set the probabilities of six causes
ROOT = Path('/var/tmp/jbauer/generative_practice_disentangle')
LAYERS, FRESH = ROOT / 'layers/run_N256_20261025_221256', ROOT / 'layers/run_N0_20261025_221256'
PRIOR_RUN = ROOT / 'prior/deep_N256_rotated_hebb_20261025_111826/results.json'  # practice_disentangle_prior.py, label identity
BUDGET = ROOT / 'budget'
B_MAIN, B_AE = BUDGET / 'run_20261105_001836', BUDGET / 'run_ae_20261105_135239'  # fresh supplied examples: ws + ws_teach, ae
B_DATA = [BUDGET / 'run_dataN_ws_20261105_190201', BUDGET / 'run_dataN_base_20261105_191420']  # N distinct examples, reused
B_DREAMS = BUDGET / 'run_dreams_20261105_171433'
DREAMS_ONE = ROOT / 'dreams/run_20261105_172152/results.json'  # practice_disentangle_dreams.py: one layer, fixed true prior
AE_DEEP, AE_FACT, AE_N128 = (ROOT / 'ae_floor' / r for r in ['run_N0_20261106_191817', 'run_N0_20261106_192831', 'run_N128_20261106_193636'])  # practice_ws_ae_floor.py
DEPTH = ROOT / 'depth/run_N0_20261106_191756'  # practice_ws_depth.py
AE_SPECS = ['ws:x=0.7', 'ws:x=1', 'ae:top=mf:s=0.05', 'ae:top=mf:s=1', 'ae:top=bin:s=1', 'ae:top=mf:s=1:wd=0.01', 'vae:s=1']  # learners of section 8, in colour order
CONDS = ['two_layer', 'true_prior', 'independent']  # learners of sections 3-4, in colour order
LINES = [('ws', '0.7'), ('ws', '1.0'), ('ae', '1.0'), ('ws_teach', '1.0')]  # learners of sections 5-6
CODE = ['experiments/practice_ws_layers.py', 'experiments/practice_ws_budget.py', 'experiments/toy_model_2_ws_sim.js', 'experiments/practice_disentangle.py',
        'experiments/practice_disentangle_prior.py', 'experiments/practice_disentangle_dreams.py', 'experiments/practice_ws_ae_floor.py', 'experiments/practice_ws_depth.py',
        'tex/practice_disentangle.tex']

# The two-cause numbers, from the simulation the page runs (16 seeds, start 30 degrees, sigma 0.5).
NODE = r"""
const WS = require(process.env.SIM), S = 16, T = 3000, run = (upd, s) => {
  const w = WS.world(1000 + s, 0.5), L = WS.learner(Math.PI / 6, 0.5), r = WS.rng(5000 + s), tr = [];
  for (let t = 0; t <= T; t++) { tr.push([WS.genAngle(L), WS.recAngle(L)]); if (t < T) upd(L, w, r); }
  return tr; };
const mean = a => a.reduce((x, y) => x + y, 0) / a.length, runs = upd => Array.from({length: S}, (_, s) => run(upd, s));
const b = runs((L, w, r) => WS.step(L, w, r)), f = runs((L, w, r) => WS.step(L, w, r, {frozen: true})), at = (rs, t) => rs.map(tr => Math.abs(tr[t][0]));
const a = runs((L, w, r) => WS.aeStep(L, w, r)), a1 = runs((L, w, r) => WS.aeStep(L, w, r, {rate: 1}));
process.stdout.write(JSON.stringify({
  b250: mean(at(b, 250)), b500: mean(at(b, 500)), b1000: mean(at(b, 1000)), b3000: mean(at(b, T)), lag250: mean(b.map(tr => tr[250][1] - tr[250][0])),
  frozen: mean(at(f, T)), frozenMin: Math.min(...at(f, T)), frozenMax: Math.max(...at(f, T)), land: WS.landscape(0.5, [15, 30, 45]),
  ae500: mean(at(a, 500)), ae1000: mean(at(a, 1000)), ae3000: mean(at(a, T)), aeMin: Math.min(...at(a, T)), aeMax: Math.max(...at(a, T)), aeF500: mean(at(a1, 500)), aeF3000: mean(at(a1, T)),
}));
"""


def band(v): v = np.asarray(v, float); return {'mean': v.mean(0), 'sem': v.std(0, ddof=1) / np.sqrt(len(v))}


def bootstrap(diff, rng): b = rng.choice(diff, (10000, len(diff))).mean(1); return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def budget_curves(*runs, key='kl'):
    """(learner, x) -> array (world, checkpoint) of one key, and the checkpoint totals, merged over runs (fresh examples)."""
    out = {}
    for run in runs:
        for learner, byx in json.loads((run / 'curves.json').read_text())['curves'][PRIOR].items():
            for x, ws in byx.items(): out[learner, x] = np.array([[p[key] for p in ws[w]] for w in sorted(ws, key=int)])
    P = np.array([p['P'] for p in json.loads((runs[0] / 'curves.json').read_text())['curves'][PRIOR]['ws']['0.7']['0']])
    return out, P


def capped(runs):
    """(N, learner, x) -> list over worlds of curves, for runs with --Ns."""
    out = {}
    for run in runs:
        for f in run.glob('N*/curves.json'):
            j = json.loads(f.read_text())
            for learner, byx in j['curves'][PRIOR].items():
                for x, ws in byx.items(): out[j['args']['N'], learner, x] = [ws[w] for w in sorted(ws, key=int)]
    return out


def main():
    args = argparse.ArgumentParser(description=__doc__); args.add_argument('--project', action='store_true', help=f'also write the page into {PROJECT}/'); args = args.parse_args()
    root = Path(__file__).resolve().parents[1]; rng = np.random.default_rng(0)
    deg, nat, pct = (lambda v: f'{v:.0f}°'), (lambda v: f'{v:.2f}'), (lambda v: f'{100 * v:.0f}%')
    pow2 = lambda P: f'2<sup>{int(np.log2(P))}</sup>'
    nums = {}

    # ---- Sections 3-4: practice on 256 stored samples, one wake and one sleep batch per update ----
    main_c = json.loads((LAYERS / 'curves.json').read_text()); run_args = main_c['args']
    curves, fresh = main_c['curves'][PRIOR], json.loads((FRESH / 'curves.json').read_text())['curves'][PRIOR]
    end = lambda c, cond, k: float(np.mean([r[-1][k] for r in c[cond]]))
    start = lambda c, cond, k: float(np.mean([r[0][k] for r in c[cond]]))
    nums |= {'worlds': str(run_args['worlds']), 'N': str(run_args['N']), 'H': str(run_args['H']), 'steps': f"{run_args['steps']:,}"}
    for cond in CONDS + ['two_layer_h2']:
        nums |= {f'{cond}_kl0': nat(start(curves, cond, 'kl')), f'{cond}_kl': nat(end(curves, cond, 'kl')), f'{cond}_gen0': deg(start(curves, cond, 'gen_world')), f'{cond}_gen': deg(end(curves, cond, 'gen_world')),
                 f'{cond}_lag': f"{end(curves, cond, 'rec_lag'):.1f}°", f'{cond}_corr': nat(end(curves, cond, 'corr_err')), f'{cond}_acc0': pct(start(curves, cond, 'acc_matched')), f'{cond}_acc': pct(end(curves, cond, 'acc_matched'))}
    for cond in ['two_layer', 'true_prior']: nums |= {f'{cond}_fkl': nat(end(fresh, cond, 'kl')), f'{cond}_fgen': deg(end(fresh, cond, 'gen_world'))}
    two = curves['two_layer']; assert [p['step'] for p in two[0][:4]] == [0, 200, 400, 600]
    nums |= {'two_layer_top': f"{end(curves, 'two_layer', 'top_lag'):.1f}°", 'lag200': deg(np.mean([r[1]['rec_lag'] for r in two])), 'toplag600': deg(np.mean([r[3]['top_lag'] for r in two])),
             'top_world0': deg(start(curves, 'two_layer', 'top_world')), 'top_world': deg(end(curves, 'two_layer', 'top_world')),
             'own0': pct(start(curves, 'two_layer', 'acc_own')), 'own': pct(end(curves, 'two_layer', 'acc_own')),
             'two_range': f"{min(r[-1]['gen_world'] for r in two):.0f}–{max(r[-1]['gen_world'] for r in two):.0f}°"}
    nums |= {'gauss_gen0': deg(start(curves, 'gaussian', 'gen_world')), 'gauss_gen': deg(end(curves, 'gaussian', 'gen_world'))}  # the linear-Gaussian twin, quoted once
    lag_all = [r[-1]['rec_lag'] for c in CONDS for r in curves[c]]; nums |= {'lagMin': f'{min(lag_all):.0f}°', 'lagMax': f'{max(lag_all):.0f}°'}
    pr = json.loads(PRIOR_RUN.read_text())['results']; sym = lambda c: pct(np.mean([r[-1]['acc_sym'] for r in pr[c]]))
    nums |= {'priorWorlds': str(len(pr['top_layer'])), 'priorTopSym': sym('top_layer'), 'priorFixedSym': sym('fixed_true')}
    steps = np.array([p['step'] for p in two[0]])
    layers = {'steps': steps, 'summary': {c: {k: band([[p[k] for p in r] for r in curves[c]]) for k in ['kl', 'gen_world', 'rec_lag', 'corr_err'] + (['top_lag'] if c == 'two_layer' else [])} for c in CONDS}}

    # ---- Section 2: the two-cause simulation ----
    sim = root / 'experiments/toy_model_2_ws_sim.js'
    t2 = json.loads(subprocess.run(['node', '-e', NODE], capture_output=True, text=True, check=True, env={'SIM': str(sim), 'PATH': '/usr/bin:/bin:' + str(Path(shutil.which('node')).parent)}).stdout)
    nums |= {'b250': deg(t2['b250']), 'b500': deg(t2['b500']), 'b1000': f"{t2['b1000']:.1f}°", 'b3000': f"{t2['b3000']:.1f}°", 'lag250': f"{t2['lag250']:.1f}°",
             'frozen': deg(t2['frozen']), 'frozenRange': f"{t2['frozenMin']:.0f}–{t2['frozenMax']:.0f}°", 'land15': f"{-t2['land'][0]:.2f}", 'land30': f"{-t2['land'][1]:.2f}", 'land45': f"{-t2['land'][2]:.2f}",
             'ae500': deg(t2['ae500']), 'ae1000': deg(t2['ae1000']), 'ae3000': f"{t2['ae3000']:.1f}°", 'aeRange': f"{t2['aeMin']:.0f}–{t2['aeMax']:.0f}°", 'aeF500': f"{t2['aeF500']:.1f}°", 'aeF3000': f"{t2['aeF3000']:.1f}°"}

    # ---- Section 5: the budget (fresh supplied examples, each used once) ----
    kl, P = budget_curves(B_MAIN, B_AE); xs = json.loads((B_MAIN / 'curves.json').read_text())['xs']; xk = [str(x) for x in xs]
    assert all(('ws', k) in kl and ('ae', k) in kl for k in xk); bargs = json.loads((B_MAIN / 'curves.json').read_text())['args']
    col = lambda Pv: int(np.flatnonzero(P == Pv)[0]); rows = P >= 2 ** 8
    heat = {l: np.array([kl[l, k].mean(0) for k in xk]).T[rows] for l in ['ws', 'ae']}  # (total, fraction)
    best = {l: heat[l].argmin(1) for l in heat}
    nums |= {'bWorlds': str(len(kl['ws', '0.7'])), 'bMax': pow2(P[-1]), 'bStart': nat(kl['ws', '0.7'][:, 0].mean()), 'batch': str(bargs['batch'])}
    for Pv in [2 ** 13, 2 ** 16, 2 ** 19, 2 ** 21]:
        c, e = col(Pv), int(np.log2(Pv)); f = lambda l, x: kl[l, x][:, c]
        nums |= {f'b{e}_{l}{round(100 * float(x))}': nat(f(l, x).mean()) for l, x in LINES + [('ws', '0.5'), ('ae', '0.9')]}
        nums |= {f'b{e}_winsAE': str(int((f('ws', '0.7') < f('ae', '1.0')).sum())), f'b{e}_winsNoGen': str(int((f('ws', '0.7') < f('ws', '1.0')).sum())), f'b{e}_winsHalf': str(int((f('ws', '0.5') < f('ws', '1.0')).sum())),
                 f'b{e}_teachWins': str(int((f('ws_teach', '1.0') < f('ws', '0.7')).sum())), f'b{e}_ratioTeach': f"{f('ws', '0.7').mean() / f('ws_teach', '1.0').mean():.1f}",
                 f'b{e}_ratioAE': f"{f('ae', '1.0').mean() / f('ws', '0.7').mean():.0f}"}
    sel = rows & (P >= 2 ** 13)
    nums |= {'stall': nat(kl['ws', '1.0'][:, sel].mean()), 'aeFloor': f"{kl['ae', '1.0'][:, P >= 2 ** 16].mean(0).min():.2f}–{kl['ae', '1.0'][:, P >= 2 ** 16].mean(0).max():.2f}",
             'bestWS': ' or '.join(pct(x) for x in sorted({xs[i] for i, Pv in zip(best['ws'], P[rows]) if Pv >= 2 ** 13})), 'bestWSsmall': '–'.join(pct(x) for x in [min(xs[i] for i, Pv in zip(best['ws'], P[rows]) if Pv <= 2 ** 12), 1.0]),
             'b21_ws99': nat(kl['ws', '0.99'][:, -1].mean()), 'b21_ws95': nat(kl['ws', '0.95'][:, -1].mean()),
             'aeSleepBest': nat(heat['ae'][-1].min()), 'aeSleepBestX': pct(xs[best['ae'][-1]]),
             'spread21': f"{np.ptp([kl['ws', k][:, -1].mean() for k in ['0.5', '0.6', '0.7', '0.8', '0.85', '0.9']]):.2f}"}
    beat = [Pv for Pv in P[P >= 2 ** 8] if all((kl['ws', '0.7'][:, col(q)] < kl['ae', '1.0'][:, col(q)]).sum() >= 11 for q in P if q >= Pv)]
    nums['beatFrom'] = pow2(min(beat)); nums['beatFromPlain'] = f'{min(beat):,}'
    budget = {'xs': xs, 'rows': P[rows], 'heat': heat, 'best': best, 'P': P[1:], 'lines': {f'{l}@{x}': band(kl[l, x][:, 1:]) for l, x in LINES}}

    # ---- Section 6: scarce data (N distinct supplied examples, reused; total fixed) ----
    cap = capped(B_DATA); Ns = sorted({k[0] for k in cap}); final = lambda N, l, x, k='kl': np.array([c[-1][k] for c in cap[N, l, x]])
    fresh_final = {f'{l}@{x}': float(kl[l, x][:, -1].mean()) for l, x in LINES}
    nums |= {'Ns': ', '.join(str(N) for N in Ns), 'capTotal': pow2(json.loads(next(B_DATA[0].glob('N*/curves.json')).read_text())['args']['P_max'])}
    for N in [32, 128, 2048]:
        nums |= {f'n{N}_{l}{round(100 * float(x))}': nat(final(N, l, x).mean()) for l, x in LINES}
        nums |= {f'n{N}_seen': f"{final(N, 'ws', '0.7', 'configs_seen').mean():.0f}", f'n{N}_mass': pct(final(N, 'ws', '0.7', 'mass_seen').mean()),
                 f'n{N}_winsAE': str(int((final(N, 'ws', '0.7') < final(N, 'ae', '1.0')).sum())), f'n{N}_winsNoGen': str(int((final(N, 'ws', '0.7') < final(N, 'ws', '1.0')).sum())),
                 f'n{N}_gap': nat((final(N, 'ws', '0.7') - final(N, 'ws', '0.7', 'kl_train')).mean())}
    nums['n2048_vsFresh'] = nat(final(2048, 'ws', '0.7').mean() - fresh_final['ws@0.7'])
    n128 = cap[128, 'ws', '0.7'][0]; Pc = np.array([p['P'] for p in n128])
    scarce = {'Ns': Ns, 'final': {f'{l}@{x}': [float(final(N, l, x).mean()) for N in Ns] for l, x in LINES}, 'fresh': fresh_final,
              'P': Pc[1:], 'curves': {f'{l}@{x}': band(np.array([[p['kl'] for p in c] for c in cap[128, l, x]])[:, 1:]) for l, x in LINES}}

    # ---- Section 6: the threshold over the size of the latent space (N reused, total 2^19) ----
    kd = {}
    for run in sorted(BUDGET.glob('run_dataK*')):
        for f in run.glob('N*/curves.json'):
            j = json.loads(f.read_text()); K, N = j['args']['K'], j['args']['N']
            for l, byx in j['curves'][PRIOR].items():
                for x, ws in byx.items(): kd[K, N, l, x] = [ws[w][-1] for w in sorted(ws, key=int)]
            kTotal = j['args']['P_max']
    Ks = sorted({k[0] for k in kd}); kN = sorted({k[1] for k in kd}); kv = lambda K, N, l, x, k='kl': np.array([c[k] for c in kd[K, N, l, x]])
    kplot, cross, aeThr = {}, {}, {}
    for K in Ks:
        adv = [kv(K, N, 'ws', '1.0') - kv(K, N, 'ws', '0.7') for N in kN]; ci = [bootstrap(a, rng) for a in adv]; m = np.array([a.mean() for a in adv])
        kplot[K] = {'mean': m, 'lo': [c[0] for c in ci], 'hi': [c[1] for c in ci], 'seen': [float(kv(K, N, 'ws', '0.7', 'configs_seen').mean()) for N in kN]}
        i = int(np.flatnonzero((m[:-1] < 0) & (m[1:] >= 0))[-1]); t = -m[i] / (m[i + 1] - m[i])  # last upward zero crossing, interpolated in log N
        cross[K] = float(np.exp(np.log(kN[i]) + t * (np.log(kN[i + 1]) - np.log(kN[i]))))
        aci = [bootstrap(kv(K, N, 'ae', '1.0') - kv(K, N, 'ws', '0.7'), rng) for N in kN]
        aeThr[K] = min(N for j, N in enumerate(kN) if all(c[0] > 0 for c in aci[j:]))
    nums |= {'Ks': ', '.join(map(str, Ks)), 'kConfigs': ', '.join(str(2 ** K) for K in Ks), 'kTotal': pow2(kTotal), 'kWorlds': str(len(kd[Ks[0], kN[0], 'ws', '0.7'])),
             'crossRange': f'{min(cross.values()):.0f}–{max(cross.values()):.0f}', 'crossList': ', '.join(f'{cross[K]:.0f}' for K in Ks),
             'aeThrList': ', '.join(str(aeThr[K]) for K in Ks), 'k8seen48': f"{kv(8, 48, 'ws', '0.7', 'configs_seen').mean():.0f}"}
    kth = {'N': kN, 'K': kplot}

    # ---- Section 7: where the dreams come from ----
    dk, Pd = budget_curves(B_DREAMS); dk_ent, _ = budget_curves(B_DREAMS, key='dream_latent_entropy')
    for Pv in [2 ** 16, 2 ** 21]:
        c, e = int(np.flatnonzero(Pd == Pv)[0]), int(np.log2(Pv))
        for l in ['ws', 'ws_replay', 'ws_replay_wide', 'ws_uniform']: nums |= {f'd{e}_{l}': nat(dk[l, '0.7'][:, c].mean()), f'd{e}_{l}_ent': nat(dk_ent[l, '0.7'][:, c].mean())}
        nums |= {f'd{e}_replayWins': str(int((dk['ws_replay', '0.7'][:, c] < dk['ws', '0.7'][:, c]).sum())), f'd{e}_uniformWins': str(int((dk['ws_uniform', '0.7'][:, c] > dk['ws', '0.7'][:, c]).sum()))}
    nums |= {'uniformEnt': nat(np.log(64)), 'dWorlds': str(len(dk['ws', '0.7']))}
    one = json.loads(DREAMS_ONE.read_text()); o = one['results'][PRIOR]; oa = one['args']
    ofin = {c: np.array([o[c][w][-1]['kl'] for w in sorted(o[c], key=int)]) for c in o}
    nums |= {f'one_{c}': nat(v.mean()) for c, v in ofin.items()} | {'one_start': nat(np.mean([o['prior'][w][0]['kl'] for w in o['prior']])), 'oneWorlds': str(len(ofin['prior'])), 'oneN': str(oa['N'])}
    dreams = {'P': Pd[1:], 'lines': {l: band(dk[l, '0.7'][:, 1:]) for l in ['ws', 'ws_replay', 'ws_uniform']}}

    # ---- Section 8: why the autoencoder plateaus (practice_ws_ae_floor.py; same worlds and start as section 5, own random draws) ----
    def floor_runs(run):
        j = json.loads((run / 'curves.json').read_text()); prior = next(iter(j['curves']))
        return j['args'], {spec: [ws[w] for w in sorted(ws, key=int)] for spec, ws in j['curves'][prior].items()}
    fargs, F = floor_runs(AE_DEEP); _, FN = floor_runs(AE_N128); _, FF = floor_runs(AE_FACT)
    assert list(F) and all(sp in F for sp in AE_SPECS) and fargs['P_max'] == 2 ** 21
    Pf = np.array([p['P'] for p in F['ws:x=0.7'][0]]); keepf = Pf >= 2 ** 8
    fin = lambda R, sp, k: np.array([r[-1][k] for r in R[sp]])
    at = lambda R, sp, k, Pv: np.array([[p[k] for p in r if p['P'] == Pv][0] for r in R[sp]])
    tag = {'ws:x=0.7': 'ws', 'ws:x=1': 'ws100', 'ae:top=mf:s=0.05': 'ae', 'ae:top=mf:s=1': 'aeF', 'ae:top=bin:s=1': 'aeB', 'ae:top=mf:s=1:wd=0.01': 'aeW', 'vae:s=1': 'vae'}
    nums |= {'fWorlds': str(len(F['ws:x=0.7'])), 'aeWd': f"{0.01:g}"}
    for sp, t in tag.items():
        nums |= {f'f_{t}': nat(fin(F, sp, 'kl').mean()), f'f_{t}_refit': nat(fin(F, sp, 'kl_refit').mean()), f'f_{t}_gen': deg(fin(F, sp, 'gen_world').mean()),
                 f'f_{t}_v': f"{fin(F, sp, 'v_norm').mean():.1f}", f'f_{t}_loss': nat(fin(F, sp, 'ae_loss_top').mean()), f'f_{t}_sat': f"{fin(F, sp, 'sat_top').mean():.2f}"}
        if sp != 'ws:x=0.7': nums[f'f_{t}_wsWins'] = str(int((fin(F, 'ws:x=0.7', 'kl') < fin(F, sp, 'kl')).sum()))
    for Pv in [2 ** 12, 2 ** 15]:
        e = int(np.log2(Pv)); nums |= {f'f_ae_v{e}': f"{at(F, 'ae:top=mf:s=0.05', 'v_norm', Pv).mean():.1f}", f'f_ae_refit{e}': nat(at(F, 'ae:top=mf:s=0.05', 'kl_refit', Pv).mean()),
                                     f'f_aeF_kl{e}': nat(at(F, 'ae:top=mf:s=1', 'kl', Pv).mean())}
    nums |= {'n_ws': nat(fin(FN, 'ws:x=0.7', 'kl').mean()), 'n_vae': nat(fin(FN, 'vae:s=1', 'kl').mean()), 'n_ae': nat(fin(FN, 'ae:top=mf:s=0.05', 'kl').mean()), 'n_ws100': nat(fin(FN, 'ws:x=1', 'kl').mean()),
             'n_vaeWins': str(int((fin(FN, 'vae:s=1', 'kl') < fin(FN, 'ws:x=0.7', 'kl')).sum()))}
    for sp, t in tag.items():
        if sp in FF: nums |= {f'x_{t}': nat(fin(FF, sp, 'kl').mean()), f'x_{t}_gen': deg(fin(FF, sp, 'gen_world').mean()), f'x_{t}_refit': nat(fin(FF, sp, 'kl_refit').mean()), f'x_{t}_v': f"{fin(FF, sp, 'v_norm').mean():.1f}"}
    nums |= {'x_wsWinsAE': str(int((fin(FF, 'ws:x=0.7', 'kl') < fin(FF, 'ae:top=mf:s=0.05', 'kl')).sum())), 'x_wsWinsNoGen': str(int((fin(FF, 'ws:x=0.7', 'kl') < fin(FF, 'ws:x=1', 'kl')).sum())),
             'x_vaeWins': str(int((fin(FF, 'vae:s=1', 'kl') < fin(FF, 'ws:x=0.7', 'kl')).sum())), 'x_kl0': nat(np.mean([r[0]['kl'] for r in FF['ws:x=0.7']])), 'x_gen0': deg(np.mean([r[0]['gen_world'] for r in FF['ws:x=0.7']])),
             'x_plateau': nat(np.mean([[p['kl'] for p in r if p['P'] == 2 ** 14][0] for r in FF['ws:x=0.7']]))}
    series = lambda R, k, specs: {sp: band(np.array([[p[k] for p in r] for r in R[sp]])[:, keepf]) for sp in specs if sp in R}
    floor = {'P': Pf[keepf], 'kl': series(F, 'kl', AE_SPECS), 'v': series(F, 'v_norm', AE_SPECS), 'gen': series(F, 'gen_world', AE_SPECS)}
    fact = {'P': Pf[keepf], 'kl': series(FF, 'kl', AE_SPECS), 'gen': series(FF, 'gen_world', AE_SPECS)}

    # ---- Section 4: is the top layer learned later than the bottom? (practice_ws_depth.py, fresh samples) ----
    dj = json.loads((DEPTH / 'curves.json').read_text()); DC = {k: [r[w] for w in sorted(r, key=int)] for k, r in dj['curves'].items()}
    dsteps = np.array([p['step'] for p in DC['rotated'][0]]); dk = dsteps >= 1; DV = lambda st, k: np.array([[p[k] for p in r] for r in DC[st]])
    def t_frac(v, f=0.9, k=7):  # first update at which the smoothed curve has covered a fraction f of its drop to its final level
        vs = np.convolve(np.pad(v, (k // 2, k // 2), mode='edge'), np.ones(k) / k, mode='valid'); v0, vf = vs[:3].mean(), vs[-10:].mean()
        return dsteps[np.argmax(vs <= v0 - f * (v0 - vf))]
    t90 = {(st, k): np.array([t_frac(v) for v in DV(st, k)]) for st in DC for k in ['gen_world', 'top_world'] if not (st == 'oracle_bottom' and k == 'gen_world')}
    nums |= {'dWorldsD': str(len(DC['rotated'])), 'dSteps': f"{dj['args']['steps']:,}", 'oracle90': f"{np.median(t90['oracle_bottom', 'top_world']):,.0f}",
             'oracleTop': deg(DV('oracle_bottom', 'top_world')[:, -1].mean())}
    for st, t in [('rotated', 'R'), ('random', 'X')]:
        tb, tt, to = t90[st, 'gen_world'], t90[st, 'top_world'], t90['oracle_bottom', 'top_world']
        nums |= {f'd{t}_bot90': f"{np.median(tb):,.0f}", f'd{t}_top90': f"{np.median(tt):,.0f}", f'd{t}_later': str(int((tt > tb).sum())), f'd{t}_slow': f"{np.median(tt / np.maximum(to, 1)):.0f}",
                 f'd{t}_slowN': str(int((tt > to).sum())), f'd{t}_top': deg(DV(st, 'top_world')[:, -1].mean()), f'd{t}_bot': deg(DV(st, 'gen_world')[:, -1].mean())}
    nums |= {'dX_top200': deg(DV('random', 'top_world')[:, np.argmin(np.abs(dsteps - 200))].mean()), 'dX_top1k': deg(DV('random', 'top_world')[:, np.argmin(np.abs(dsteps - 1000))].mean())}
    depth = {'steps': dsteps[dk], 'lines': {f'{st}/{k}': band(DV(st, k)[:, dk]) for st in DC for k in ['gen_world', 'top_world'] if not (st == 'oracle_bottom' and k == 'gen_world')}}

    # ---- assemble ----
    def serial(v):
        if isinstance(v, np.ndarray): return v.tolist() if v.dtype.kind in 'iu' else np.round(v, 4).tolist()
        if isinstance(v, (np.integer, np.floating)): return v.item()
        raise TypeError(type(v))

    payload = {'layers': layers, 'budget': budget, 'scarce': scarce, 'K': kth, 'dreams': dreams, 'floor': floor, 'fact': fact, 'depth': depth}
    sources = [LAYERS / 'snapshots.npz', B_MAIN / 'curves.json', B_AE / 'curves.json', AE_DEEP / 'curves.json', DEPTH / 'curves.json']
    nums |= {'built': datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z'), 'filesDir': f'{NAME}-files',
             'sourceHash': ' · '.join(f'{p.parent.name}/{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()[:12]}' for p in sources)}
    html = (root / 'experiments/toy_model_2_ws_report.html').read_text()
    for key, value in nums.items(): html = html.replace('{{' + key + '}}', value)
    assert '{{' not in html, html[html.index('{{'):html.index('{{') + 60]
    head, rest = html.split('<main>', 1); body, tail = rest.split('</main>', 1)
    tail = tail.replace('/*__SIM__*/', sim.read_text()).replace('__REPORT_DATA__', json.dumps(payload, default=serial, separators=(',', ':')))
    html = head.replace('/*__KATEX_CSS__*/', stylesheet()) + '<main>' + typeset(body) + '</main>' + tail

    # Shared copy: world-readable dirs and files on ceph (umask here is 077, so set modes explicitly).
    files = SHARED / f'{NAME}-files'; files.mkdir(parents=True, exist_ok=True)
    for src in CODE: shutil.copy2(root / src, files / Path(src).name)
    out = SHARED / f'{NAME}.html'; out.write_text(html)
    for p in [SHARED.parent, SHARED, files]: p.chmod(0o755)
    for p in [out, *files.iterdir()]: p.chmod(0o644)
    for target in [out, files]:
        link = root / 'reports' / target.name
        if link.is_symlink(): link.unlink()
        link.symlink_to(target)
    print(out); print(f'http://127.0.0.1:8095/{NAME}.html'); print(f'{len(html) / 1e6:.2f} MB')
    if args.project:  # the project has the write-up itself: link to it there
        page = root / PROJECT / f'{NAME}.html'; (page.parent / files.name).mkdir(exist_ok=True)
        for src in CODE:
            if not src.endswith('.tex'): shutil.copy2(root / src, page.parent / files.name / Path(src).name)
        page.write_text(html.replace(f'href="{NAME}-files/practice_disentangle.tex"', 'href="../practice_disentangle.tex"'))
        print(page)


if __name__ == '__main__': main()
