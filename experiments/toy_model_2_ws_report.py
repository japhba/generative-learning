"""Build toy_model_2_WS.html: the two-layer wake-sleep toy, its latent-space rotation, the alignment of recognition with
generation layer by layer, and its linear-Gaussian twin.

Every number in the page is computed here from the saved runs (practice_ws_layers.py), the pre-registered confirmation of
the one-layer toy (practice_disentangle.py) and the two-factor simulation (toy_model_2_ws_sim.js, run under node, the same
code the page animates); nothing is typed in by hand. The page goes to the world-readable ceph reports folder with copies
of the code it describes, and is symlinked into reports/ for the local 127.0.0.1:8095 server.
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
RUNS = Path('/var/tmp/jbauer/generative_practice_disentangle/layers')
MAIN, FRESH = RUNS / 'run_N256_20261025_221256', RUNS / 'run_N0_20261025_221256'
CONFIRM = Path('/var/tmp/jbauer/generative_practice_disentangle/confirm/claims.json')
PRIOR_RUN = Path('/var/tmp/jbauer/generative_practice_disentangle/prior/chain_N256_rotated_hebb_20261025_111826/results.json')  # practice_disentangle_prior.py, label identity
CONDS = ['two_layer', 'true_prior', 'independent', 'gaussian']  # the four learners of the summary charts, in colour order
SHOWN = ['two_layer', 'true_prior', 'gaussian']  # learners in the explorer
EVERY = 2  # explorer frame = every second snapshot (400 updates)

# The two-factor numbers, from the simulation the page runs (16 seeds, start 30 degrees, sigma 0.5).
NODE = r"""
const WS = require(process.env.SIM), S = 16, T = 3000, run = (kind, frozen, s) => {
  const w = WS.world(1000 + s, 0.5), L = WS.learner(kind, Math.PI / 6, 0.5), r = WS.rng(5000 + s), tr = [];
  for (let t = 0; t <= T; t++) { tr.push([WS.genAngle(L), WS.recAngle(L)]); if (t < T) WS.step(L, w, r, {frozen}); }
  return tr; };
const all = (kind, frozen) => Array.from({length: S}, (_, s) => run(kind, frozen, s));
const mean = a => a.reduce((x, y) => x + y, 0) / a.length, sd = a => Math.sqrt(mean(a.map(x => (x - mean(a)) ** 2)));
const b = all('binary', false), bf = all('binary', true), g = all('gaussian', false), gf = all('gaussian', true);
const at = (runs, t, i = 0) => runs.map(tr => tr[t][i]);
process.stdout.write(JSON.stringify({
  b250: mean(at(b, 250).map(Math.abs)), b500: mean(at(b, 500).map(Math.abs)), b1000: mean(at(b, 1000).map(Math.abs)), b3000: mean(at(b, T).map(Math.abs)),
  lag250: mean(b.map(tr => tr[250][1] - tr[250][0])), frozen: mean(at(bf, T)), frozenMin: Math.min(...at(bf, T)), frozenMax: Math.max(...at(bf, T)),
  g3000: mean(at(g, T)), gsd: sd(at(g, T)), gmin: Math.min(...at(g, T)), gmax: Math.max(...at(g, T)), grec: mean(g.map(tr => Math.abs(tr[T][1] - tr[T][0]))),
  gfrozen: mean(at(gf, T)), land: WS.landscape(0.5, [15, 30, 45]),
}));
"""


def band(runs, key):
    v = np.array([[p[key] for p in r] for r in runs]); return {'mean': v.mean(0), 'sem': v.std(0, ddof=1) / np.sqrt(len(v))}


def explorer(snap, curves, prior, cond, w):
    """Per frame: |cos| of the generator's columns with the world's (rows: learner units in the order of the world cause they
    render at the end), |cos| of the recognizer's rows with the sleep fixed point's, the learner's prior correlations in
    world order, and for two layers the same recognizer comparison one layer up."""
    g = lambda k: snap[f'{prior}/{cond}/{w}/{k}'][::EVERY]
    W, R, Rs, corr = g('W'), g('R'), g('Rstar'), g('corr'); A = snap[f'{prior}/world/{w}/A']
    order = np.argsort(snap[f'{prior}/{cond}/{w}/perm'][-1])  # row r: the learner unit that renders world cause r at the end
    unit = lambda M, ax: M / np.maximum(np.linalg.norm(M, axis=ax, keepdims=True), 1e-12)
    G = np.abs(np.einsum('fdi,dj->fij', unit(W, 1)[:, :, order], unit(A, 0)))
    Rr = np.abs(np.einsum('fid,fjd->fij', unit(R, 2)[:, order], unit(Rs, 2)[:, order]))
    S = snap[f'{prior}/world/{w}/Sigma']; d = np.sqrt(np.diag(S))
    curve = curves[prior][cond][w][::EVERY]
    out = {'G': np.rint(100 * G).astype(int).reshape(len(G), -1), 'Rr': np.rint(100 * Rr).astype(int).reshape(len(G), -1), 'C': np.rint(100 * corr).astype(int).reshape(len(G), -1),
           'world': np.rint(100 * S / np.outer(d, d)).astype(int).ravel(), 'gen': [p['gen_world'] for p in curve], 'lag': [p['rec_lag'] for p in curve], 'kl': [p['kl'] for p in curve]}
    if cond == 'two_layer':
        R2, R2s = g('R2'), g('R2star'); o = np.argsort(-np.linalg.norm(snap[f'{prior}/{cond}/{w}/V'][-1], axis=0))
        out['T'] = np.rint(100 * np.abs(np.einsum('fid,fjd->fij', unit(R2, 2)[:, o], unit(R2s, 2)[:, o]))).astype(int).reshape(len(G), -1)
        out['T'][0] = 0  # the top recognizer starts at zero, so it has no direction yet
        out['top'] = [p['top_lag'] for p in curve]
    return out


def main():
    args = argparse.ArgumentParser(description=__doc__); args.add_argument('--project', action='store_true', help=f'also write the page into {PROJECT}/'); args = args.parse_args()
    root = Path(__file__).resolve().parents[1]
    main_c, fresh_c = json.loads((MAIN / 'curves.json').read_text()), json.loads((FRESH / 'curves.json').read_text())
    curves, fresh, run_args = main_c['curves'], fresh_c['curves'], main_c['args']
    snap = np.load(MAIN / 'snapshots.npz'); claims = json.loads(CONFIRM.read_text())
    sim = root / 'experiments/toy_model_2_ws_sim.js'
    two = json.loads(subprocess.run(['node', '-e', NODE], capture_output=True, text=True, check=True, env={'SIM': str(sim), 'PATH': '/usr/bin:/bin:' + str(Path(shutil.which('node')).parent)}).stdout)

    end = lambda c, p, cond, k: float(np.mean([r[-1][k] for r in c[p][cond]]))
    start = lambda c, p, cond, k: float(np.mean([r[0][k] for r in c[p][cond]]))
    deg, nat = (lambda v: f'{v:.0f}°'), (lambda v: f'{v:.2f}')
    nums = {'worlds': str(run_args['worlds']), 'N': str(run_args['N']), 'H': str(run_args['H']), 'steps': f"{run_args['steps']:,}"}
    for p in ['chain', 'deep']:
        for cond in CONDS + ['two_layer_h2']:
            P = f'{p}_{cond}'
            nums |= {f'{P}_kl0': nat(start(curves, p, cond, 'kl')), f'{P}_kl': nat(end(curves, p, cond, 'kl')), f'{P}_gen0': deg(start(curves, p, cond, 'gen_world')), f'{P}_gen': deg(end(curves, p, cond, 'gen_world')),
                     f'{P}_lag': f"{end(curves, p, cond, 'rec_lag'):.1f}°", f'{P}_corr': nat(end(curves, p, cond, 'corr_err')), f'{P}_acc': f"{100 * end(curves, p, cond, 'acc_matched'):.0f}%", f'{P}_acc0': f"{100 * start(curves, p, cond, 'acc_matched'):.0f}%"}
            if cond.startswith('two_layer'): nums |= {f'{P}_top': f"{end(curves, p, cond, 'top_lag'):.1f}°"}
        for cond in ['two_layer', 'true_prior', 'gaussian']:
            nums |= {f'{p}_{cond}_fkl': nat(end(fresh, p, cond, 'kl')), f'{p}_{cond}_fgen': deg(end(fresh, p, cond, 'gen_world'))}
        nums[f'{p}_lag200'] = f"{np.mean([r[1]['rec_lag'] for r in curves[p]['two_layer']]):.0f}°"
        nums[f'{p}_toplag600'] = f"{np.mean([r[3]['top_lag'] for r in curves[p]['two_layer']]):.0f}°"
        lo, hi = [min(r[-1]['gen_world'] for r in curves[p][c]) for c in ['two_layer', 'gaussian']], [max(r[-1]['gen_world'] for r in curves[p][c]) for c in ['two_layer', 'gaussian']]
        nums |= {f'{p}_two_range': f'{lo[0]:.0f}–{hi[0]:.0f}°', f'{p}_gauss_range': f'{lo[1]:.0f}–{hi[1]:.0f}°'}
    nums |= {'deep_top_world0': deg(start(curves, 'deep', 'two_layer', 'top_world')), 'deep_top_world': deg(end(curves, 'deep', 'two_layer', 'top_world')),
             'chain_two_layer_own0': f"{100 * start(curves, 'chain', 'two_layer', 'acc_own'):.0f}%", 'chain_two_layer_own': f"{100 * end(curves, 'chain', 'two_layer', 'acc_own'):.0f}%"}
    pr = json.loads(PRIOR_RUN.read_text())['results']; sym = lambda c: f"{100 * np.mean([r[-1]['acc_sym'] for r in pr[c]]):.0f}%"
    nums |= {'priorWorlds': str(len(pr['top_layer'])), 'priorTopSym': sym('top_layer'), 'priorFixedSym': sym('fixed_true')}
    lag_all = [r[-1]['rec_lag'] for p in curves for c in curves[p] for r in curves[p][c]]
    nums |= {'lagMin': f'{min(lag_all):.0f}°', 'lagMax': f'{max(lag_all):.0f}°'}
    nums |= {'b250': f"{two['b250']:.0f}°", 'b500': f"{two['b500']:.0f}°", 'b1000': f"{two['b1000']:.1f}°", 'b3000': f"{two['b3000']:.1f}°", 'lag250': f"{two['lag250']:.1f}°",
             'frozen': f"{two['frozen']:.0f}°", 'frozenRange': f"{two['frozenMin']:.0f}–{two['frozenMax']:.0f}°", 'g3000': f"{two['g3000']:.0f}°", 'gsd': f"{two['gsd']:.0f}°",
             'grange': f"{two['gmin']:.0f}–{two['gmax']:.0f}°", 'grec': f"{two['grec']:.1f}°", 'gfrozen': f"{two['gfrozen']:.1f}°",
             'land15': f"{-two['land'][0]:.2f}", 'land30': f"{-two['land'][1]:.2f}", 'land45': f"{-two['land'][2]:.2f}"}
    c1 = claims['C1_chain']['comparisons']; c2 = claims['C2']['comparisons']; c3 = claims['C3']['comparisons']['chain_N256']
    nums |= {'c1Frozen': nat(c1['wake_frozen']['rival']), 'c1Fresh': nat(c1['wake_fresh']['rival']), 'c1Practice': nat(c1['wake_frozen']['practice']), 'c1Bae': nat(c1['binary_ae']['rival']),
             'c1Assign': nat(c1['binary_ae_assigned']['rival']), 'c1AssignWins': str(c1['binary_ae_assigned']['wins']),
             'c2n64': nat(c2['chain_N64']['rival']), 'c2n64p': nat(c2['chain_N64']['practice']), 'c3f05': nat(c3['f05']), 'c3f0': nat(c3['f0'])}
    dens = claims['descriptive_vs_density_users']; nums |= {'vaeChain': nat(dens['chain_N256_vae']['rival']), 'emChain': nat(dens['chain_N256_exact_em']['rival'])}

    steps = np.array([p['step'] for p in curves['chain']['two_layer'][0]])
    summary = {p: {cond: {k: band(curves[p][cond], k) for k in ['kl', 'gen_world', 'rec_lag', 'corr_err'] + (['top_lag'] if cond == 'two_layer' else [])} for cond in CONDS} for p in ['chain', 'deep']}
    # The explorer's snapshots (most of the bytes) go in a file of their own, fetched once the page runs: OverLyX's canvas
    # must start within 20 s, and in edit mode it is sent the page twice, over the Remote-SSH tunnel.
    payload = {'steps': steps, 'summary': summary, 'frameSteps': steps[::EVERY]}
    frames = {p: {cond: [explorer(snap, curves, p, cond, w) for w in range(run_args['worlds'])] for cond in SHOWN} for p in ['chain', 'deep']}

    def serial(v):
        if isinstance(v, np.ndarray): return v.tolist() if v.dtype.kind == 'i' else np.round(v, 4).tolist()
        if isinstance(v, (np.integer, np.floating)): return v.item()
        raise TypeError(type(v))

    nums |= {'built': datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z'), 'source': str(MAIN / 'snapshots.npz'),
             'sourceHash': hashlib.sha256((MAIN / 'snapshots.npz').read_bytes()).hexdigest()[:16], 'filesDir': f'{NAME}-files'}
    html = (root / 'experiments/toy_model_2_ws_report.html').read_text()
    for key, value in nums.items(): html = html.replace('{{' + key + '}}', value)
    assert '{{' not in html, html[html.index('{{'):html.index('{{') + 60]
    head, rest = html.split('<main>', 1); body, tail = rest.split('</main>', 1)
    tail = tail.replace('/*__SIM__*/', sim.read_text()).replace('__REPORT_DATA__', json.dumps(payload, default=serial, separators=(',', ':')))
    html = head.replace('/*__KATEX_CSS__*/', stylesheet()) + '<main>' + typeset(body) + '</main>' + tail

    # Shared copy: world-readable dirs and files on ceph (umask here is 077, so set modes explicitly).
    files = SHARED / f'{NAME}-files'; files.mkdir(parents=True, exist_ok=True)
    for src in ['experiments/practice_ws_layers.py', 'experiments/toy_model_2_ws_sim.js', 'experiments/practice_disentangle.py', 'experiments/practice_disentangle_prior.py', 'tex/practice_disentangle.tex']: shutil.copy2(root / src, files / Path(src).name)
    (files / 'explorer.json').write_text(json.dumps(frames, default=serial, separators=(',', ':')))
    out = SHARED / f'{NAME}.html'; out.write_text(html)
    for p in [SHARED.parent, SHARED, files]: p.chmod(0o755)
    for p in [out, *files.iterdir()]: p.chmod(0o644)
    for target in [out, files]:
        link = root / 'reports' / target.name
        if link.is_symlink(): link.unlink()
        link.symlink_to(target)
    print(out); print(f'http://127.0.0.1:8095/{NAME}.html')
    if args.project:  # the project has the write-up itself: link to it there
        page = root / PROJECT / f'{NAME}.html'; (page.parent / files.name).mkdir(exist_ok=True)
        for f in files.iterdir():
            if f.suffix != '.tex': shutil.copy2(f, page.parent / files.name / f.name)
        page.write_text(html.replace(f'href="{NAME}-files/practice_disentangle.tex"', 'href="../practice_disentangle.tex"'))
        print(page)


if __name__ == '__main__': main()
