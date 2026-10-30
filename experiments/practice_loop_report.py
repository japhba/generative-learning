"""Build the practice-loop report from saved runs.

Every number in the page is computed here from the saved trajectories and summaries; nothing is typed in by hand.
The page is written to the world-readable ceph reports folder (so other cluster users can open it), together with
copies of the LaTeX write-ups it links to, and symlinked into reports/ for the local 127.0.0.1:8095 server.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np

from report_katex import stylesheet, typeset

SHARED = Path('/ceph/scratch/jbauer/reports/generative-learning')
PROJECT = 'tex/practice-loop'  # in the clone of the OverLyX project jan/generative-learning


def pct(x): return f'{100 * x:.1f}%'
def pp(x): return f'{100 * x:.1f}'
def ci(b): return f'[{100 * b[0]:.1f}, {100 * b[1]:.1f}]'


def main():
    args = argparse.ArgumentParser(description=__doc__); args.add_argument('--project', action='store_true', help=f'also write the page into {PROJECT}/ (overwrites edits made there in OverLyX)'); args = args.parse_args()
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    fast = Path(os.environ['FAST_CACHE_DIR'])
    ring = fast / 'generative_practice_loop_ring'
    J = lambda p: json.loads(Path(p).read_text())
    source = ring / 'confirmation/trajectories.npz'
    c, nat = np.load(source), np.load(ring / 'controls/trajectories.npz')
    confirmation, controls, analysis = J(ring / 'confirmation/summary.json'), J(ring / 'controls/summary.json'), J(ring / 'analysis/summary.json')
    bypass, identity, transfer = J(ring / 'data_bypasses/summary.json'), J(ring / 'state_reconstruction/summary.json'), J(ring / 'transfer/summary.json')
    sizes, gain, certified = J(ring / 'sizes_learned_queries/summary.json'), J(ring / 'trainable_gain/summary.json'), J(ring / 'certified/summary.json')
    orient, noise, bayes = J(ring / 'orientation/summary.json'), J(ring / 'orientation/noise_control/summary.json'), J(fast / 'generative_practice_loop_bayesian/summary.json')['results']
    assert all(a == 1.0 for a in confirmation['checker_accuracy'])
    assert all(analysis['identity_reconstruction_diagnostics'][k]['identity_accuracy'] == 1.0 for k in ['alias_gamma1', 'natural_gamma1'])

    methods = {'practice': 'practice', 'continue': 'atomic_reconstruction', 'unjudged': 'unfiltered', 'shuffled': 'shuffled'}
    curves = {k: c[f'curves_{m}'] for k, m in methods.items()}  # (checkpoint, run, [true success, checker-predicted success, single-step loss])
    end = {k: v[-1].mean(axis=0) for k, v in curves.items()}
    start = curves['practice'][0].mean(axis=0)
    finals = curves['practice'][-1, :, 0]
    s = analysis['geometry'], analysis['spectral_interventions'], analysis['orientation_interventions']
    tr = transfer['summary']['natural']
    nums = {
        'start': pct(start[0]), 'practice': pct(end['practice'][0]), 'continue': pct(end['continue'][0]), 'unjudged': pct(end['unjudged'][0]), 'shuffled': pct(end['shuffled'][0]),
        'lossStart': f'{start[2]:.4f}', 'lossPractice': f'{end["practice"][2]:.3f}', 'lossUnjudged': f'{end["unjudged"][2]:.3f}', 'lossShuffled': f'{end["shuffled"][2]:.3f}', 'lossIdeal': f'{np.log1p(np.exp(-3)):.4f}',
        'ideal': pct(confirmation['ideal_winding_one_success'][0]), 'gain': pp(analysis['exact_optimum_gain']['mean']), 'gainCI': ci(analysis['exact_optimum_gain']['paired_bootstrap_95']),
        'above90': str(int((finals > .9).sum())), 'below30': str(int((finals < .3).sum())), 'runs': str(len(finals)),
        'mode1': f'{s[0]["final_mode1_overlap"]:.2f}', 'keepMode1': pct(s[1]['keep_mode1']['mean']), 'removeMode1': pct(s[1]['remove_mode1']['mean']), 'reflect': pct(s[2]['reflect_final']['mean']),
        'natStart': pct(controls['natural_after_atomic_pretraining'][0]), 'natPractice': pct(controls['final']['natural_practice'][0]), 'natContinue': pct(controls['final']['natural_atomic'][0]),
        'natBest': pct(analysis['natural_passive_envelope']['best_common_true_success']), 'natCI': ci(analysis['natural_practice_vs_continue_atoms']['paired_bootstrap_95']),
        'trStart': pct(tr['partial_goals']['initial_mean'][0]), 'trPractice': pct(tr['partial_goals']['final_mean'][0]), 'trContinue': pct(tr['atomic_only']['final_mean'][0]), 'trCI': ci(tr['paired_held_goal_practice_minus_atomic']['bootstrap95']),
        'idAlias': pct(identity['final']['alias_gamma1'][0]), 'idNat': pct(identity['final']['natural_gamma1'][0]),
        'rawAlias': pct(bypass['raw_final']['alias_raw_replay'][0]), 'rawNat': pct(bypass['raw_final']['natural_raw_replay'][0]), 'slow': pct(bypass['spectral_final_mean'][0]),
        'gStart': pct(gain['pretrained'][0]), 'gPractice': pct(gain['final']['practice'][0]), 'gContinue': pct(gain['final']['atomic'][0]),
        'oStart': pct(noise['initial'][0]), 'oPractice': pct(noise['final']['practice'][0]), 'oScrambled': pct(noise['final']['sign_scrambled'][0]), 'oContinue': pct(orient['final']['atomic'][0]),
        'oRevStart': pct(orient['initial'][3]), 'oRevPractice': pct(orient['final']['practice'][3]),
        'certWorlds': str(certified['seeds']), 'certBlocks': str(certified['accepted_count']), 'certStart': pct(np.mean(certified['initial_success'])), 'certFinal': pct(certified['final_mean_success']),
    }
    for level in ['35', '55']:
        b = bayes[f'noise{level}']
        nums |= {f'b{level}Acc': pct(b['recognizer_modal_transition_accuracy']), f'b{level}Start': pct(b['pretrained'][0]), f'b{level}Practice': pct(b['final']['practice'][0]), f'b{level}Continue': pct(b['final']['atomic'][0]), f'b{level}Worse': str(b['worlds_worse_under_practice'])}

    # Why the folded map fails: generator's probability of stepping towards a goal d steps ahead, folded (90 deg/step) vs once-round (30 deg/step).
    toward = lambda step, d: 1 / (1 + np.exp(-6 * np.sin(np.deg2rad(step * d))))
    why = ''.join(f'<tr><th scope="row">{d}</th><td>{toward(90, d):.1%}</td><td>{toward(30, d):.1%}</td></tr>' for d in range(2, 7))
    size_rows = ''.join(f'<tr><th scope="row">{n}</th><td>{pct(r["pretrained"][0])}</td><td>{pct(r["final"]["practice"][0])}</td><td>{pct(r["final"]["atomic"][0])}</td><td>{pct(r["final"]["shuffled"][0])}</td></tr>' for n, r in sizes['results'].items())

    # Interactive data. Phases are reordered so index i is the i-th state around the true ring (for drawing only).
    phases = np.take_along_axis(c['angles_practice'], c['ordering'][None], axis=2)
    band = lambda v: {'mean': v.mean(axis=1), 'sem': v.std(axis=1, ddof=1) / np.sqrt(v.shape[1])}
    payload = {
        'steps': c['steps'], 'phases': np.round(phases, 4), 'runs': curves['practice'][:, :, 0].T,
        'success': {k: band(v[:, :, 0]) for k, v in curves.items()}, 'loss': {k: band(v[:, :, 2]) for k, v in curves.items()},
        'natural': {'pretrainSteps': nat['pretrain_steps'], 'steps': nat['steps'], 'pretraining': band(nat['pretraining'][:, :, 0]), 'practice': band(nat['curves_natural_practice'][:, :, 0]), 'continue': band(nat['curves_natural_atomic'][:, :, 0])},
    }
    def serial(v):
        if isinstance(v, np.ndarray): return np.round(v, 6).tolist()
        raise TypeError(type(v))

    built = datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z')
    html = (root / 'experiments/practice_loop_report.html').read_text()
    nums |= {'built': built, 'sourceHash': hashlib.sha256(source.read_bytes()).hexdigest()[:16], 'source': str(source), 'whyRows': why, 'sizeRows': size_rows}
    for key, value in nums.items(): html = html.replace('{{' + key + '}}', value)
    assert '{{' not in html, html[html.index('{{'):html.index('{{') + 40]
    head, rest = html.split('<main>', 1)
    body, tail = rest.split('</main>', 1)
    html = head.replace('/*__KATEX_CSS__*/', stylesheet()) + '<main>' + typeset(body) + '</main>' + tail.replace('__REPORT_DATA__', json.dumps(payload, default=serial, separators=(',', ':')))

    # Shared copy: world-readable dirs and files on ceph (umask here is 077, so set modes explicitly).
    files = SHARED / 'toy_model_1_ring-files'
    (files / 'figures').mkdir(parents=True, exist_ok=True)
    for doc in sorted(root.glob('tex/practice_loop*.tex')) + sorted(root.glob('tex/figures/practice-loop-*.png')): shutil.copy2(doc, files / doc.relative_to(root / 'tex'))
    out = SHARED / 'toy_model_1_ring.html'
    out.write_text(html)
    for p in [SHARED.parent, SHARED, files, files / 'figures']: p.chmod(0o755)
    for p in [out, *files.rglob('*.*')]: p.chmod(0o644)
    for target in [out, files]:
        link = root / 'reports' / target.name
        if link.is_symlink(): link.unlink()
        link.symlink_to(target)
    print(out)
    print('http://127.0.0.1:8095/toy_model_1_ring.html')
    if args.project:  # the project has the write-ups itself: link to them, copy nothing
        page = root / PROJECT / 'toy_model_1_ring.html'
        page.parent.mkdir(exist_ok=True)
        page.write_text(html.replace('href="toy_model_1_ring-files/', 'href="../'))
        print(page)


if __name__ == '__main__': main()
