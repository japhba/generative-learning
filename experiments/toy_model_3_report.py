"""Build toy_model_3_chain.html: the Oracle (a learner made of tables) on a chain world. Checking a chain is one lookup of a
pair table per link; writing one in a single pass needs a tree of bridge tables. The pair table is learned from few teacher
chains; practice (joining held units, writing in one pass, checking, counting) fills the bridge tables without teacher data.

Every number in the page comes from /var/tmp/jbauer/generative_practice_chain/numbers.json (toy_model_3_figures.py, from the
runs of toy_model_3_oracle.py); nothing is typed in by hand. The page goes to the world-readable ceph reports folder with its
figures and code, and is symlinked into reports/ for the local 127.0.0.1:8095 server; --project also writes it into the
OverLyX project (tex/practice-loop/).
Usage: python toy_model_3_report.py [--project]"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

from zoneinfo import ZoneInfo

from report_katex import stylesheet, typeset
from toy_model_3_oracle import M_MAX

SHARED = Path('/ceph/scratch/jbauer/reports/generative-learning')
PROJECT = 'tex/practice-loop'  # in the clone of the OverLyX project jan/generative-learning
NAME = 'toy_model_3_chain'
RUNS = Path('/var/tmp/jbauer/generative_practice_chain')
FIGS = ['scheme', 'headstart', 'practice', 'staircase', 'tradeoff', 'supervision', 'checks', 'bridge', 'own-foreign']
CODE = ['experiments/toy_model_3_oracle.py', 'experiments/toy_model_3_figures.py', 'experiments/toy_model_3_report.py', 'experiments/toy_model_3_report.html']
COUNT = r'half_.*|stair12_\d|gain_peak_N\d|sched_[NM]|map_\w+_(10|lo10|hi10|w1)|big_\w+_10|noisy_\w+_(wrong|half|wins)|bridge_entries|split\d+_wins'  # whole numbers
TIMES = r'lead(_theory)?\d|saving'  # factors
SHARE = r'multi_route\d|n\d+_\w+|gain_peak\d|sched_(watch_first|alternate|practise_first)|noisy_\w+_own0?|slip_\w+|false\d+_(\d|pred3)|usable_(at|pred)_10|msplit\w+|split\d+_(all|half)|stair_q\d+|bridge_multi|pairs_only_gap|read_write_gap'


def count(x): return f'{x:,.0f}'
def times(x): return f'{x:,.0f}' if x >= 10 else f'{x:.1f}'
def share(x): return '0%' if x == 0 else '100%' if x >= 0.9995 else f'{100 * x:.2f}%' if x < 0.00095 else f'{100 * x:.1f}%' if x < 0.095 or x > 0.905 else f'{100 * x:.0f}%'
def million(x): return f'{x / 1e6:.1f} million'


def main():
    args = argparse.ArgumentParser(description=__doc__); args.add_argument('--project', action='store_true', help=f'also write the page into {PROJECT}/'); args = args.parse_args()
    root = Path(__file__).resolve().parents[1]; v = json.loads((RUNS / 'numbers.json').read_text()); nums = {}
    for k, x in v.items():
        nums[k] = x if isinstance(x, str) else count(x) if re.fullmatch(COUNT, k) else times(x) if re.fullmatch(TIMES, k) else share(x) if re.fullmatch(SHARE, k) else None
    nums |= {'S': count(v['S']), 'm': str(v['m']), 'n': str(v['n']), 'pairs_total': count(v['S'] * v['m']), 'spans_total': count(v['S'] * v['m'] ** (v['n'] - 1)), 'm_max': million(M_MAX), 'bridge_M': million(2 ** 22),
             'map_ratio_10': times(v['map_foreign_10'] / v['map_own_10']), 'bridge_cond1': f"{v['bridge_cond1']:.2g}", 'bridge_cond2': f"{v['bridge_cond2']:.2g}", 'bridge_corr': f"{v['bridge_corr']:.3f}", 'filesDir': f'{NAME}-files',
             'built': datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z'),
             'sourceHash': ' · '.join(f'{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()[:12]}' for p in [RUNS / 'watch.json', RUNS / 'practice.json', RUNS / 'map.json', RUNS / 'numbers.json'])}
    html = (root / 'experiments/toy_model_3_report.html').read_text()
    for key in sorted(set(re.findall(r'\{\{(\w+)\}\}', html))): assert nums.get(key) is not None, f'no value for {key}'; html = html.replace('{{' + key + '}}', nums[key])
    head, rest = html.split('<main>', 1); body, tail = rest.split('</main>', 1)
    html = head.replace('/*__KATEX_CSS__*/', stylesheet()) + '<main>' + typeset(body) + '</main>' + tail

    # Shared copy: world-readable dirs and files on ceph (umask here is 077, so set modes explicitly).
    files = SHARED / f'{NAME}-files'; files.mkdir(parents=True, exist_ok=True)
    for src in CODE: shutil.copy2(root / src, files / Path(src).name)
    for f in FIGS: shutil.copy2(root / 'reports' / f'toy3-{f}.png', files / f'toy3-{f}.png')
    out = SHARED / f'{NAME}.html'; out.write_text(html)
    for p in [SHARED.parent, SHARED, files]: p.chmod(0o755)
    for p in [out, *files.iterdir()]: p.chmod(0o644)
    for target in [out, files]:
        link = root / 'reports' / target.name
        if link.is_symlink(): link.unlink()
        link.symlink_to(target)
    print(out); print(f'http://127.0.0.1:8095/{NAME}.html'); print(f'{len(html) / 1e6:.2f} MB')
    if args.project:
        page = root / PROJECT / f'{NAME}.html'; (page.parent / files.name).mkdir(exist_ok=True)
        for f in files.iterdir(): shutil.copy2(f, page.parent / files.name / f.name)
        page.write_text(html); print(page)


if __name__ == '__main__': main()
