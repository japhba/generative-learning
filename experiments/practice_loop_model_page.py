"""Build the interactive specification of the practice-loop toy model (task, arrays, objectives, updates, a live sandbox).

The page embeds the 32 saved worlds of the main run and re-implements the learner in JavaScript; the browser check
(practice_loop_model_page_check.cjs) verifies that implementation against the saved checkpoints. Written to the
world-readable ceph reports folder with a copy of the exact run source, and symlinked into reports/.
"""

import argparse
from datetime import datetime
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


def main():
    args = argparse.ArgumentParser(description=__doc__); args.add_argument('--project', action='store_true', help=f'also write the page into {PROJECT}/ (overwrites edits made there in OverLyX)'); args = args.parse_args()
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    run = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/confirmation'
    c, summary = np.load(run / 'trajectories.npz'), json.loads((run / 'summary.json').read_text())
    frames = [0, 75, 150]
    worlds = [{'seed': 86200 + w, 'truth': c['truth'][w], 'counts': c['counts'][w], 'learned': c['learned'][w], 'order': c['ordering'][w], 'queries': c['queries'][w],
               'saved': c['curves_practice'][:, w, 0], 'savedLoss': c['curves_practice'][:, w, 2],
               'check': {'frames': frames, 'theta': c['angles_practice'][frames, w], 'success': c['curves_practice'][frames, w, 0], 'internal': c['curves_practice'][frames, w, 1], 'loss': c['curves_practice'][frames, w, 2]}}
              for w in range(c['truth'].shape[0])]
    payload = {'steps': c['steps'], 'worlds': worlds, 'savedMean': c['curves_practice'][-1, :, 0].mean()}
    def serial(v):
        if isinstance(v, np.ndarray): return np.round(v, 12).tolist() if v.dtype.kind == 'f' else v.tolist()
        if isinstance(v, np.floating): return float(v)
        raise TypeError(type(v))
    nums = {'built': datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z'), 'optimum': f'{np.log1p(np.exp(-6)):.5f}', 'savedMean': f'{100 * payload["savedMean"]:.1f}%',
            'ideal': f'{100 * np.mean(summary["ideal_winding_one_success"]):.1f}%', 'folded': f'{100 * c["curves_practice"][0, :, 0].mean():.1f}%'}
    html = (root / 'experiments/practice_loop_model_page.html').read_text()
    for key, value in nums.items(): html = html.replace('{{' + key + '}}', value)
    assert '{{' not in html, html[html.index('{{'):html.index('{{') + 40]
    head, rest = html.split('<main>', 1)
    body, tail = rest.split('</main>', 1)
    html = head.replace('/*__KATEX_CSS__*/', stylesheet()) + '<main>' + typeset(body) + '</main>' + tail.replace('__MODEL_DATA__', json.dumps(payload, default=serial, separators=(',', ':')))

    files = SHARED / 'practice-loop-model-files'
    files.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / 'run_source.py', files / 'practice_loop_ring_confirmation.py')
    out = SHARED / 'practice-loop-model.html'
    out.write_text(html)
    for p in [SHARED.parent, SHARED, files]: p.chmod(0o755)
    for p in [out, *files.iterdir()]: p.chmod(0o644)
    for target in [out, files]:
        link = root / 'reports' / target.name
        if link.is_symlink(): link.unlink()
        link.symlink_to(target)
    print(out)
    print('http://127.0.0.1:8095/practice-loop-model.html')
    if args.project:
        page = root / PROJECT / 'practice-loop-model.html'
        (page.parent / files.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / 'run_source.py', page.parent / files.name / 'practice_loop_ring_confirmation.py')
        page.write_text(html)
        print(page)


if __name__ == '__main__': main()
