"""Prospective fresh worlds for the latent drawing-program comparison."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm

import practice_drawing_library as m


def run(job):
    output, seed = job
    with (output.parent/(output.name+'.log')).open('w') as log:
        subprocess.run([sys.executable, str(Path(m.__file__).resolve()), '--output', str(output), '--worlds', '1',
                        '--seed', str(seed), '--steps', '600', '--conditions', 'sleep', 'wake', 'frozen', 'canonical', 'exact', 'calibrated'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20103020)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), seed=args.seed, worlds=args.worlds,
        source_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(), config=m.CONFIG,
        primary='Paired true image KL at 400 generator updates, stopped models held fixed: sleep minus wake, calibrated recognition, and CV direct fits.',
        secondary='Internally stopped endpoints, original-cause fit, untrained recognition, and exact posterior inference.',
        calibration='A stronger fixed recognizer gets 600 sleep updates on the initial fixed generator, then is frozen. The exploratory pilot used 200.',
        direct='Fourfold CV among nine fixed candidates: five uniform-renderer-smoothed histograms, two original-program count fits, independent pixels, and a pixel Chow-Liu tree. All folds refit on training observations only.',
        caveat='This tests generator-update budgets, not equal wall time. Population-risk guarantees are not inferred from likelihood monotonicity.')
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(m.__file__, args.output/'source.py')
    shutil.copy2(__file__, args.output/'batch_source.py')
    with ProcessPoolExecutor(args.workers) as executor:
        futures = [executor.submit(run, (args.output/f'run_{i}', args.seed+i)) for i in range(args.worlds)]
        for future in tqdm(as_completed(futures), total=args.worlds, desc='Fresh drawing worlds'): print(future.result(), flush=True)
    rows = []
    for i in range(args.worlds):
        source = args.output/f'run_{i}'
        row = json.loads((source/'summary.json').read_text())[0]
        rows.append(row)
        shutil.copy2(source/'world_0.npz', args.output/f'world_{i}.npz')
        for condition in row['conditions']: shutil.copy2(source/f'{condition}_0.npz', args.output/f'{condition}_{i}.npz')
    (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__': main()
