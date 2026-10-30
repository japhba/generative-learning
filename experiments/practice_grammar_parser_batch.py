"""Run fresh worlds in independent single-thread CPU processes, then collect."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm

import practice_grammar_parser as model


def run_world(task):
    output, seed, steps, conditions = task
    with (output.parent/f'{output.name}.log').open('w') as log:
        subprocess.run([sys.executable, model.__file__, '--output', str(output), '--worlds', '1', '--seed', str(seed), '--steps', str(steps), '--no-plot', '--conditions', *conditions], stdout=log, stderr=subprocess.STDOUT, check=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20102920)
    parser.add_argument('--steps', type=int, default=1500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = args.output/'workers'; jobs.mkdir()
    conditions = ['sleep', 'wake', 'frozen']
    protocol = dict(uk_start=datetime.now(ZoneInfo('Europe/London')).isoformat(), worlds=args.worlds, workers=args.workers, seed=args.seed,
        steps=args.steps, conditions=conditions, config=model.CONFIG,
        runtime_caution='Concurrent single-thread CPU processes. Wall time includes evaluation and is a secondary, implementation-dependent measurement.',
        unused_dream_work=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, args.output/'batch_source.py'); shutil.copy2(model.__file__, args.output/'source.py')
    tasks = [(jobs/f'world_{i}', args.seed+i, args.steps, conditions) for i in range(args.worlds)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, output in enumerate(tqdm(pool.map(run_world, tasks), total=args.worlds, desc='Fresh grammar-parser worlds')):
            row = json.loads((output/'summary.json').read_text())[0]
            row['world'] = index
            rows.append(row)
            for name in ['world', *conditions]: shutil.copy2(output/f'{name}_0.npz', args.output/f'{name}_{index}.npz')
            (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    model.plot(args.output, conditions)
    print(json.dumps(dict(output=str(args.output), worlds=len(rows)), indent=2))


if __name__ == '__main__': main()
