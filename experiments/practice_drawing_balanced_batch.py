"""Prospective fresh worlds for the posterior-balancing practice loop."""

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

import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as model
import practice_drawing_balanced as experiment


def run(job):
    root,index=job
    output=root/f'run_{index}'
    with (root/f'run_{index}.log').open('w') as log:
        subprocess.run([sys.executable,str(Path(experiment.__file__).resolve()),'--source',str(root/'worlds'),
                        '--output',str(output),'--worlds','1','--first-world',str(index),'--steps','1600'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worlds',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20103120)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'worlds').mkdir()
    protocol=dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),seed=args.seed,worlds=args.worlds,
        config=experiment.CONFIG,source_sha256=hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest(),
        primary='At 1600 updates, with stopped models held fixed: practice minus the fully frozen initial posterior and practice minus cross-validated direct fits.',
        secondary='Practice versus prior-fixed generation and recognition-only cause proposals. Equal generator-update counts are not equal wall times.',
        guarantee='Audit initial tightness, nonnegative local ELBO gains, and the cumulative lower bound on actual finite-data likelihood; no actual likelihood is used to accept or stop training.',
        recognition='Both proposal directions use their current distributions. R is a normalized table over program and motif causes, sampled through an updated cumulative-sum tree.',
        direct='The nine existing count/pixel controls, plus a planned noniterative spectral mixture fit to initial-posterior command moments. Refitting inside CV folds must exclude validation examples.',
        limitations='The exact initial posterior is computed by exhaustive enumeration. This initialization cost and the large recognition table are explicit assumptions.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    shutil.copy2(experiment.__file__,args.output/'source.py')
    shutil.copy2(__file__,args.output/'batch_source.py')
    cat=model.Catalogue()
    for i in range(args.worlds): np.savez_compressed(args.output/'worlds'/f'world_{i}.npz',**model.world(cat,args.seed+i))
    with ProcessPoolExecutor(args.workers) as executor:
        futures=[executor.submit(run,(args.output,i)) for i in range(args.worlds)]
        for future in tqdm(as_completed(futures),total=args.worlds,desc='Fresh balanced worlds'): print(future.result(),flush=True)
    rows=[]
    for i in range(args.worlds):
        source=args.output/f'run_{i}'
        row=json.loads((source/'summary.json').read_text())[0]
        rows.append(row)
        for condition in row['conditions']: shutil.copy2(source/f'{condition}_{i}.npz',args.output/f'{condition}_{i}.npz')
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__': main()
