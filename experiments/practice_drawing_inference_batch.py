"""Fresh confirmation of the loop with no direct dream-to-memory bypass."""

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm

import practice_drawing_sampled as s


def run(job):
    root,seed,index=job
    with (root/f'run_{index}.log').open('w') as log:
        subprocess.run([sys.executable,str(Path(s.__file__).resolve()),'--output',str(root/f'run_{index}'),'--seed',str(seed),
                        '--route','proposal','--recognition-initialization','zero',
                        '--conditions','sleep','prior','wake','frozen_proposal','fixed_posterior','fixed_cache'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((root/f'run_{index}/summary.json').read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20103160)
    parser.add_argument('--worlds',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),seed=args.seed,worlds=args.worlds,config=s.CONFIG,
        source_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),
        architecture='No neural recognition pretraining. Generated practice trains the inverse network; only conditional network proposals enter program memories after initialization.',
        primary='At 1600 updates, stopped runs held fixed: sleep minus frozen neural proposal, fixed initial program cache, fully fixed initial posterior, and fourfold-CV-selected analytic fits. Report every contrast.',
        secondary='Sleep versus initial-generator training and cached-cause wake training; no claim of equal wall time.',
        direct='Same eleven controls and independent-fold refitting as sampled_confirmation.',
        guarantee='A .999-per-world sampled bound on likelihood improvement, valid at all subsequent times and for all conditions. Eight-world union bound .992.',
        initialization='Caches and initial-likelihood counts use 2^20 initial-generator draws. Neural inverse weights remain zero. Teacher and exhaustive probabilities are evaluation-only.',
        motivation='A retrospective routing control found a recognition-learning benefit only after removing direct dream-to-cache insertion. This is a new independent test of that architecture.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    shutil.copy2(s.__file__,args.output/'source.py')
    shutil.copy2(__file__,args.output/'batch_source.py')
    rows={}
    with ProcessPoolExecutor(args.workers) as executor:
        futures={executor.submit(run,(args.output,args.seed+i,i)):i for i in range(args.worlds)}
        for future in tqdm(as_completed(futures),total=args.worlds,desc='Fresh inference-loop worlds'):
            i=futures[future]; rows[i]=future.result()
            (args.output/'summary.json').write_text(json.dumps({str(k):v for k,v in sorted(rows.items())},indent=2)+'\n')
            print(json.dumps(dict(world=i,**rows[i])),flush=True)


if __name__=='__main__': main()
