"""Prospective fresh-world test of sampled causal recognition."""

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
        subprocess.run([sys.executable,str(Path(s.__file__).resolve()),'--output',str(root/f'run_{index}'),'--seed',str(seed)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((root/f'run_{index}/summary.json').read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worlds',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20103140)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),seed=args.seed,worlds=args.worlds,config=s.CONFIG,
        source_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),
        primary='At 1600 steps with stopped runs held fixed: current-generator sleep minus fully frozen posterior and minus cross-validated direct fits. Count worlds with positive finite-sample likelihood certificate.',
        controls='Four-fold CV among 5 histogram fits, canonical counts, shared command counts, pixels, Chow-Liu, and two noniterative spectral mixture controls. Canonical mappings are fixed renderer-derived lexicographic programs; spectral initial parameters refit within each fold.',
        secondary='Current-generator sleep versus initial-generator sleep, wake-on-cached-causes, and a frozen learned proposal network.',
        scope='Fresh worlds after a one-world smoke test. Probability .999 per world covers every time and condition jointly; union bound gives .992 for all eight worlds.',
        initialization='Initial generator from observed ink count and independent randomness. No exact initial posterior or supplied canonical inverse is used in training.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    shutil.copy2(s.__file__,args.output/'source.py')
    shutil.copy2(__file__,args.output/'batch_source.py')
    result={}
    with ProcessPoolExecutor(args.workers) as executor:
        futures={executor.submit(run,(args.output,args.seed+i,i)):i for i in range(args.worlds)}
        for future in tqdm(as_completed(futures),total=args.worlds,desc='Fresh sampled worlds'):
            i=futures[future]; result[i]=future.result()
            (args.output/'summary.json').write_text(json.dumps({str(k):v for k,v in sorted(result.items())},indent=2)+'\n')
            print(json.dumps(dict(world=i,**result[i])),flush=True)


if __name__=='__main__': main()
