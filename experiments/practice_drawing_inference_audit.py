"""Check the recorded final experiment against its claimed architecture."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

import practice_drawing_sampled as s
import practice_drawing_library as m


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((args.source/'protocol.json').read_text())
    assert hashlib.sha256((args.source/'source.py').read_bytes()).hexdigest()==protocol['source_sha256']
    rows=[]; programs=0; runs=0
    for i in range(protocol['worlds']):
        root=args.source/f'run_{i}'
        world=np.load(root/'world.npz'); initial=np.load(root/'initial.npz')
        setting=json.loads((root/'protocol.json').read_text())
        assert setting['route']=='proposal' and setting['recognition_initialization']=='zero'
        assert np.count_nonzero(initial['recognition'])==0
        a,b=s.initial_parameters(world['observed'],world['noise'],world['initial_a'])
        assert np.array_equal(a,world['initial_a']) and np.array_equal(b,world['initial_b'])
        targets=np.unique(world['observed'])
        for condition in setting['conditions']:
            run=np.load(root/f'{condition}.npz')
            assert not bool(run['cache_dreams'])
            assert np.all(run['cache_sizes']<=s.CONFIG['cache_size'])
            assert np.array_equal(m.render(s.decode(run['caches'])),np.repeat(targets,run['cache_sizes']))
            assert np.all(run['a']>=.02/3-1e-12) and np.all(run['b']>=.02/4-1e-12)
            assert np.allclose(run['a'].sum(axis=-1),1) and np.allclose(run['b'].sum(axis=-1),1)
            if condition in ['frozen_proposal','fixed_posterior','fixed_cache']:
                assert np.count_nonzero(run['recognition'])==0
            else: assert np.linalg.norm(run['recognition'])>0
            assert np.max(np.diff(run['records'][:,3]))<1e-9
            programs+=len(run['caches']); runs+=1
        learned=np.load(root/'sleep.npz')
        assert learned['records'][-1,4]>0 and learned['records'][-1,1]<learned['records'][0,1]
        assert np.linalg.norm(learned['a']-a)>0 and np.linalg.norm(learned['b']-b)>0
        names=np.array(list('FRLJ'))
        rows.append(dict(world=i,dominant_motifs=[''.join(row) for row in names[learned['b'].argmax(axis=-1)]],
            recognition_weight_norm=float(np.linalg.norm(learned['recognition'])),
            initial_recognition_valid=float(learned['records'][0,6]),final_recognition_valid=float(learned['records'][-1,6])))
    result=dict(source_hash_verified=True,runs_checked=runs,retained_programs_checked=programs,
                all_retained_programs_reconstruct_exactly=True,initial_inverse_weights_all_zero=True,
                no_direct_dream_insertion=True,rows=rows)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(__file__,args.output/'source.py')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
