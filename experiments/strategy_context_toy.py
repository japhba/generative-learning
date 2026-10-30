"""Exact finite-sample curves for unjudged rehearsal in a small strategy-state model."""
from datetime import datetime
import hashlib
import json
from math import lgamma
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from task_search_api import ROOT

NS=[0,1,2,4,8,16,32,64,128,256,512]


def softmax(value,axis=-1):
    shifted=np.exp(value-np.max(value,axis=axis,keepdims=True))
    return shifted/shifted.sum(axis=axis,keepdims=True)


def expectation(n,probability,initial,salience):
    dimension=len(probability)
    counts=np.array([[n-j,j] for j in range(n+1)] if dimension==2 else [[n-j-k,j,k] for j in range(n+1) for k in range(n-j+1)],dtype=int)
    log_weight=lgamma(n+1)+counts@np.log(probability)-np.array([sum(lgamma(int(c)+1) for c in row) for row in counts])
    weight=np.exp(log_weight);assert abs(weight.sum()-1)<1e-9
    prediction=softmax(initial+counts*salience,axis=1)
    return weight@prediction


def run():
    root=ROOT.resolve();directory=root/'strategy_context_toy';directory.mkdir(exist_ok=True);out=Path(__file__).resolve().parents[1]/'reports'
    design={'initial_two':[0.,-1.],'salience_two':[1.,4.],'success_two':[.25,.9],'initial_three':[0.,-1.,-3.],'salience_three':[1.,4.,45.],'success_three':[.25,.9,0.],'sample_counts':NS,'selection':'Illustrative parameters chosen to show a possible mechanism; not fitted to Qwen data.','update':'Generate each style independently from softmax(initial). Read every sample with h <- h + diag(salience) one_hot(style). Final strategy probabilities are softmax(h). Both source conditions use identical readout and the same fixed task-input stream. No success score enters the update.','limits':'Latent strategy probabilities summarize behavior across queries. The toy does not reproduce greedy token decoding, infer new world information, or claim that Qwen implements this state update.','time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=directory/'design.json';assert not path.exists(),'Keep the original toy design and output if changing the model.';path.write_text(json.dumps(design,indent=2));rows=[]
    for dimension in [2,3]:
        suffix='two' if dimension==2 else 'three';initial=np.array(design['initial_'+suffix]);salience=np.array(design['salience_'+suffix]);success=np.array(design['success_'+suffix]);p=softmax(initial)
        for n in tqdm(NS,desc=f'Exact {dimension}-strategy rehearsal'):
            own=expectation(n,p,initial,salience);teacher=softmax(initial+np.array([n]+[0]*(dimension-1))*salience);aligned=softmax(initial+np.array([0,n]+[0]*(dimension-2))*salience);equal=expectation(n,p,initial,np.ones(dimension))
            if n==0: assert np.allclose(own,p) and np.allclose(teacher,p)
            rows.append({'strategies':dimension,'n':n,'own_strategy_mass':own.tolist(),'teacher_strategy_mass':teacher.tolist(),'own_accuracy':float(own@success),'teacher_accuracy':float(teacher@success),'aligned_teacher_accuracy':float(aligned@success),'equal_salience_accuracy':float(equal@success),'replayed_own_accuracy':float(own@success),'own_if_trap_valid':float(own@np.array([.25,.9,1.])) if dimension==3 else None})
        fig,ax=plt.subplots();subset=[r for r in rows if r['strategies']==dimension]
        for key,label in [('teacher_accuracy','Teacher: correct examples, strategy B'),('own_accuracy','Own: unjudged examples'),('aligned_teacher_accuracy','Teacher: correct examples, strategy A')]: ax.plot(NS,[r[key] for r in subset],marker='o',label=label)
        if dimension==2: ax.plot(NS,[r['equal_salience_accuracy'] for r in subset],label='Own control: equal cue salience',linestyle='--')
        else: ax.plot(NS,[r['own_if_trap_valid'] for r in subset],label='Same own contexts if strategy C is valid',linestyle='--')
        ax.set(xscale='symlog',xlabel='Additional examples N',ylabel='Expected held-out accuracy',ylim=(0,1),title='Toy: accessible strategy cues' if dimension==2 else 'Toy: a salient invalid strategy can take over');ax.legend();fig.tight_layout()
        name='task-search-strategy-toy.png' if dimension==2 else 'task-search-strategy-toy-trap.png';fig.savefig(directory/name);fig.savefig(out/name);plt.close(fig)
    (directory/'curves.json').write_text(json.dumps(rows,indent=2));(directory/'source.py').write_bytes(Path(__file__).read_bytes())
    assert all(r['own_accuracy']==r['replayed_own_accuracy'] for r in rows)
    print(json.dumps({'exact_mass_normalization_verified':True,'same_content_replay_identical':True,'endpoint': [r for r in rows if r['n']==512]},indent=2))


if __name__=='__main__': run()
