"""Build the fixed arm-3 evaluation sets (fresh seeds, never used in training)."""
import hashlib
import json
from pathlib import Path
from random import Random

import reasoning_gym

from mlayout_search import header
from mlayout_tasks import world as units_world
import mlayout3_search as M3

COT='\n\nSolve this new problem. Reason step by step, then finish with a final line "Answer: <answer>".\nProblem: '
DIRECT='\n\nSolve this new problem. Do not show any working: reply with only the final answer.\nProblem: '


def seed(tag,i): return int(hashlib.sha256(f'msft_eval|{tag}|{i}'.encode()).hexdigest()[:8],16)


def main():
    sets={'operator_answer_only':[],'operator_cot':[],'units_answer_only':[],'gsm_cot':[]}
    for e in range(24):
        w=M3.make_world(seed('operator',e))
        for i,t in enumerate(w['tests'][:8]):
            for name,suffix in [('operator_answer_only',DIRECT),('operator_cot',COT)]: sets[name].append({'id':f'op{e}_{i}','answer':t['answer'],'messages':[{'role':'user','content':header(w)+suffix+t['question']}]})
    for e in range(16):
        w=units_world('units',seed('units',e),4,0,8)
        for i,t in enumerate(w['tests']): sets['units_answer_only'].append({'id':f'un{e}_{i}','answer':t['answer'],'messages':[{'role':'user','content':header(w)+DIRECT+t['question']}]})
    data=reasoning_gym.create_dataset('gsm_symbolic',seed=seed('gsm',0),size=120)
    for i in range(120):
        x=data[i];sets['gsm_cot'].append({'id':f'gsm{i}','answer':str(x['answer']),'messages':[{'role':'user','content':'Solve the problem. Reason step by step, then finish with a final line "Answer: <number>".\nProblem: '+x['question']}]})
    Path('/ceph/scratch/jbauer/mlayout/msft_eval_sets.json').write_text(json.dumps(sets));print({k:len(v) for k,v in sets.items()})


if __name__=='__main__': main()
