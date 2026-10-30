"""Post-hoc answer-format sensitivity; never changes prompts or primary scoring."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_task_search import paired_interval,sign_flip_p
from task_search_api import ROOT
from witness_task_search import valid


def last_array(text):
    decoder=json.JSONDecoder();candidates=[]
    for start,character in enumerate(text):
        if character!='[': continue
        try: value,length=decoder.raw_decode(text[start:])
        except json.JSONDecodeError: continue
        if isinstance(value,list): candidates.append((start+length,-start,value))
    return json.dumps(max(candidates,key=lambda item:item[:2])[2]) if candidates else text


def main():
    root=ROOT.resolve();directory=root/'witness_format_sensitivity';directory.mkdir(exist_ok=True)
    assert last_array('Reasoning [1, 2]. Answer: [2, 1]')=='[2, 1]'
    assert last_array('[[1, 2], [3, 4]]')=='[[1, 2], [3, 4]]'
    assert last_array('no answer')=='no answer'
    plan={'motivation':'A complete manual spot check of the finished K=4 Hamiltonian confirmation found a valid final path after extra prose. Primary scoring remains strict JSON.','rule':'For completed (stop) outputs, use the last syntactically complete JSON array by ending position, preferring the outer array on a tie. If no array exists, leave the text unchanged. No correctness test is used in extraction. Truncations remain incorrect. Apply to every selected fresh K=4 and K=0 confirmation after all inference completes.','status':'Post-hoc sensitivity, not a new primary endpoint or candidate selection rule.','time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=directory/'plan.json'
    assert not path.exists(),'Preserve the original sensitivity plan.'
    path.write_text(json.dumps(plan,indent=2));(directory/'source.py').write_bytes(Path(__file__).read_bytes())


def analyze():
    root=ROOT.resolve();directory=root/'witness_format_sensitivity';plan=json.loads((directory/'plan.json').read_text());assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['source_sha256'];rows=[]
    for initial,phase,completion in [(4,'witness_confirmation','witness_complete.json'),(0,'witness_zero_confirmation','witness_zero_complete.json')]:
        done=json.loads((root/completion).read_text())
        for spec in done['selected_tasks']:
            worlds=[];changes={'zero':0,'teacher':0,'self_fixed':0}
            for path in sorted((root/phase/'evaluations').glob(spec+'__*.json')):
                result=json.loads(path.read_text());world=json.loads((root/phase/'worlds'/path.name).read_text());cells={}
                for condition in result['conditions']:
                    if condition['n'] not in [0,32]: continue
                    assert condition['status']=='ok';arm=condition['arm'];scores=[]
                    for prediction,call,query in zip(condition['predictions'],condition['calls'],world['tests']):
                        score=valid(last_array(prediction['answer']),query) if call['status']=='ok' else False
                        scores.append(score);changes[arm]+=int(score)-int(prediction['correct'])
                    cells[arm]=sum(scores)/len(scores)
                worlds.append(cells)
            own=np.array([w['self_fixed'] for w in worlds]);teacher=np.array([w['teacher'] for w in worlds]);baseline=np.array([w['zero'] for w in worlds]);dt=own-teacher;dz=own-baseline
            rows.append({'initial_examples':initial,'spec':spec,'episodes':len(worlds),'initial':float(baseline.mean()),'teacher':float(teacher.mean()),'own':float(own.mean()),'net_additional_correct':changes,'delta_teacher':float(dt.mean()),'ci95_teacher':paired_interval(dt),'delta_initial':float(dz.mean()),'ci95_initial':paired_interval(dz),'p_joint_unadjusted':max(sign_flip_p(dt),sign_flip_p(dz))})
    (directory/'analysis.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))


if __name__=='__main__':
    import sys
    {'plan':main,'analyze':analyze}[sys.argv[1]]()
