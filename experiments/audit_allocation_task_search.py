"""Independent graph constraints for every completed allocation-task phase."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from audit_task_search_constraints import answer_valid
from task_search_api import ROOT


def main():
    root=ROOT.resolve(); online=json.loads((root/'palette_online_confirmation_plan.json').read_text()); phases=['palette_screen','frame_screen','palette_online_screen','frame_confirmation']
    if online['tasks']: phases.append('palette_online_confirmation')
    for phase in phases:
        directory=root/phase; ready=[r['world'] for r in json.loads((directory/'preflight.json').read_text()) if r['status']=='ready']; assert sorted(ready)==sorted(p.name for p in (directory/'evaluations').glob('*.json'))
        teacher_count=0; changes=[]; evaluations=[]
        for name in tqdm(ready,desc='Independent allocation constraints: '+phase,mininterval=10):
            world=json.loads((directory/'worlds'/name).read_text()); result=copy.deepcopy(json.loads((directory/'evaluations'/name).read_text()))
            for entry in world['support']+world['practice']+world['tests']:
                assert answer_valid('graph_color',entry['answer'],entry),(phase,name,entry); teacher_count+=1
            for condition in result['conditions']:
                if condition['status']!='ok': continue
                for prediction,entry in zip(condition['predictions'],world['tests']):
                    native=prediction['correct']; strict=native and answer_valid('graph_color',prediction['answer'],entry)
                    if native!=strict: changes.append({'spec':world['spec']['id'],'world':name,'arm':condition['arm'],'n':condition['n'],'question':entry['question'],'prediction':prediction['answer'],'native_correct':native,'strict_correct':strict})
                    prediction['native_correct']=native; prediction['correct']=strict
                condition['accuracy']=float(np.mean([p['correct'] for p in condition['predictions']]))
            evaluations.append(result)
        record={'phase':phase,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'teacher_examples_checked':teacher_count,'scoring_changes':changes,'evaluations':evaluations,
                'metric':'Secondary sensitivity: native full credit intersected with exact graph-map keys, allowed numeric colors excluding booleans, and all edge constraints.'}
        (directory/'constraint_audit.json').write_text(json.dumps(record,indent=2)); print(json.dumps({'phase':phase,'teacher_examples_checked':teacher_count,'scoring_changes':len(changes)}))


if __name__=='__main__': main()
