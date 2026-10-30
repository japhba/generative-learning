"""Confirm every selected shared-wording candidate with frozen fresh episodes."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    code=Path(__file__).parent; root=(code.parent/'results/task_search').resolve(); phase='frame_confirmation'; selection=root/'frame_confirmation_selection.json'
    selected=json.loads(selection.read_text()); assert len(selected['specifications'])==1
    plan={'phase':phase,'selection_sha256':hashlib.sha256(selection.read_bytes()).hexdigest(),'tasks':[r['id'] for r in selected['specifications']],
          'worlds_per_task':32,'tests_per_world':16,'practice_per_world':32,'sample_counts':[0,1,2,4,8,16,32],'primary_endpoint':8,
          'arms':['teacher','self_fixed'],'generation':'Independent answers to the literally identical rehearsal questions, conditioned on the initial four correct examples only.',
          'analysis':'Finish all episodes before inspecting confirmation accuracy. Episode-paired contrasts against teacher and initial-only; joint p is the maximum of the two one-sided sign-flip p-values. Include this attempt with all six original fixed-question confirmation attempts and every additional online confirmation in the final primary Holm family.',
          'scaling':'The full N curve is descriptive. No steeper power-law claim from a positive endpoint alone.'}
    path=root/'frame_confirmation_plan.json'
    if path.exists(): assert json.loads(path.read_text())==plan
    else: path.write_text(json.dumps(plan,indent=2))
    commands=[['framed_task_search.py','prepare','--phase',phase,'--selection',str(selection),'--worlds','32','--practice','32','--tests','16'],
              ['framed_task_search.py','run','--phase',phase,'--ns',*[str(n) for n in plan['sample_counts']],'--arms','teacher','self_fixed','--concurrency','16','--world-concurrency','4'],
              ['analyze_task_search.py','--phase',phase,'--endpoint','8'],['framed_task_search.py','audit','--phase',phase],
              ['framed_task_search.py','strategy_audit','--phase',phase],['verify_task_search.py','--phases',phase]]
    for index,command in enumerate(commands):
        if index==1:
            records=json.loads((root/phase/'preflight.json').read_text()); assert len(records)==32 and all(r['status']=='ready' for r in records)
        print(command,flush=True)
        with (root/f'frame_confirmation_step_{index}.log').open('a') as log: subprocess.run([sys.executable,str(code/command[0]),*command[1:]],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Shared-wording fresh confirmation and all request audits completed.',flush=True)


if __name__=='__main__': main()
