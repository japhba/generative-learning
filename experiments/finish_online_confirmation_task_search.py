"""Confirm all selected sequential-rehearsal candidates on fresh graph episodes."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    code=Path(__file__).parent; root=(code.parent/'results/task_search').resolve(); selection=root/'palette_online_confirmation_selection.json'; phase='palette_online_confirmation'; source=phase+'_inputs'
    pipeline=json.loads((root/'online_pipeline.json').read_text())['pid']
    while not selection.exists():
        assert psutil.pid_exists(pipeline) and psutil.Process(pipeline).status()!=psutil.STATUS_ZOMBIE,'Online screen ended without its audited candidate selection.'
        time.sleep(10)
    time.sleep(2); selected=json.loads(selection.read_text()); tasks=[r['id'] for r in selected['specifications']]; assert len(tasks)<=2
    plan={'phase':phase,'source_phase':source,'selection_sha256':hashlib.sha256(selection.read_bytes()).hexdigest(),'tasks':tasks,
          'worlds_per_task':32,'tests_per_world':16,'practice_per_world':32,'sample_counts':[0,1,2,4,8,16,32],'primary_endpoint':32,
          'arms':['teacher','self_fixed'],'generation':'Each answer sees the initial four correct examples and all earlier own unjudged rehearsal answers; questions are literally matched to the teacher.',
          'analysis':'Complete all selected episodes before inspecting confirmation accuracy. Joint one-sided episode sign-flip p is the maximum of own-versus-teacher and own-versus-initial-only p. Every selected attempt joins the six original primary attempts and the one shared-wording confirmation in the final primary Holm family.',
          'scaling':'All seven sample counts are reported. A positive endpoint alone does not establish a steeper power law. Context-censored comparisons cannot establish a positive result.'}
    path=root/'palette_online_confirmation_plan.json'
    if path.exists(): assert json.loads(path.read_text())==plan
    else: path.write_text(json.dumps(plan,indent=2))
    if not tasks:
        (root/'palette_online_confirmation_complete.json').write_text(json.dumps({'selected_tasks':[],'status':'No eligible positive screen candidate; no fresh inference performed.'},indent=2)); print('No eligible online candidate.',flush=True); return
    commands=[['strategy_task_search.py','prepare','--phase',source,'--selection',str(selection),'--worlds','32','--practice','32','--tests','16'],
              ['online_task_search.py','run','--source-phase',source,'--phase',phase,'--ns',*[str(n) for n in plan['sample_counts']],'--concurrency','16','--world-concurrency','16'],
              ['analyze_task_search.py','--phase',phase,'--endpoint','32'],
              ['-c',f"import sys; sys.path.insert(0,{str(code)!r}); from strategy_task_search import register; register(); from audit_task_search import main; main()",'--phase',phase],
              ['audit_palette_strategy.py','--phase',phase],['online_task_search.py','verify','--phase',phase]]
    for index,command in enumerate(commands):
        if index==1:
            records=json.loads((root/source/'preflight.json').read_text()); assert len(records)==32*len(tasks) and all(r['status']=='ready' for r in records)
        print(command,flush=True); executable=command if command[0]=='-c' else [str(code/command[0]),*command[1:]]
        with (root/f'online_confirmation_step_{index}.log').open('a') as log: subprocess.run([sys.executable,*executable],stdout=log,stderr=subprocess.STDOUT,check=True)
    (root/'palette_online_confirmation_complete.json').write_text(json.dumps({'selected_tasks':tasks,'status':'Fresh inference, sample audit, strategy audit, and request reconstruction completed.'},indent=2)); print('Online confirmations completed.',flush=True)


if __name__=='__main__': main()
