"""Complete a bounded, source-matched arithmetic strategy search and confirmation."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def write_plan(path,plan):
    if path.exists(): assert json.loads(path.read_text())==plan
    else: path.write_text(json.dumps(plan,indent=2))


def main():
    code=Path(__file__).parent; root=(code.parent/'results/task_search').resolve(); selection=root/'decomposition_screen_selection.json'; confirmation=root/'decomposition_confirmation_selection.json'
    plan={'task_operations':['sum','product'],'arities':[2,3],'digit_settings':[3,6,12],'calibration_tests':4,'initial_examples':4,
          'selection':'One magnitude per operation/arity using baseline accuracy closest to0.5, with shorter prompts breaking ties. Retain all four operation/arity settings.',
          'screen':{'worlds_per_setting':4,'tests_per_world':8,'practice_per_world':32,'sample_counts':[0,2,8,32],'endpoint':32},
          'confirmation':'At most two uncensored settings positive against both teacher and initial-only atN32, ranked by their smaller gain. Each gets32fresh episodes,16tests,32practice,N0,1,2,4,8,16,32, primary endpoint32. Include every attempt with all seven earlier primary attempts in the primary Holm family.',
          'constraints':'Identical questions, initial examples, final prompt and decoder. No generation-specific hint, test access, prior own answers, judging, filtering or feedback. Sum teachers are uniform weak compositions; product teachers draw independent positive factors and give their product as the question. All valid alternative decompositions are accepted.',
          'scaling':'Descriptive log-error window N2,4,8,16,32, same as earlier longer confirmations; no pseudocount, error floor, or asymptotic claim.',
          'model':'The same local Qwen3.5-9B checkpoint and inference manifest settings.'}
    write_plan(root/'decomposition_plan.json',plan)
    commands=[['decomposition_task_search.py','prepare','--phase','decomposition_calibration'],
              ['decomposition_task_search.py','run','--phase','decomposition_calibration','--ns','0','--arms','--concurrency','16','--world-concurrency','4'],
              ['decomposition_task_search.py','select','--phase','decomposition_calibration','--output',str(selection)],
              ['decomposition_task_search.py','prepare','--phase','decomposition_screen','--selection',str(selection),'--worlds','4','--tests','8'],
              ['decomposition_task_search.py','run','--phase','decomposition_screen','--ns','0','2','8','32','--arms','teacher','self_fixed','--concurrency','16','--world-concurrency','4'],
              ['analyze_task_search.py','--phase','decomposition_screen','--endpoint','32'],
              ['decomposition_task_search.py','audit','--phase','decomposition_screen'],['decomposition_task_search.py','strategy_audit','--phase','decomposition_screen'],
              ['verify_task_search.py','--phases','decomposition_screen']]
    def run(command,label):
        print(command,flush=True)
        with (root/f'decomposition_{label}.log').open('a') as log: subprocess.run([sys.executable,str(code/command[0]),*command[1:]],stdout=log,stderr=subprocess.STDOUT,check=True)
    for i,command in enumerate(commands): run(command,'step_'+str(i))
    path=root/'decomposition_screen/analysis.json'; analysis=json.loads(path.read_text()); candidates=[r for r in analysis['comparisons'] if r['arm']=='self_fixed' and r['delta_teacher']>0 and r['delta_zero']>0 and not r['missing']][:2]
    selected={'purpose':'Fresh confirmation of complete-screen candidates; no inference claim from the selecting screen.','source_phase':'decomposition_screen','analysis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'specifications':[json.loads((root/'decomposition_screen/specs'/f"{r['spec']}.json").read_text()) for r in candidates],'screen_comparisons':candidates}; write_plan(confirmation,selected)
    fresh={'phase':'decomposition_confirmation','selection_sha256':hashlib.sha256(confirmation.read_bytes()).hexdigest(),'tasks':[r['spec'] for r in candidates],'worlds_per_task':32,'tests_per_world':16,'practice_per_world':32,'sample_counts':[0,1,2,4,8,16,32],'primary_endpoint':32,'analysis':'Joint one-sided episode sign-flip p against teacher and initial-only, then Holm with all seven earlier primary attempts. All selected episodes finish before accuracy inspection.'}; write_plan(root/'decomposition_confirmation_plan.json',fresh)
    if candidates:
        commands=[['decomposition_task_search.py','prepare','--phase','decomposition_confirmation','--selection',str(confirmation),'--worlds','32','--tests','16'],
                  ['decomposition_task_search.py','run','--phase','decomposition_confirmation','--ns',*[str(n) for n in fresh['sample_counts']],'--arms','teacher','self_fixed','--concurrency','16','--world-concurrency','4'],
                  ['analyze_task_search.py','--phase','decomposition_confirmation','--endpoint','32'],['decomposition_task_search.py','audit','--phase','decomposition_confirmation'],
                  ['decomposition_task_search.py','strategy_audit','--phase','decomposition_confirmation'],['verify_task_search.py','--phases','decomposition_confirmation']]
        for i,command in enumerate(commands): run(command,'confirmation_step_'+str(i))
    (root/'decomposition_complete.json').write_text(json.dumps({'selected_tasks':fresh['tasks'],'status':'All planned calibration, screen, selected confirmations and audits completed.'},indent=2)); print('Decomposition study completed.',flush=True)


if __name__=='__main__': main()
