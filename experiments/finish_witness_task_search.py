"""Complete a frozen constructive-task screen and fresh confirmation of at most two candidates."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from task_search_api import ROOT


def freeze(path,value):
    if path.exists(): assert json.loads(path.read_text())==value
    else: path.write_text(json.dumps(value,indent=2))


def main():
    root=ROOT.resolve();code=Path(__file__).parent
    plan={'families':['bezout','congruences','linear_allocation','three_squares','rational_interval','balanced_sat','perfect_matching','hamiltonian_path','topological_order','equal_partition','supersequence','register_program'],'levels':[0,1],'initial_examples':4,'calibration_tests_per_level':8,
          'calibration_selection':'One setting per family by baseline accuracy closest to 0.5; lower prompt length then spec ID break ties. Every family retained. No source outcome is used.',
          'screen':{'episodes_per_family':4,'tests_per_episode':8,'sample_counts':[0,2,8,32],'primary_endpoint':32},
          'confirmation':{'selection':'At most two uncensored screen candidates strictly exceeding teacher and initial-only at N=32, ranked by their smaller gain; deterministic spec tie break. All other screen results retained.','episodes_per_selected_family':16,'tests_per_episode':16,'sample_counts':[0,1,2,4,8,16,32],'primary_endpoint':32,'interpretation':'Fresh confirmation after exploratory selection. Joint one-sided episode sign-flip test against teacher and initial-only, Holm across every selected attempt; also show correction across all 12 screened families and the nine prior local primary attempts.'},
          'controls':'Only additional answer strings change between source conditions. Same assigned questions and order, initial pairs, native final prompt, model checkpoint, decoding and token allowance. Independent own generation sees only initial examples and its assigned question. No judgments, repairs, retries for correctness, deduplication of generations, test access or source hints.',
          'scaling':'Report every curve; descriptive log-error slope on N=2,4,8,16,32, with no error floor or pseudocount. No assumed power law.'}
    freeze(root/'witness_plan.json',plan)
    def run(script,*args):
        label=script.removesuffix('.py')+'_'+'_'.join(str(a) for a in args if not str(a).startswith('--')).replace('/','_')
        with (root/('witness_'+hashlib.sha256(label.encode()).hexdigest()[:12]+'.log')).open('a') as log:
            subprocess.run([sys.executable,str(code/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
    run('witness_task_search.py','prepare','--phase','witness_calibration','--tests',8)
    run('witness_task_search.py','run','--phase','witness_calibration','--ns',0,'--arms','--concurrency',16,'--world-concurrency',16)
    directory=root/'witness_calibration';summary=json.loads((directory/'summary.json').read_text());pre=json.loads((directory/'preflight.json').read_text());ready={r['spec']:r for r in pre if r['status']=='ready'};assert len(summary)==len(ready)
    selected=[];decisions=[]
    for family in plan['families']:
        candidates=[r for r in summary if r['task'].rsplit('_l',1)[0]=='witness_'+family];assert candidates
        best=min(candidates,key=lambda r:(abs(r['conditions'][0]['accuracy']-.5),ready[r['spec']]['max_teacher_prompt_tokens'],r['spec']))
        selected.append(json.loads((directory/'specs'/f"{best['spec']}.json").read_text()));decisions.append({'family':family,'selected':best['spec'],'baseline_candidates':candidates})
    selection=root/'witness_screen_selection.json';freeze(selection,{'specifications':selected,'decisions':decisions,'calibration_summary_sha256':hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest()})
    run('witness_task_search.py','prepare','--phase','witness_screen','--selection',selection,'--worlds',4,'--tests',8)
    run('witness_task_search.py','run','--phase','witness_screen','--ns',0,2,8,32,'--arms','teacher','self_fixed','--concurrency',16,'--world-concurrency',16)
    run('analyze_task_search.py','--phase','witness_screen','--endpoint',32)
    path=root/'witness_screen/analysis.json';screen=json.loads(path.read_text());assert screen['worlds']==48
    candidates=[r for r in screen['comparisons'] if r['arm']=='self_fixed' and r['delta_teacher']>0 and r['delta_zero']>0 and not r['missing']]
    candidates=sorted(candidates,key=lambda r:(-min(r['delta_teacher'],r['delta_zero']),r['spec']))[:2]
    confirmation=root/'witness_confirmation_selection.json';freeze(confirmation,{'specifications':[json.loads((root/'witness_screen/specs'/f"{r['spec']}.json").read_text()) for r in candidates],'screen_comparisons':candidates,'screen_analysis_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if candidates:
        run('witness_task_search.py','prepare','--phase','witness_confirmation','--selection',confirmation,'--worlds',16,'--tests',16)
        run('witness_task_search.py','run','--phase','witness_confirmation','--ns',0,1,2,4,8,16,32,'--arms','teacher','self_fixed','--concurrency',16,'--world-concurrency',16)
        run('analyze_task_search.py','--phase','witness_confirmation','--endpoint',32)
    (root/'witness_complete.json').write_text(json.dumps({'selected_tasks':[r['spec'] for r in candidates],'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat()},indent=2))


if __name__=='__main__': main()
