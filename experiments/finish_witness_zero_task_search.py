"""A separately frozen zero-initial-example comparison motivated by Self-ICL."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import psutil
from task_search_api import ROOT
from finish_witness_task_search import freeze


def main():
    root=ROOT.resolve();code=Path(__file__).parent
    plan={'initial_examples':0,'motivation':'Self-ICL starts from zero gold demonstrations. Initial pairs may already elicit the representation otherwise supplied by rehearsal. Fixed before this follow-up inference and before seeing four-initial-example confirmation outcomes.','difficulty':'All twelve previously calibrated families and settings, with new seeds. No selection on zero-initial-example source outcomes.','screen':{'episodes_per_family':4,'tests_per_episode':8,'sample_counts':[0,2,8,32],'endpoint':32},'confirmation':{'maximum_candidates':1,'rule':'Highest smaller gain over teacher and initial-only, strictly positive against both at N=32; retain all failed screens.','episodes':16,'tests_per_episode':16,'sample_counts':[0,1,2,4,8,16,32],'endpoint':32},'constraints':'Only additional answer strings change within each comparison; no hints, judgments, feedback, repair or correctness filtering. Initial-only is task instruction alone.','analysis':'Joint one-sided episode sign-flip tests, Holm with every current four/zero-initial confirmation and nine prior local primary attempts. Also report conservative Bonferroni over 33 settings: 24 new K/task screens plus nine earlier primary attempts.'}
    freeze(root/'witness_zero_plan.json',plan);base_pid=json.loads((root/'witness_pipeline.json').read_text())['pid']
    while not (root/'witness_complete.json').exists():
        assert psutil.pid_exists(base_pid) and psutil.Process(base_pid).status()!=psutil.STATUS_ZOMBIE
        time.sleep(20)
    def run(script,*args):
        key=hashlib.sha256(json.dumps([script,*map(str,args)]).encode()).hexdigest()[:12]
        with (root/f'witness_{key}.log').open('a') as log: subprocess.run([sys.executable,str(code/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
    run('witness_zero_task_search.py','prepare','--phase','witness_zero_screen','--selection',root/'witness_screen_selection.json','--worlds',4,'--tests',8)
    run('witness_zero_task_search.py','run','--phase','witness_zero_screen','--ns',0,2,8,32,'--arms','teacher','self_fixed','--concurrency',16,'--world-concurrency',16)
    run('analyze_task_search.py','--phase','witness_zero_screen','--endpoint',32)
    path=root/'witness_zero_screen/analysis.json';screen=json.loads(path.read_text());assert screen['worlds']==48
    candidates=[r for r in screen['comparisons'] if r['arm']=='self_fixed' and r['delta_teacher']>0 and r['delta_zero']>0 and not r['missing']];candidates=sorted(candidates,key=lambda r:(-min(r['delta_teacher'],r['delta_zero']),r['spec']))[:1]
    selected=root/'witness_zero_confirmation_selection.json';freeze(selected,{'specifications':[json.loads((root/'witness_zero_screen/specs'/f"{r['spec']}.json").read_text()) for r in candidates],'screen_comparisons':candidates,'screen_analysis_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if candidates:
        run('witness_zero_task_search.py','prepare','--phase','witness_zero_confirmation','--selection',selected,'--worlds',16,'--tests',16)
        run('witness_zero_task_search.py','run','--phase','witness_zero_confirmation','--ns',0,1,2,4,8,16,32,'--arms','teacher','self_fixed','--concurrency',16,'--world-concurrency',16)
        run('analyze_task_search.py','--phase','witness_zero_confirmation','--endpoint',32)
    (root/'witness_zero_complete.json').write_text(json.dumps({'selected_tasks':[r['spec'] for r in candidates],'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat()},indent=2))


if __name__=='__main__': main()
