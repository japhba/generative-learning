"""Audit completed confirmation phases and build their report without partial peeking."""

import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from task_search_api import ROOT


def main():
    root=ROOT.resolve(); pipeline=json.loads((root/'confirmation_pipeline.json').read_text())['pid']; code=Path(__file__).parent
    def run(name,*arguments):
        print(json.dumps({'command':[name,*arguments]}),flush=True)
        with (root/('finalize_'+name.removesuffix('.py')+'.log')).open('a') as log: subprocess.run([sys.executable,str(code/name),*arguments],stdout=log,stderr=subprocess.STDOUT,check=True)
    def wait_for(marker,process):
        while not marker.exists():
            assert psutil.pid_exists(process) and psutil.Process(process).status()!=psutil.STATUS_ZOMBIE,f'Pipeline ended before {marker}.'
            time.sleep(10)
        time.sleep(3)
    for phase in ['primary_confirmation','secondary_confirmation']:
        wait_for(root/phase/'diagnostics.json',pipeline)
        if not (root/phase/'constraint_audit.json').exists(): run('audit_task_search_constraints.py','--phase',phase)
        if not (root/phase/'verification.json').exists(): run('verify_task_search.py','--phases',phase)
        if phase=='secondary_confirmation' and not (root/phase/'question_distribution_audit.json').exists(): run('audit_generated_task_search.py','--phase',phase)
    frame_pipeline=json.loads((root/'frame_confirmation_pipeline.json').read_text())['pid']; online_pipeline=json.loads((root/'online_confirmation_pipeline.json').read_text())['pid']
    wait_for(root/'frame_confirmation/verification.json',frame_pipeline)
    wait_for(root/'palette_online_confirmation_complete.json',online_pipeline)
    decomposition_pipeline=json.loads((root/'decomposition_pipeline.json').read_text())['pid']
    wait_for(root/'decomposition_complete.json',decomposition_pipeline)
    if json.loads((root/'decomposition_confirmation_plan.json').read_text())['tasks']:
        run('analyze_decomposition_task_search.py','--phases','decomposition_confirmation')
    if not (root/'palette_online_screen/online_analysis.json').exists(): run('analyze_online_task_search.py')
    run('audit_allocation_task_search.py')
    run('build_task_search_report.py'); print('All selected-task confirmations are ready for interpretation.',flush=True)


if __name__=='__main__': main()
