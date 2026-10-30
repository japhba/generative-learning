"""Finish the frozen confirmation cohorts and audit only after inference completes."""

import json
from pathlib import Path
import subprocess
import sys
import time

import httpx

from task_search_api import ROOT


def main():
    root=ROOT.resolve(); plan=json.loads((root/'confirmation_plan.json').read_text()); code=Path(__file__).parent
    deadline=time.monotonic()+900
    with httpx.Client(timeout=5) as client:
        while True:
            try: response=client.get('http://127.0.0.1:8017/health')
            except httpx.ConnectError:
                assert time.monotonic()<deadline,'Local server did not become ready.'
                time.sleep(5); continue
            response.raise_for_status(); break
    for phase in plan['phases']:
        name=phase['phase']; records=json.loads((root/name/'preflight.json').read_text())
        assert len(records)==len(phase['tasks'])*phase['worlds_per_task']
        commands=[['run_task_search.py','--phase',name,'--arms',*phase['arms'],'--ns',*[str(n) for n in plan['sample_counts']]],
                  ['analyze_task_search.py','--phase',name,'--endpoint','8'],['audit_task_search.py','--phase',name]]
        for command in commands:
            print(json.dumps({'phase':name,'command':command}),flush=True)
            with (root/(name+'_'+Path(command[0]).stem+'.log')).open('a') as log:
                subprocess.run([sys.executable,str(code/command[0]),*command[1:]],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('All frozen confirmations and native-score analyses completed.',flush=True)


if __name__=='__main__': main()
