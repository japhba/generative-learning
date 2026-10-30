"""Audit the completed search, refresh the stable report, release the GPU and archive."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

import psutil
from task_search_api import ROOT


def main():
    root=ROOT.resolve();runner=json.loads((root/'witness_zero_pipeline.json').read_text())['pid'];code=Path(__file__).parent
    while not (root/'witness_zero_complete.json').exists():
        assert psutil.pid_exists(runner) and psutil.Process(runner).status()!=psutil.STATUS_ZOMBIE,'Search stopped before completing the fixed plan; inspect the logs and resume cached requests.'
        time.sleep(20)
    for path in [code/'audit_witness_task_search.py',code/'report_witness_task_search.py',code.parent/'reports/build_task_search_brief.py']: subprocess.run([sys.executable,str(path)],check=True)
    for name in ['task-search-report.html','task-search-witness-screen.png','task-search-strategy-toy.png','task-search-strategy-toy-trap.png']:
        with urllib.request.urlopen('http://127.0.0.1:8095/'+name) as response: assert response.status==200 and response.read()==(code.parent/'reports'/name).read_bytes()
    pid=json.loads((root/'witness_server.json').read_text())['pid'];server=psutil.Process(pid);command=server.cmdline();assert '8017' in command and any('vllm' in s for s in command);processes=server.children(recursive=True)+[server];server.terminate();_,alive=psutil.wait_procs(processes,timeout=30);assert not alive
    (root/'witness_status.json').write_text(json.dumps({'complete':True,'audit_passed':True,'report_updated':True,'server_stopped':True,'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat()},indent=2))
    subprocess.run([sys.executable,str(code/'archive_witness_task_search.py')],check=True)


if __name__=='__main__': main()
