"""Finish the constrained grid, audit it, refresh the report, and release its GPU."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import psutil

from task_search_api import ROOT


def main():
    root=ROOT.resolve(); directory=root/'nontrivial_initial_count_grid'; runner=json.loads((root/'nontrivial_pipeline.json').read_text())['pid']
    while not (directory/'complete.json').exists():
        assert psutil.pid_exists(runner) and psutil.Process(runner).status()!=psutil.STATUS_ZOMBIE, 'Inference runner stopped before completion; resume its cached requests.'
        time.sleep(20)
    for script in ['experiments/analyze_nontrivial_task_search.py','reports/build_task_search_brief.py']:
        subprocess.run([sys.executable,script],check=True)
    pid=json.loads((root/'nontrivial_server.json').read_text())['pid']; server=psutil.Process(pid)
    command=server.cmdline(); assert '8017' in command and any('vllm' in word for word in command)
    processes=server.children(recursive=True)+[server]; server.terminate(); gone,alive=psutil.wait_procs(processes,timeout=30)
    assert not alive, 'The owned model server has not shut down cleanly; inspect it.'
    status={'phase':directory.name,'complete':True,'audited':True,'report_updated':True,'server_stopped':True,'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat()}
    (root/'nontrivial_initial_count_grid_status.json').write_text(json.dumps(status,indent=2))
    subprocess.run([sys.executable,'experiments/archive_nontrivial_task_search.py'],check=True)


if __name__=='__main__': main()
