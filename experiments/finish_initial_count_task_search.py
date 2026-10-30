"""Finish auditing and reporting after the entire fixed grid has completed."""
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
    root=ROOT.resolve(); project=Path(__file__).resolve().parents[1]; pipeline=json.loads((root/'initial_count_pipeline.json').read_text())
    while not (root/'initial_count_grid/complete.json').exists():
        assert psutil.pid_exists(pipeline['pid']), 'Grid inference exited without completing; inspect initial_count_run.log.'
        time.sleep(20)
    for relative in ['experiments/analyze_initial_count_task_search.py','reports/build_task_search_brief.py']:
        print('Run '+relative,flush=True); subprocess.run([sys.executable,str(project/relative)],check=True)
    record={'status':'All grid inference, exact request/scoring verification, heatmaps, curves, and HTML report completed.','time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat(),'initial_counts':[0,1,2,4,8,16],'rehearsal_counts':[0,1,2,4,8,16,32],'episodes_per_cell':16,'test_queries_per_cell':256,'plot_paths':['reports/task-search-teacher-heatmap.png','reports/task-search-own-heatmap.png'],'report_url':'http://127.0.0.1:8095/task-search-report.html','server_cleanup':'Pending root-agent cleanup; server launch PID in initial_count_server.json.','archive':'Pending root-agent packaging and verification.'}
    (root/'initial_count_grid_status.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record,indent=2))


if __name__=='__main__': main()
