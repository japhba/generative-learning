"""Complete the online palette screen and select only uncensored fresh-test candidates."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    code=Path(__file__).parent; root=(code.parent/'results/task_search').resolve(); phase='palette_online_screen'
    commands=[('run',[str(code/'online_task_search.py'),'run','--phase',phase,'--concurrency','16','--world-concurrency','16']),
              ('analyze',[str(code/'analyze_task_search.py'),'--phase',phase,'--endpoint','32']),
              ('audit',['-c',f"import sys; sys.path.insert(0,{str(code)!r}); from strategy_task_search import register; register(); from audit_task_search import main; main()",'--phase',phase]),
              ('strategy_audit',[str(code/'audit_palette_strategy.py'),'--phase',phase]),
              ('verify',[str(code/'online_task_search.py'),'verify','--phase',phase])]
    for label,command in commands:
        print(label,flush=True)
        with (root/f'palette_online_{label}.log').open('a') as log: subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,check=True)
    path=root/phase/'analysis.json'; analysis=json.loads(path.read_text()); candidates=[r for r in analysis['comparisons'] if r['arm']=='self_fixed' and r['delta_teacher']>0 and r['delta_zero']>0 and not r['missing']][:2]
    selection={'purpose':'Fresh confirmation of sequential unjudged rehearsal candidates; no inference claim from this exploratory screen.','source_phase':phase,'endpoint':32,'analysis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'eligibility':'Positive relative to both teacher and initial-only, no source-context censoring, ranked by the smaller gain.','specifications':[json.loads((root/phase/'specs'/f"{r['spec']}.json").read_text()) for r in candidates],'screen_comparisons':candidates}
    (root/'palette_online_confirmation_selection.json').write_text(json.dumps(selection,indent=2)); print(json.dumps(selection,indent=2),flush=True)


if __name__=='__main__': main()
