"""Archive the new search and exact toy with their complete referenced raw requests."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from tqdm.auto import tqdm

from task_search_api import ROOT


def main():
    load_dotenv(Path.home()/'.env');root=ROOT.resolve();project=Path(__file__).resolve().parents[1];done=json.loads((root/'witness_complete.json').read_text());phases=['witness_calibration','witness_screen']+(['witness_confirmation'] if done['selected_tasks'] else []);zero=json.loads((root/'witness_zero_complete.json').read_text());phases+=['witness_zero_screen']+(['witness_zero_confirmation'] if zero['selected_tasks'] else [])
    destination=Path(os.environ['CACHE_DIR'])/'generative-learning/witness_search_20260911';destination.mkdir(parents=True,exist_ok=True);archive=destination/'experiment.tar.gz';partial=destination/'experiment.tar.gz.incomplete';assert not archive.exists() and not partial.exists()
    engine=json.loads((root/'witness_screen/inference_manifest.json').read_text())['engine_fingerprint'];cache=root/'api'/engine;keys=json.loads((root/'witness_request_keys.json').read_text());entries=[(root/phase,'results/task_search/'+phase) for phase in phases+['witness_analysis','strategy_context_toy','witness_format_sensitivity']]
    entries.extend((cache/(key+'.json'),f'results/task_search/api/{engine}/{key}.json') for key in keys);entries.append((cache/'engine.json',f'results/task_search/api/{engine}/engine.json'))
    keyset=set(keys);entries.extend((p,f'results/task_search/api/{engine}/{p.name}') for p in cache.iterdir() if '.failure-' in p.name and p.name.split('.')[0] in keyset)
    entries.extend((p,'results/task_search/'+p.name) for p in root.glob('witness_*.json'))
    entries.extend((p,'results/task_search/'+p.name) for p in root.glob('witness_*.log') if re.fullmatch(r'witness_[0-9a-f]{12}\.log',p.name) or p.name in {'witness_pipeline.log','witness_zero_pipeline.log'})
    entries.extend((p,'experiments/'+p.name) for p in (project/'experiments').glob('*witness*.py'))
    entries.extend((project/'experiments'/name,'experiments/'+name) for name in ['WITNESS_TASK_SEARCH_PROTOCOL.md','STRATEGY_CONTEXT_TOY.md','strategy_context_toy.py','LINEAR_ELICITATION_TOY.md','analyze_task_search.py','verify_task_search.py'])
    entries.extend((p,'reports/'+p.name) for p in (project/'reports').glob('task-search-*') if p.is_file())
    entries.extend((project/'reports'/name,'reports/'+name) for name in ['build_task_search_brief.py','witness-search-section.html'])
    entries.append((root/'literature/self_icl_2023.pdf','results/task_search/literature/self_icl_2023.pdf'))
    entries.extend((root/name,'results/task_search/'+name) for name in ['archive.json','initial_count_archive.json','nontrivial_archive.json'])
    with tarfile.open(partial,'w:gz',compresslevel=1,dereference=True) as bundle:
        for source,name in tqdm(entries,desc='Archive constructive search',mininterval=5): bundle.add(source,arcname=name)
    with partial.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    partial.replace(archive);count=0
    with tarfile.open(archive,'r|gz') as bundle:
        for member in tqdm(bundle,desc='Verify archive',mininterval=5):
            if member.isdir(): continue
            assert member.isfile()
            with bundle.extractfile(member) as saved,(project/member.name).open('rb') as current: assert hashlib.file_digest(saved,'sha256').digest()==hashlib.file_digest(current,'sha256').digest(),member.name
            count+=1
    record={'archive':str(archive),'sha256':sha,'files_verified':count,'bytes':archive.stat().st_size,'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat(),'earlier_archives_preserved':True};(root/'witness_archive.json').write_text(json.dumps(record,indent=2));(destination/'archive.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__': main()
