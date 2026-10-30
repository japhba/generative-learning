"""Archive the grid and exactly its referenced requests; preserve the earlier study."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from tqdm.auto import tqdm

from task_search_api import ROOT


def main():
    load_dotenv(Path.home()/'.env'); root=ROOT.resolve(); project=Path(__file__).resolve().parents[1]; grid=root/'initial_count_grid'
    assert json.loads((grid/'verification.json').read_text())['all_raw_requests_and_scores_verified']
    keys=json.loads((grid/'request_keys.json').read_text()); fingerprint=json.loads((grid/'inference_manifest.json').read_text())['engine_fingerprint']; cache=root/'api'/fingerprint
    destination=Path(os.environ['CACHE_DIR'])/'generative-learning/initial_count_grid_20260911'; destination.mkdir(parents=True,exist_ok=True); archive=destination/'experiment.tar.gz'; partial=destination/'experiment.tar.gz.incomplete'
    assert not archive.exists() and not partial.exists(), 'Preserve any existing archive and inspect it before packaging again.'
    entries=[(grid,'results/task_search/initial_count_grid'),(cache/'engine.json',f'results/task_search/api/{fingerprint}/engine.json')]
    entries.extend((cache/(key+'.json'),f'results/task_search/api/{fingerprint}/{key}.json') for key in keys)
    key_set=set(keys)
    entries.extend((p,f'results/task_search/api/{fingerprint}/{p.name}') for p in cache.iterdir() if ('.failure-' in p.name or p.name.endswith('.error.json')) and p.name.split('.')[0] in key_set)
    entries.extend((p,str(p.relative_to(project))) for p in sorted((project/'experiments').glob('*initial_count_task_search.py')))
    entries.extend((p,str(p.relative_to(project))) for p in sorted((project/'reports').glob('task-search-*')) if p.is_file())
    entries.extend([(project/'reports/build_task_search_brief.py','reports/build_task_search_brief.py'),(project/'reports/initial-count-heatmap-section.html','reports/initial-count-heatmap-section.html'),(root/'initial_count_grid_status.json','results/task_search/initial_count_grid_status.json'),(root/'archive.json','results/task_search/original_study_archive.json'),(root/'initial_count_run.log','results/task_search/initial_count_run.log'),(root/'initial_count_prepare.log','results/task_search/initial_count_prepare.log'),(root/'initial_count_server.json','results/task_search/initial_count_server.json'),(root/'initial_count_pipeline.json','results/task_search/initial_count_pipeline.json')])
    entries.extend((root/name,'results/task_search/'+name) for name in ['initial_count_finalize.log','initial_count_finalizer.json'])
    with tarfile.open(partial,'w:gz',compresslevel=1,dereference=True) as bundle:
        for source,name in tqdm(entries,desc='Archive grid evidence',mininterval=5): bundle.add(source,arcname=name)
    with partial.open('rb') as stream: sha=hashlib.file_digest(stream,'sha256').hexdigest()
    partial.replace(archive)
    files=0
    aliases={'results/task_search/original_study_archive.json':root/'archive.json'}
    with tarfile.open(archive,'r|gz') as bundle:
        for member in tqdm(bundle,desc='Verify archived grid files',mininterval=5):
            if member.isdir(): continue
            assert member.isfile()
            source=aliases[member.name] if member.name in aliases else project/member.name
            with bundle.extractfile(member) as saved,source.open('rb') as current:
                assert hashlib.file_digest(saved,'sha256').digest()==hashlib.file_digest(current,'sha256').digest(),member.name
            files+=1
    record={'archive':str(archive),'sha256':sha,'bytes':archive.stat().st_size,'files_verified_against_originals':files,'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat(),'original_study_archive_unchanged':True,'excluded':'Redownloadable model weights and private build/runtime caches.'}
    (destination/'archive.json').write_text(json.dumps(record,indent=2)); (root/'initial_count_archive.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record,indent=2))


if __name__=='__main__': main()
