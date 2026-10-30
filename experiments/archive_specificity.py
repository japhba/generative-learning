"""Keep an independently verified durable copy of this cohort and its raw calls."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import os
from tqdm.auto import tqdm

from specificity_experiment import DIRECTORY


def main():
    load_dotenv(Path.home()/'.env');project=Path(__file__).resolve().parents[1]
    analysis=json.loads((DIRECTORY/'analysis.json').read_text())
    destination=Path(os.environ['CACHE_DIR'])/'generative-learning/specificity_20260912';destination.mkdir(parents=True,exist_ok=True)
    files={}
    for p in DIRECTORY.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.relative_to(DIRECTORY).parts[0]!='api': files['experiment/'+str(p.relative_to(DIRECTORY))]=p
    for reader in ['qwen','gemma']:
        fingerprint=json.loads((DIRECTORY/f'{reader}_inference.json').read_text())['engine_fingerprint']
        directory=(DIRECTORY.parent if reader=='qwen' else DIRECTORY)/'api'/fingerprint
        keys={json.loads(p.read_text())['call']['key'] for folder in ['generations','evaluations'] for p in (DIRECTORY/folder/reader).glob('*.json')}
        files[f'raw/{reader}/engine.json']=directory/'engine.json'
        for key in keys: files[f'raw/{reader}/{key}.json']=directory/f'{key}.json'
        for pattern in ['*.failure-*.json','*.error.json']:
            for p in directory.glob(pattern):
                if p.name.split('.')[0] in keys: files[f'raw/{reader}/'+p.name]=p
    for pattern in ['task-search-specificity*','task-search-report.html','specificity-section.html']:
        for p in (project/'reports').glob(pattern): files['reports/'+p.name]=p
    # Include linked earlier figures/sections so the canonical report is self-contained.
    for pattern in ['task-search-*','*-heatmap-section.html','witness-search-section.html']:
        for p in (project/'reports').glob(pattern): files['reports/'+p.name]=p
    for name in ['SPECIFICITY_PROTOCOL.md','SPECIFICITY_RESULTS.md','SPECIFICITY_SLEEP_TOY.md','specificity_experiment.py','specificity_api.py','analyze_specificity.py','report_specificity.py','specificity_sleep_toy.py','archive_specificity.py']:
        files['source/'+name]=project/'experiments'/name
    files['source/build_task_search_brief.py']=project/'reports/build_task_search_brief.py'
    hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
    manifest={'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat(),'files':hashes,'audit':analysis['audit']}
    manifest_path=DIRECTORY/'archive_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
    path=destination/'experiment.tar.gz'
    assert not path.exists(),'Never replace a durable evidence archive.'
    with tarfile.open(path,'w:gz') as archive:
        archive.add(manifest_path,arcname='manifest.json')
        for name,p in tqdm(sorted(files.items()),desc='Archive specificity evidence',mininterval=5): archive.add(p,arcname=name)
    with tarfile.open(path,'r:gz') as archive:
        checked=0
        for member in tqdm(archive,desc='Verify archived contents',total=len(hashes)+1,mininterval=5):
            if member.name=='manifest.json': continue
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==hashes[member.name],member.name
            checked+=1
        assert checked==len(hashes)
    pointer={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'verified_files':len(files),'bytes':path.stat().st_size,'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat()}
    (DIRECTORY/'archive.json').write_text(json.dumps(pointer,indent=2));print(json.dumps(pointer,indent=2))


if __name__=='__main__': main()
