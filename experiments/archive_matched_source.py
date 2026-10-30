"""Archive the completed source-only study to persistent cache without deleting data."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from tqdm.auto import tqdm

PROJECT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): value.update(chunk)
    return value.hexdigest()


def main():
    load_dotenv(Path.home()/'.env'); root=(PROJECT/'results/matched_source_icl').resolve()
    spending=json.loads((root/'api/spending.json').read_text()); verification=json.loads((root/'verification.json').read_text())
    assert verification['spending']==spending
    names=['matched_source_tasks.py','run_matched_source.py','analyze_matched_source.py','verify_matched_source.py','evaluate_matched_traces.py',
           'test_matched_source.py','run_matched_bbh.py','prepare_matched_bbh_inputs.py','replay_matched_bbh_labels.py','build_matched_source_report.py',
           'archive_matched_source.py','inverse_icl_api.py','self_icl_api.py','MATCHED_SOURCE_PROTOCOL.md','MATCHED_SOURCE_RESULTS.md','README.md']
    deliverable=root/'deliverable'; (deliverable/'experiments').mkdir(parents=True); (deliverable/'reports').mkdir()
    for name in names: shutil.copy2(PROJECT/'experiments'/name,deliverable/'experiments'/name)
    for name in ('pyproject.toml','uv.lock'): shutil.copy2(PROJECT/name,deliverable/name)
    for path in sorted((PROJECT/'reports').glob('matched-source-*.png')): shutil.copy2(path,deliverable/'reports'/path.name)
    audits={str(p.relative_to(root)):json.loads(p.read_text()) for p in sorted(root.glob('*/verification.json'))}
    assert {'trace_screen/verification.json','trace_confirmation/verification.json','bbh_screen/verification.json','bbh_confirmation/verification.json','bbh_label_replay/verification.json'}<=set(audits)
    completion={'completed_at':datetime.now(ZoneInfo('Europe/London')).isoformat(),'spending':spending,'offline_tests':11,
                'selection_hashes_verified':3,'source_only_verification':verification,'additional_audits':audits,
                'case_counts':{p.name:len(list((p/'cases').glob('*.json'))) for p in sorted(root.iterdir()) if (p/'cases').is_dir()},
                'deliverable_sha256':{str(p.relative_to(deliverable)):digest(p) for p in sorted(deliverable.rglob('*')) if p.is_file()},
                'conclusion':'No confirmed primary source-only generation advantage or useful steeper generated learning law in this search. Secondary Gemini hidden-affine benefit is consistent with higher solution quality; both learners favor Gemini. BBH parser replay leaves endpoints unchanged.'}
    with (root/'completion.json').open('x') as stream: json.dump(completion,stream,indent=2)
    (root/'ARCHIVE_README.md').write_text('''# Source-only rehearsal experiment

All current-study raw requests, responses, spending records, selections, task
worlds, verification results, and PNG learning curves are retained. No generated
answer was filtered or repaired using correctness. The separate BBH label-parser
replay is post hoc and reuses the original generations. The early interface run
is retained for provenance and is excluded from confirmation claims.

The complete report and protocol are in deliverable/experiments. Current code,
its API dependencies, pyproject.toml, and uv.lock are included. Restore the data
at results/matched_source_icl in a project checkout and place deliverable's
experiments and reports alongside it to use the documented commands. Use the
existing uv environment outside NFS; load API credentials from ~/.env. Offline
analysis and audits need no API calls. Earlier unrelated experiment data are
not part of this archive, although experiments/README.md links to their reports.

completion.json records checks, cohort counts, costs, and code/plot hashes.
The archive checksum is in the archive.json sidecar next to the persistent tar
file and in the local data directory; it cannot be embedded in its own checksum.
''')
    local=root.parent/(root.name+'.tar.gz'); persistent=Path(os.environ['CACHE_DIR'])/'generative-learning'/root.name
    persistent.mkdir(parents=True,exist_ok=True); destination=persistent/'experiment.tar.gz'
    files=sorted(p for p in root.rglob('*') if p.is_file())
    with tarfile.open(local,'x:gz') as archive:
        for path in tqdm(files,desc='Archive source-only study',unit='file'): archive.add(path,arcname=str(Path(root.name)/path.relative_to(root)),recursive=False)
    with local.open('rb') as source,destination.open('xb') as target: shutil.copyfileobj(source,target,8*1024*1024)
    checksum=digest(local); assert digest(destination)==checksum
    with tarfile.open(destination,'r:gz') as archive:
        members=archive.getmembers(); assert len(members)==len(files)
        expected=str(Path(root.name)/'deliverable/experiments/MATCHED_SOURCE_RESULTS.md')
        assert archive.extractfile(expected).read()==(PROJECT/'experiments/MATCHED_SOURCE_RESULTS.md').read_bytes()
    metadata={'created_at':datetime.now(ZoneInfo('Europe/London')).isoformat(),'local_archive':str(local),'persistent_archive':str(destination),
              'bytes':destination.stat().st_size,'sha256':checksum,'members':len(files),'verified_persistent_checksum':True,'verified_report_contents':True}
    for path in (root/'archive.json',persistent/'archive.json'):
        with path.open('x') as stream: json.dump(metadata,stream,indent=2)
    print(json.dumps(metadata,indent=2))


if __name__=='__main__': main()
