"""Persist raw task-search evidence and code, excluding redownloadable model weights."""

import hashlib
import json
import os
from pathlib import Path
import tarfile

from dotenv import load_dotenv
from tqdm.auto import tqdm


def main():
    load_dotenv(Path.home()/'.env'); project=Path(__file__).resolve().parents[1]; root=(project/'results/task_search').resolve()
    destination=Path(os.environ['CACHE_DIR'])/'generative-learning/task_search_20260910'; destination.mkdir(parents=True,exist_ok=True)
    archive=destination/'experiment.tar.gz'; temporary=destination/'experiment.tar.gz.incomplete'
    assert not archive.exists() and not temporary.exists(), 'Existing archive or partial archive must be preserved; inspect it before packaging again.'
    # Runtime caches and weights are reproducible; their exact versions/provenance are retained.
    runtime_names=['model.json','environment.json','nvml_package.json','serve_command.json','server_restarts.jsonl','token_count_check.json','cache_check','server.log','server_missing_headers.log','server_missing_nvcc.log','server_mixed_cuda_toolkit.log','server_sampling_linker.log']
    data_entries=[p for p in sorted(root.iterdir()) if p.name not in {'models','runtime','archive.json'}]
    code_entries=sorted(set((project/'experiments').glob('*task_search*.py'))|set((project/'experiments').glob('TASK_SEARCH*.md'))|{project/'experiments/palette_teacher_control.py',project/'experiments/number_format_copy_control.py',project/'experiments/audit_palette_strategy.py'})
    code_entries += [project/'experiments/task_search_runtime']+sorted((project/'reports').glob('task-search-*.png'))
    entries=[(p,'results/task_search/'+p.name) for p in data_entries]+[(root/'runtime'/name,'results/task_search/runtime/'+name) for name in runtime_names]+[(p,str(p.relative_to(project))) for p in code_entries]
    with tarfile.open(temporary,'w:gz',compresslevel=1,dereference=True) as bundle:
        for path,arcname in tqdm(entries,desc='Archive task-search evidence',mininterval=5): bundle.add(path,arcname=arcname)
    sha=hashlib.sha256()
    with temporary.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): sha.update(block)
    temporary.replace(archive)
    record={'archive':str(archive),'sha256':sha.hexdigest(),'bytes':archive.stat().st_size,'excluded':'Downloaded model weights, build caches, extracted toolkit/NVML/header files; versions and download provenance retained.'}
    (destination/'archive.json').write_text(json.dumps(record,indent=2)); (root/'archive.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record,indent=2))


if __name__=='__main__': main()
