"""Archive completed tracking runs with file-by-file size and SHA256 verification."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
from tqdm.auto import tqdm

from train_feedback import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    load_dotenv(Path.home()/'.env')
    assert not args.manifest.exists()
    plan=[]
    for directory in args.runs:
        config=json.loads((directory/'config.json').read_text())
        source=Path(config['cache_directory'])
        destination=Path(os.environ['CACHE_DIR'])/'generative-learning/tracking'/directory.name
        assert source.is_dir() and not destination.exists()
        for arm in ('warm',*config['arms']):
            final=json.loads((directory/f'{arm}_final.json').read_text())
            assert Path(final['checkpoint'])==source/arm
            assert sha256(source/arm/'model.safetensors')==final['model_sha256']
            if arm=='warm': assert sha256(source/arm/'optimizer.pt')==final['optimizer_sha256']
            else:
                assert final['counters']['labels_acquired']==config['labels']
                assert final['counters']['training_examples']==2*config['labels']
        probes=list((source/'weight_probes').glob('*.npz'))
        assert len(probes)==len(config['arms'])*len(config['checkpoints'])
        plan.append((directory,source,destination))
    manifest={'verified_at_utc':None,'runs':[],'files':[]}
    for directory,source,destination in tqdm(plan,desc='Archive tracking runs'):
        files=sorted(p.relative_to(source) for p in source.rglob('*') if p.is_file())
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,destination)
        assert sorted(p.relative_to(destination) for p in destination.rglob('*') if p.is_file())==files
        for relative in tqdm(files,desc=directory.name,leave=False):
            original,archived=source/relative,destination/relative
            size,digest=original.stat().st_size,sha256(original)
            assert archived.stat().st_size==size and sha256(archived)==digest
            manifest['files'].append({'run':directory.name,'source':str(original),'archive':str(archived),'bytes':size,'sha256':digest})
        manifest['runs'].append({'run':str(directory.resolve()),'source':str(source),'archive':str(destination),'files':len(files)})
    manifest['total_bytes']=sum(row['bytes'] for row in manifest['files'])
    manifest['verified_at_utc']=datetime.now(timezone.utc).isoformat()
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"Archived and verified {len(manifest['files'])} files, {manifest['total_bytes']} bytes.")


if __name__=='__main__': main()
