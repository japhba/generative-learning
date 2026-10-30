"""Fetch and preserve the original BBH question files at a pinned commit."""

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]/'results/matched_source_icl/bbh_inputs'


async def main():
    cached=json.loads((ROOT/'tasks.json').read_text()); provenance=json.loads((ROOT/'provenance.json').read_text())
    raw=ROOT/'original'; raw.mkdir(exist_ok=True); progress=tqdm(total=len(cached),desc='Original BBH')
    async with httpx.AsyncClient(timeout=60) as client:
        async def fetch(name):
            url=f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{provenance['bbh_commit']}/bbh/{name}.json"
            response=await client.get(url); response.raise_for_status(); payload=response.content
            with (raw/f'{name}.json').open('xb') as stream: stream.write(payload)
            progress.update()
            return name,{'description':cached[name]['description'],'examples':json.loads(payload)['examples']},url,hashlib.sha256(payload).hexdigest()
        fetched=await asyncio.gather(*(fetch(name) for name in sorted(cached)))
    progress.close(); tasks={name:task for name,task,_,_ in fetched}
    with (ROOT/'original_tasks.json').open('x') as stream: json.dump(tasks,stream,indent=2)
    metadata={'bbh_commit':provenance['bbh_commit'],'description_source':provenance['self_icl_commit'],
              'files':{name:{'url':url,'sha256':digest} for name,_,url,digest in fetched},
              'tasks_sha256':hashlib.sha256((ROOT/'original_tasks.json').read_bytes()).hexdigest(),
              'reason':'Use original BBH inputs and labels for every task. Both the original release and the cache contain three invalid option lists; record these before sampling and exclude them identically for both source arms.'}
    with (ROOT/'original_provenance.json').open('x') as stream: json.dump(metadata,stream,indent=2)
    print(json.dumps({'tasks':len(tasks),'examples':sum(len(t['examples']) for t in tasks.values()),'sha256':metadata['tasks_sha256']},indent=2))


if __name__=='__main__': asyncio.run(main())
