"""Replay identical deterministic requests with unique cache salts as an engine check."""

import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx
from tqdm.auto import tqdm

from task_search_api import ROOT


async def main():
    root=ROOT.resolve(); directory=root/'runtime/cache_check'; directory.mkdir(exist_ok=True); sources=[]
    for phase,spec,arm,n in [('palette_calibration','graph_palette_v6_c12_p40__default','zero',0),('screen','number_format__default','teacher',8),('screen','number_format__default','self_fixed',8)]:
        p=root/phase; engine=json.loads((p/'inference_manifest.json').read_text())['engine_fingerprint']; files=sorted((p/'evaluations').glob(spec+'__*.json'))
        for file in files:
            result=json.loads(file.read_text()); condition=next(c for c in result['conditions'] if c['arm']==arm and c['n']==n)
            # Fixed query IDs, not selected for disagreement or correctness.
            for prediction in condition['predictions']:
                if prediction['id'] not in {1,2}: continue
                sources.append({'phase':phase,'spec':spec,'seed':result['seed'],'arm':arm,'n':n,'id':prediction['id'],'key':prediction['api_key'],'engine':engine})
    manifest={'purpose':'Numerical/backend sanity check only. Identical text and deterministic decoding; a unique request cache_salt prevents using a previously cached prompt prefix.','sources':sources,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)); semaphore=asyncio.Semaphore(2)
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8017',timeout=600) as client:
        async def one(source):
            path=directory/(source['key']+'.json')
            if path.exists(): return json.loads(path.read_text())
            original=json.loads((root/'api'/source['engine']/(source['key']+'.json')).read_text()); request={**original['request'],'cache_salt':'task-search-independent-prefix-'+source['key']}
            async with semaphore:
                started=time.time(); response=await client.post('/v1/chat/completions',json=request); response.raise_for_status(); fresh=response.json()
            record={'source':source,'request':request,'original_response':original['response'],'fresh_response':fresh,'elapsed_seconds':time.time()-started,
                    'exact_content_match':original['response']['choices'][0]['message']['content']==fresh['choices'][0]['message']['content']}
            path.write_text(json.dumps(record,indent=2)); return record
        results=[]
        for future in tqdm(asyncio.as_completed([one(s) for s in sources]),total=len(sources),desc='Independent prefix check',mininterval=10): results.append(await future)
    summary={'requests':len(results),'exact_matches':sum(r['exact_content_match'] for r in results),'nonmatching_keys':[r['source']['key'] for r in results if not r['exact_content_match']]}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__=='__main__': asyncio.run(main())
