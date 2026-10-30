"""Pinned remote reader with resumable requests and complete, secret-free records."""
import asyncio
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx

from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'specificity'
GEMMA={'model':'google/gemma-4-26b-a4b-it','reasoning':{'enabled':False},'provider':{'only':['dekallm/bf16'],'allow_fallbacks':False,'require_parameters':True}}


class Gemma:
    def __init__(self,concurrency=16):
        load_dotenv(Path.home()/'.env')
        self.engine={'endpoint':'https://openrouter.ai/api/v1/chat/completions',**GEMMA}
        self.fingerprint=digest(self.engine);self.directory=DIRECTORY/'api'/self.fingerprint;self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'engine.json').write_text(json.dumps(self.engine,indent=2))
        self.client=httpx.AsyncClient(timeout=300,headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
        self.semaphore=asyncio.Semaphore(concurrency);self.locks={}
        self.cost=sum(json.loads(p.read_text())['response']['usage']['cost'] for p in self.directory.glob('*.json') if len(p.stem)==64)

    async def close(self): await self.client.aclose()

    async def call(self,prompt,output_schema,seed,temperature=0.,max_tokens=256):
        assert output_schema is None
        request={**GEMMA,'messages':[{'role':'user','content':prompt}],'temperature':temperature,'top_p':.9 if temperature else 1.,'max_tokens':max_tokens,'seed':seed}
        key=digest(request);path=self.directory/f'{key}.json'
        if key not in self.locks: self.locks[key]=asyncio.Lock()
        async with self.locks[key]:
            if path.exists(): record=json.loads(path.read_text())
            else:
                async with self.semaphore:
                    assert self.cost<5,'Remote budget ceiling reached; cached work is retained.'
                    for attempt in range(8):
                        started=time.time()
                        try: response=await self.client.post(self.engine['endpoint'],json=request)
                        except (httpx.TimeoutException,httpx.TransportError) as error:
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'request':request,'error':repr(error),'time':started},indent=2))
                            if attempt==7: raise
                            await asyncio.sleep(min(2**attempt,30));continue
                        if response.status_code in {408,429,500,502,503,504,520,522,524,529}:
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'request':request,'status':response.status_code,'body':response.text,'time':started},indent=2))
                            if attempt==7: response.raise_for_status()
                            await asyncio.sleep(min(2**attempt,30));continue
                        if response.is_error:
                            (self.directory/f'{key}.error.json').write_text(json.dumps({'request':request,'status':response.status_code,'body':response.text},indent=2))
                        response.raise_for_status()
                        record={'request':request,'response':response.json(),'started_unix':started,'elapsed_seconds':time.time()-started}
                        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)
                        self.cost+=record['response']['usage']['cost'];break
        response=record['response'];choice=response['choices'][0];message=choice['message']
        assert response['provider']=='DekaLLM'
        assert response['model']==GEMMA['model']
        assert message['reasoning'] is None
        assert response['usage']['completion_tokens_details']['reasoning_tokens']==0
        assert choice['finish_reason'] in {'stop','length'},choice
        return {'key':key,'status':'ok' if choice['finish_reason']=='stop' else 'truncated','value':{'answer':message['content'].strip()},'finish_reason':choice['finish_reason'],'usage':response['usage']}
