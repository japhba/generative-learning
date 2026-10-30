"""Pinned OpenRouter Gemma reader for the cross-task specificity search, with a study-level billed-cost ceiling."""
import asyncio
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx

from specificity_api import GEMMA
from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'xspec'
BUDGET=9.
NAMES={'dekallm/bf16':'DekaLLM','nextbit/bf16':'NextBit','novita/bf16':'Novita'}
SPENT=[]  # one study-wide billed-cost counter shared by every provider instance


class Gemma:
    def __init__(self,concurrency=24,provider='dekallm/bf16'):
        load_dotenv(Path.home()/'.env')
        self.config={**GEMMA,'provider':{**GEMMA['provider'],'only':[provider]}};self.name=NAMES[provider]  # dekallm/bf16 reproduces GEMMA exactly
        self.engine={'endpoint':'https://openrouter.ai/api/v1/chat/completions',**self.config}
        self.fingerprint=digest(self.engine);self.directory=DIRECTORY/'api'/self.fingerprint;self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'engine.json').write_text(json.dumps(self.engine,indent=2))
        self.client=httpx.AsyncClient(timeout=300,headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
        self.semaphore=asyncio.Semaphore(concurrency);self.locks={}
        if not SPENT: SPENT.append(sum(json.loads(p.read_text())['response']['usage']['cost'] for p in (DIRECTORY/'api').glob('*/*.json') if len(p.stem)==64))

    @property
    def cost(self): return SPENT[0]

    async def close(self): await self.client.aclose()

    async def call(self,prompt,output_schema,seed,temperature=0.,max_tokens=256):
        assert output_schema is None
        request={**self.config,'messages':[{'role':'user','content':prompt}],'temperature':temperature,'top_p':.9 if temperature else 1.,'max_tokens':max_tokens,'seed':seed}
        key=digest(request);path=self.directory/f'{key}.json'
        if key not in self.locks: self.locks[key]=asyncio.Lock()
        async with self.locks[key]:
            if path.exists(): record=json.loads(path.read_text())
            else:
                async with self.semaphore:
                    assert SPENT[0]<BUDGET,f'Study budget ceiling ${BUDGET} reached; cached work is retained.'
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
                        try: body=response.json()
                        except json.JSONDecodeError: body={'unparseable_body':response.text}  # transient: 200 with a broken body
                        if 'choices' not in body:  # OpenRouter occasionally returns a 200 with an upstream error object or broken body
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'request':request,'status':200,'body':body,'time':started},indent=2))
                            if attempt==7: raise RuntimeError(body)
                            await asyncio.sleep(min(2**attempt,30));continue
                        record={'request':request,'response':body,'started_unix':started,'elapsed_seconds':time.time()-started}
                        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)
                        SPENT[0]+=record['response']['usage']['cost'];break
        response=record['response'];choice=response['choices'][0];message=choice['message']
        assert response['provider']==self.name,(response['provider'],self.name)
        assert response['model']==GEMMA['model']
        assert message['reasoning'] is None
        assert response['usage']['completion_tokens_details']['reasoning_tokens']==0
        assert choice['finish_reason'] in {'stop','length'},choice
        return {'key':key,'status':'ok' if choice['finish_reason']=='stop' else 'truncated','value':{'answer':message['content'].strip()},'finish_reason':choice['finish_reason'],'usage':response['usage']}
