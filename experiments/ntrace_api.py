"""Pinned OpenRouter readers for the native-traces study, with one study-level billed-cost ceiling."""
import asyncio
import json
import os
import random
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx

from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'ntrace'
BUDGET=4.
NAMES={'dekallm/bf16':'DekaLLM','nextbit/bf16':'NextBit','novita/bf16':'Novita','novita/fp8':'Novita','deepinfra/fp8':'DeepInfra','deepinfra/bf16':'DeepInfra','parasail/bf16':'Parasail','nebius/fp8':'Nebius'}
SPENT=[]  # shared by every model/provider instance
RETRY={408,429,500,502,503,504,520,522,524,529}


class Remote:
    def __init__(self,model,provider,reasoning_flag=False,concurrency=24):
        load_dotenv(Path.home()/'.env')
        self.config={'model':model,'provider':{'only':[provider],'allow_fallbacks':False,'require_parameters':True}}
        if reasoning_flag: self.config['reasoning']={'enabled':False}  # hybrid-thinking models only; others reject the parameter
        self.name=NAMES[provider];self.engine={'endpoint':'https://openrouter.ai/api/v1/chat/completions',**self.config}
        self.fingerprint=digest(self.engine);self.directory=DIRECTORY/'api'/self.fingerprint;self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'engine.json').write_text(json.dumps(self.engine,indent=2))
        self.client=httpx.AsyncClient(timeout=300,headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
        self.semaphore=asyncio.Semaphore(concurrency);self.locks={}
        if not SPENT: SPENT.append(sum(json.loads(p.read_text())['response']['usage']['cost'] for p in (DIRECTORY/'api').glob('*/*.json') if len(p.stem)==64))

    async def close(self): await self.client.aclose()

    async def call(self,prompt,seed,temperature,max_tokens):
        request={**self.config,'messages':[{'role':'user','content':prompt}],'temperature':temperature,'top_p':.9 if temperature else 1.,'max_tokens':max_tokens,'seed':seed}
        key=digest(request);path=self.directory/f'{key}.json'
        if key not in self.locks: self.locks[key]=asyncio.Lock()
        async with self.locks[key]:
            if path.exists(): record=json.loads(path.read_text())
            else:
                async with self.semaphore:
                    assert SPENT[0]<BUDGET,f'Study budget ceiling ${BUDGET} reached; cached work is retained.'
                    attempt=0;limited=0
                    while True:
                        started=time.time()
                        try: response=await self.client.post(self.engine['endpoint'],json=request)
                        except (httpx.TimeoutException,httpx.TransportError) as error:
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'request':request,'error':repr(error),'time':started},indent=2))
                            attempt+=1
                            if attempt==10: raise
                            await asyncio.sleep(min(2**attempt,30));continue
                        if response.status_code==429:  # upstream rate limit: transient, retried patiently with jitter (never fatal before ~35 min)
                            limited+=1;(self.directory/f'{key}.failure-r{limited}.json').write_text(json.dumps({'request':request,'status':429,'body':response.text,'time':started},indent=2))
                            if limited==40: response.raise_for_status()
                            await asyncio.sleep(min(2**limited,60)*(.5+random.random()));continue
                        if response.status_code in RETRY:
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'request':request,'status':response.status_code,'body':response.text,'time':started},indent=2))
                            attempt+=1
                            if attempt==10: response.raise_for_status()
                            await asyncio.sleep(min(2**attempt,30));continue
                        if response.is_error: (self.directory/f'{key}.error.json').write_text(json.dumps({'request':request,'status':response.status_code,'body':response.text},indent=2))
                        response.raise_for_status()
                        try: body=response.json()
                        except json.JSONDecodeError: body={'unparseable_body':response.text}
                        if 'choices' not in body:  # 200 with an upstream error object or broken body: transient
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'request':request,'status':200,'body':body,'time':started},indent=2))
                            attempt+=1
                            if attempt==10: raise RuntimeError(body)
                            await asyncio.sleep(min(2**attempt,30));continue
                        record={'request':request,'response':body,'started_unix':started,'elapsed_seconds':time.time()-started}
                        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)
                        SPENT[0]+=record['response']['usage']['cost'];break
        response=record['response'];choice=response['choices'][0];message=choice['message']
        assert response['provider']==self.name,(response['provider'],self.name)
        assert not message.get('reasoning'),message.get('reasoning')
        assert choice['finish_reason'] in {'stop','length'},choice
        if message['content'] is None: return {'key':key,'status':'content_missing','text':'','usage':response['usage']}  # provider dropped the text (seen: Novita Qwen3-Coder); counted in the report
        return {'key':key,'status':'ok' if choice['finish_reason']=='stop' else 'truncated','text':message['content'].strip(),'usage':response['usage']}
