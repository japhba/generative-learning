"""Resumable local inference with complete requests, responses, and engine provenance."""

import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import httpx
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]/'results/task_search'


def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def schema(fields):
    return {'type':'object','properties':{field:{'type':'string'} for field in fields},'required':list(fields),'additionalProperties':False}


def answers_schema(count):
    return {'type':'object','properties':{'answers':{'type':'array','items':{'type':'string'},'minItems':count,'maxItems':count}},'required':['answers'],'additionalProperties':False}


class LocalModel:
    def __init__(self,concurrency=16):
        self.root=ROOT.resolve(); runtime=self.root/'runtime'
        model=json.loads((runtime/'model.json').read_text()); server=json.loads((runtime/'serve_command.json').read_text())
        self.engine={'model':model,'command':server['command'],'sampler':server['sampler'],'vllm':importlib.metadata.version('vllm'),'torch':importlib.metadata.version('torch')}
        self.fingerprint=digest(self.engine); self.directory=self.root/'api'/self.fingerprint; self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'engine.json').write_text(json.dumps(self.engine,indent=2))
        self.tokenizer=AutoTokenizer.from_pretrained(model['path']); self.client=httpx.AsyncClient(base_url='http://127.0.0.1:8017',timeout=600)
        self.semaphore=asyncio.Semaphore(concurrency); self.locks={}

    async def close(self): await self.client.aclose()

    async def call(self,prompt,output_schema,seed,temperature=0.,max_tokens=2048):
        request={'model':'qwen35-9b','messages':[{'role':'user','content':prompt}],'temperature':temperature,'top_p':.9 if temperature else 1.,
                 'max_tokens':max_tokens,'seed':seed,'chat_template_kwargs':{'enable_thinking':False}}
        if output_schema is not None: request['response_format']={'type':'json_schema','json_schema':{'name':'result','strict':True,'schema':output_schema}}
        key=digest(request); path=self.directory/f'{key}.json'
        rendered=self.tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompt_tokens=len(self.tokenizer.encode(rendered,add_special_tokens=False))
        if prompt_tokens+max_tokens>16384: return {'key':key,'status':'context_overflow','prompt_tokens':prompt_tokens,'value':None}
        if key not in self.locks: self.locks[key]=asyncio.Lock()
        async with self.locks[key]:
            if path.exists(): record=json.loads(path.read_text())
            else:
                async with self.semaphore:
                    for attempt in range(5):
                        started=time.time()
                        try: response=await self.client.post('/v1/chat/completions',json=request)
                        except (httpx.TimeoutException,httpx.TransportError) as error:
                            failure={'request':request,'attempt':attempt,'error':repr(error),'time':started}
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps(failure,indent=2))
                            if attempt==4: raise
                            await asyncio.sleep(2**attempt); continue
                        if response.status_code in {429,500,502,503,504}:
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'request':request,'attempt':attempt,'status':response.status_code,'body':response.text,'time':started},indent=2))
                            if attempt==4: response.raise_for_status()
                            await asyncio.sleep(2**attempt); continue
                        if response.is_error:
                            (self.directory/f'{key}.error.json').write_text(json.dumps({'request':request,'status':response.status_code,'body':response.text},indent=2))
                        response.raise_for_status()
                        record={'request':request,'response':response.json(),'started_unix':started,'elapsed_seconds':time.time()-started}
                        temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(record,indent=2)); temporary.replace(path); break
        choice=record['response']['choices'][0]; content=choice['message']['content']
        if output_schema is None:
            return {'key':key,'status':'ok' if choice['finish_reason']=='stop' else 'truncated','value':{'answer':content.strip()},'finish_reason':choice['finish_reason'],'usage':record['response']['usage']}
        try: value=json.loads(content)
        except (json.JSONDecodeError,TypeError): return {'key':key,'status':'invalid_output','value':None,'finish_reason':choice['finish_reason'],'usage':record['response']['usage']}
        status='ok' if choice['finish_reason']=='stop' else 'truncated'
        return {'key':key,'status':status,'value':value,'finish_reason':choice['finish_reason'],'usage':record['response']['usage']}
