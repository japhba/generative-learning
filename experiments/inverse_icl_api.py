"""Cached OpenRouter chat requests with one persistent $3 budget for inverse ICL."""

import asyncio
from collections import defaultdict
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
import time

import httpx
from tqdm.auto import tqdm

from self_icl_api import gather, save_json

MODELS={
    'qwen_flash':{'id':'qwen/qwen3.5-flash-02-23','provider':'alibaba','input':.065,'output':.26},
    'qwen36':{'id':'qwen/qwen3.6-35b-a3b','provider':'darkbloom/fp4','input':.05,'output':.7},
    'gemini31':{'id':'google/gemini-3.1-flash-lite','provider':'google-ai-studio','input':.25,'output':1.5},
}


class ChatRouter:
    def __init__(self,directory,key,concurrency=8):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=(self.directory/'writer.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.ledger=self.directory/'ledger.jsonl'; self.ledger.touch(exist_ok=True)
        self.reserved,self.settled={},{}
        for line in self.ledger.read_text().splitlines():
            event=json.loads(line)
            (self.reserved if event['kind']=='reserve' else self.settled)[event['id']]=Decimal(event['usd'])
        self.client=httpx.AsyncClient(headers={'Authorization':'Bearer '+key,'X-OpenRouter-Title':'Compositional inverse ICL'},timeout=120)
        self.semaphore=asyncio.Semaphore(concurrency)
        self.call_locks=defaultdict(asyncio.Lock)
        self.progress=tqdm(desc='OpenRouter',unit='call',mininterval=5)

    def spending(self):
        paid=sum(self.settled.values(),Decimal(0))
        uncertain=sum((value for key,value in self.reserved.items() if key not in self.settled),Decimal(0))
        return {'paid_usd':str(paid),'unresolved_reserved_usd':str(uncertain),'committed_usd':str(paid+uncertain),'cap_usd':'3.00','paid_calls':len(self.settled),'attempts':len(self.reserved)}

    def event(self,kind,key,value):
        ledger=self.reserved if kind=='reserve' else self.settled
        assert key not in ledger
        if kind=='reserve': assert Decimal(self.spending()['committed_usd'])+value<=Decimal('3.00'),'Inverse ICL $3 cap reached before request.'
        else: assert Decimal(0)<=value<=self.reserved[key]
        with self.ledger.open('a') as stream: stream.write(json.dumps({'kind':kind,'id':key,'usd':str(value)})+'\n'); stream.flush()
        ledger[key]=value

    async def complete(self,model,messages,max_tokens):
        config=MODELS[model]
        payload={'model':config['id'],'messages':messages,'temperature':0,'top_p':1,'max_tokens':max_tokens,
                 'reasoning':{'enabled':False},'provider':{'only':[config['provider']],'allow_fallbacks':False,'require_parameters':True,
                 'max_price':{'prompt':config['input'],'completion':config['output'],'request':0}}}
        serialized=json.dumps(payload,sort_keys=True)
        call_id=hashlib.sha256(serialized.encode()).hexdigest()
        path=self.directory/(call_id+'.json')
        async with self.call_locks[call_id]:
            if path.exists():
                result=json.loads(path.read_text())
                if result['attempt_id'] not in self.settled: self.event('settle',result['attempt_id'],Decimal(str(result['response']['usage']['cost'])))
                return result
            bound=len(serialized.encode())+2048
            reserve=Decimal(2)*(Decimal(bound)*Decimal(str(config['input']))+Decimal(max_tokens)*Decimal(str(config['output'])))/1_000_000
            previous=[int(key.rsplit('_',1)[1]) for key in self.reserved if key.startswith(call_id+'_')]
            start=max(previous)+1 if previous else 0
            for retry in range(3):
                attempt=f'{call_id}_{start+retry}'
                async with self.semaphore:
                    self.event('reserve',attempt,reserve)
                    started=time.monotonic()
                    try: response=await self.client.post('https://openrouter.ai/api/v1/chat/completions',json=payload)
                    except httpx.TransportError as error:
                        save_json(self.directory/(attempt+'_error.json'),{'request':payload,'transport_error':repr(error)})
                        if retry==2: raise
                    else:
                        body=response.json()
                        result={'call_id':call_id,'attempt_id':attempt,'request':payload,'response':body,'status':response.status_code,'seconds':time.monotonic()-started}
                        if response.status_code in (408,429,500,502,503,504):
                            save_json(self.directory/(attempt+'_error.json'),result)
                            if retry==2: response.raise_for_status()
                        else:
                            if response.status_code!=200 or 'error' in body:
                                save_json(self.directory/(attempt+'_error.json'),result)
                                raise RuntimeError(f'OpenRouter failure saved: {attempt}')
                            result['text']=body['choices'][0]['message']['content']
                            result['finish_reason']=body['choices'][0]['finish_reason']
                            result['reported_output_tokens_over_allowance']=max(0,body['usage']['completion_tokens']-max_tokens)
                            save_json(path,result)
                            assert body['usage']['prompt_tokens']<=bound
                            self.event('settle',attempt,Decimal(str(body['usage']['cost'])))
                            self.progress.update(); self.progress.set_postfix(usd=self.spending()['paid_usd'],refresh=False)
                            return result
                await asyncio.sleep(3*2**retry)

    async def close(self):
        save_json(self.directory/'spending.json',self.spending())
        self.progress.close(); await self.client.aclose(); self.lock.close()


def work(record):
    usage=record['response']['usage']
    return {'call_id':record['call_id'],'prompt_tokens':usage['prompt_tokens'],'completion_tokens':usage['completion_tokens'],'cost_usd':usage['cost'],'finish_reason':record['finish_reason'],'seconds':record['seconds']}
