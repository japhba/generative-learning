"""Client for the pinned vLLM servers on gpu_lowp (one model per job), with complete cached request/response records."""
import asyncio
import json
from pathlib import Path
import time

import httpx

from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'mlayout'
SERVERS=Path('/ceph/scratch/jbauer/mlayout')


class Served:
    def __init__(self,name,concurrency=64):
        server=json.loads((SERVERS/f'{name}.server.json').read_text());self.name=name
        self.engine={k:server[k] for k in ['name','repo','revision','command']}
        self.fingerprint=digest(self.engine);self.directory=DIRECTORY/'api'/self.fingerprint;self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'engine.json').write_text(json.dumps(self.engine,indent=2))
        key=(Path.home()/'.config/mlayout_vllm_key').read_text().strip()
        self.client=httpx.AsyncClient(base_url=f"http://{server['host']}:{server['port']}",timeout=3600,headers={'Authorization':'Bearer '+key})
        self.semaphore=asyncio.Semaphore(concurrency);self.locks={}

    async def close(self): await self.client.aclose()

    async def call(self,messages,seed,temperature,max_tokens,n=1,prefill=False,thinking=False):
        request={'model':self.name,'messages':messages,'temperature':temperature,'top_p':.9 if temperature else 1.,'max_tokens':max_tokens,'seed':seed,'n':n,
                 'chat_template_kwargs':{'enable_thinking':thinking}}
        if prefill: request|={'continue_final_message':True,'add_generation_prompt':False}
        key=digest(request);path=self.directory/f'{key}.json'
        if key not in self.locks: self.locks[key]=asyncio.Lock()
        async with self.locks[key]:
            if path.exists(): record=json.loads(path.read_text())
            else:
                async with self.semaphore:
                    for attempt in range(8):
                        started=time.time()
                        try: response=await self.client.post('/v1/chat/completions',json=request)
                        except (httpx.TimeoutException,httpx.TransportError) as error:
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'request':request,'error':repr(error),'time':started},indent=2))
                            if attempt==7: raise
                            await asyncio.sleep(min(2**attempt,60));continue
                        if response.status_code in {429,500,502,503,504}:
                            (self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'request':request,'status':response.status_code,'body':response.text,'time':started},indent=2))
                            if attempt==7: response.raise_for_status()
                            await asyncio.sleep(min(2**attempt,60));continue
                        if response.is_error: (self.directory/f'{key}.error.json').write_text(json.dumps({'request':request,'status':response.status_code,'body':response.text},indent=2))
                        response.raise_for_status()
                        record={'request':request,'response':response.json(),'started_unix':started,'elapsed_seconds':time.time()-started}
                        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path);break
        choices=sorted(record['response']['choices'],key=lambda c:c['index'])
        reasoning=lambda m:m['reasoning'] if 'reasoning' in m else m['reasoning_content']  # vLLM field name differs across versions
        return [{'key':key,'status':'ok' if c['finish_reason']=='stop' else 'truncated','text':(c['message']['content'] or '').strip() if thinking else c['message']['content'].strip(),
                 **({'reasoning':reasoning(c['message']) or ''} if thinking else {})} for c in choices]
