"""Persistent, budgeted OpenRouter requests for the Self-ICL replication."""

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

MODELS = {
    "gpt35": {"id": "openai/gpt-3.5-turbo-instruct", "provider": "openai", "prompt_price": 1.5, "completion_price": 2.0, "api": "completions"},
    "qwen": {"id": "qwen/qwen3-30b-a3b-instruct-2507", "provider": "streamlake", "prompt_price": 0.06, "completion_price": 0.22, "api": "chat/completions"},
    "mistral": {"id": "mistralai/mistral-small-3.2-24b-instruct", "provider": "parasail/bf16", "prompt_price": 0.10, "completion_price": 0.32, "api": "chat/completions"},
    "gemini": {"id": "google/gemini-2.5-flash-lite", "provider": "google-ai-studio", "prompt_price": 0.10, "completion_price": 0.40, "api": "chat/completions"},
}
CHAT_INSTRUCTIONS = {
    'inputs': 'Generate exactly ALL of the requested new question instances. Label each one "New instance n:" followed by "Q:". Give questions only, without answers or explanations. Do not stop after the first instance.',
    'direct': 'Answer the final question with only its answer label: a letter for lettered options, otherwise the appropriate True/False, Yes/No, or valid/invalid label. Do not give reasoning, explanations, or repeat the question.',
    'cot': 'Answer the final question by reasoning step by step. End with the sentence "Therefore, the correct answer is LABEL." Replace LABEL with the actual answer. Use an option letter only when the question has explicitly lettered options; otherwise use the requested answer word. Do not repeat the supplied examples.',
}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class Router:
    def __init__(self, directory, key, concurrency=24):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lockfile = (self.directory/'writer.lock').open('a')
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.ledger_path = self.directory/'ledger.jsonl'
        self.ledger_path.touch(exist_ok=True)
        self.reserved, self.settled = {}, {}
        for line in self.ledger_path.read_text().splitlines():
            event = json.loads(line)
            destination = self.reserved if event['event'] == 'reserve' else self.settled
            destination[event['id']] = Decimal(event['usd'])
        self.client = httpx.AsyncClient(headers={'Authorization': 'Bearer '+key, 'X-OpenRouter-Title': 'Self-ICL replication', 'X-OpenRouter-Cache': 'false'}, timeout=180)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.call_locks = defaultdict(asyncio.Lock)
        self.progress = tqdm(desc='OpenRouter', unit='call', mininterval=5)

    def spending(self):
        paid = sum(self.settled.values(), Decimal('0'))
        uncertain = sum((cost for key, cost in self.reserved.items() if key not in self.settled), Decimal('0'))
        return {'paid_usd': str(paid), 'unresolved_reserved_usd': str(uncertain), 'committed_usd': str(paid+uncertain), 'cap_usd': '5.00', 'paid_calls': len(self.settled), 'attempts': len(self.reserved)}

    def event(self, kind, key, cost):
        target = self.reserved if kind == 'reserve' else self.settled
        assert key not in target
        if kind == 'reserve':
            assert Decimal(self.spending()['committed_usd'])+cost <= Decimal('5.00'), 'Self-ICL $5 cap reached before dispatch.'
        else: assert Decimal('0') <= cost <= self.reserved[key], 'Charge exceeded reserved upper bound.'
        with self.ledger_path.open('a') as stream:
            stream.write(json.dumps({'event': kind, 'id': key, 'usd': str(cost)})+'\n')
            stream.flush()
        target[key] = cost

    async def complete(self, model, prompt, max_tokens, stage):
        config = MODELS[model]
        payload = {'model': config['id'], 'temperature': 0, 'top_p': 1, 'max_tokens': max_tokens,
                   'provider': {'only': [config['provider']], 'allow_fallbacks': False, 'require_parameters': True,
                                'max_price': {'prompt': config['prompt_price'], 'completion': config['completion_price'], 'request': 0}}}
        if config['api'] == 'completions': payload['prompt'] = prompt
        else: payload['messages'] = [
            {'role': 'system', 'content': CHAT_INSTRUCTIONS[stage]},
            {'role': 'user', 'content': prompt}]
        if model == 'gemini': payload['reasoning'] = {'enabled': False}
        serialized = json.dumps(payload, sort_keys=True)
        call_id = hashlib.sha256((config['api']+serialized).encode()).hexdigest()
        path = self.directory/(call_id+'.json')
        async with self.call_locks[call_id]:
            if path.exists():
                record = json.loads(path.read_text())
                if record['attempt_id'] not in self.settled:
                    self.event('settle', record['attempt_id'], Decimal(str(record['response']['usage']['cost'])))
                return record
            # UTF-8 bytes bound text tokens; reserve additional template overhead and 2x margin.
            input_bound = len(serialized.encode())+2048
            reserve = Decimal('2')*(Decimal(input_bound)*Decimal(str(config['prompt_price']))+Decimal(max_tokens)*Decimal(str(config['completion_price'])))/1_000_000
            previous_attempts = [int(key.rsplit('_',1)[1]) for key in self.reserved if key.startswith(call_id+'_')]
            start_attempt = max(previous_attempts)+1 if previous_attempts else 0
            for retry in range(3):
                attempt = start_attempt+retry
                attempt_id = f'{call_id}_{attempt}'
                async with self.semaphore:
                    self.event('reserve', attempt_id, reserve)
                    started = time.monotonic()
                    try:
                        response = await self.client.post('https://openrouter.ai/api/v1/'+config['api'], json=payload)
                    except httpx.TransportError as error:
                        save_json(self.directory/(attempt_id+'_error.json'), {'attempt_id': attempt_id, 'request': payload, 'transport_error': repr(error)})
                        if retry == 2: raise
                    else:
                        body = response.json()
                        record = {'call_id': call_id, 'attempt_id': attempt_id, 'request': payload, 'response': body,
                                  'status_code': response.status_code, 'elapsed_seconds': time.monotonic()-started,
                                  'input_token_bound': input_bound, 'reserved_usd': str(reserve)}
                        if response.status_code in (408, 429, 500, 502, 503, 504):
                            save_json(self.directory/(attempt_id+'_error.json'), record)
                            if 'usage' in body: self.event('settle', attempt_id, Decimal(str(body['usage']['cost'])))
                            if retry == 2: response.raise_for_status()
                        else:
                            if response.status_code != 200 or 'error' in body:
                                save_json(self.directory/(attempt_id+'_error.json'), record)
                                raise RuntimeError(f'OpenRouter error saved to {attempt_id}_error.json: {body}')
                            usage = body['usage']
                            assert usage['prompt_tokens'] <= input_bound
                            assert usage['completion_tokens'] <= max_tokens
                            record['text'] = body['choices'][0]['text'] if config['api'] == 'completions' else body['choices'][0]['message']['content']
                            record['finish_reason'] = body['choices'][0]['finish_reason']
                            save_json(path, record)
                            self.event('settle', attempt_id, Decimal(str(usage['cost'])))
                            self.progress.update()
                            self.progress.set_postfix(usd=self.spending()['paid_usd'], refresh=False)
                            return record
                await asyncio.sleep(5*2**retry)

    async def close(self):
        save_json(self.directory/'spending.json', self.spending())
        self.progress.close()
        await self.client.aclose()
        self.lockfile.close()


async def gather(*coroutines):
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures: raise BaseExceptionGroup('Requests finished and cached; inspect failures before resuming.', failures)
    return results
