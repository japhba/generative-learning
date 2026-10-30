"""Self-ICL: query-conditioned inputs, independent pseudo-labels, then frozen-model ICL."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from zoneinfo import ZoneInfo
from datetime import datetime

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from self_icl_api import MODELS, Router, gather, save_json

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'dyck_languages', 'multistep_arithmetic_two', 'object_counting', 'word_sorting'}
COT = " Let's think step by step."
FORMAT = 'Format:\nStarting with "Therefore, the correct answer is ..." before giving your final answer.\nIf options are availbale, you must pick one as the final answer.\nIt\'s very important that you stick to the format.\n\n'


def prediction_prompt(description, query, shots=(), cot=False):
    prompt = f'Task description: {description}\n\n'+(FORMAT if cot else '')
    for question, answer in shots: prompt += f'Q: {question}\nA:{COT if cot else ""} {answer}\n\n'
    return prompt+f'Q: {query}\nA:'+(COT if cot else '')


def generation_prompt(description, query, k):
    return f'Following is an example instance for the task: {description} Please come up with {k} new, diverse, and creative instances for the task.\n\nExample instance:\nQ: {query}\n\nNew instance 1:\nQ:'


def split_inputs(text, k):
    # Only presentation syntax is normalized; no quality or correctness selection.
    clean = re.sub(r'\*\*|__', '', text).strip()
    heading = r'(?:^|\n)\s*(?:#{1,4}\s*)?New instance\s+(\d+)\s*:\s*Q\s*:\s*'
    matches = list(re.finditer(heading, clean, re.I))
    if not matches or matches[0].group(1) != '1':
        clean = 'New instance 1:\nQ: '+clean
        matches = list(re.finditer(heading, clean, re.I))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, k+1)): return {'valid': False, 'count': len(matches), 'inputs': [], 'numbers': numbers}
    inputs = [clean[match.end():matches[i+1].start() if i+1<len(matches) else len(clean)].strip() for i, match in enumerate(matches)]
    return {'valid': all(inputs), 'count': len(inputs), 'inputs': inputs, 'numbers': numbers}


def normalize(value):
    return value.strip().strip('`*_()[]{}.,:;! ').upper()


def extract(text, cot, choice, continuation_prefix=''):
    raw = continuation_prefix+text
    if cot:
        matches = list(re.finditer(r'(?:therefore,?\s+the correct answer is\s*:?|the correct answer is therefore|so the correct answer is)', raw, re.I))
        if not matches: return None
        tail = raw[matches[-1].end():].strip()
        first = re.split(r'[,.;: \n]', tail)[0]
    else:
        match = re.match(r'\s*\W*(\w+)', raw)
        if match is None: return None
        first = match.group(1)
    return normalize(first)


def prepare_inputs(root):
    source = Path(os.environ['FAST_CACHE_DIR'])/'self_icl_source'
    bbh = Path(os.environ['FAST_CACHE_DIR'])/'self_icl_bbh'
    destination = root/'inputs'
    destination.mkdir(exist_ok=True)
    descriptions = json.loads((source/'bbh/bbh_task_description.json').read_text())
    tasks = {}
    for task in sorted(set(descriptions)-EXCLUDED):
        examples = json.loads((bbh/'bbh'/f'{task}.json').read_text())['examples']
        demos = json.loads((source/'bbh/BIG-Bench-Hard/cot-prompts-parsed'/f'{task}.json').read_text())
        permutation = np.random.default_rng(9102026).permutation(len(examples)).tolist()
        tasks[task] = {'description': descriptions[task], 'examples': examples, 'official_demos': demos, 'permutation': permutation}
    save_json(destination/'tasks.json', tasks)
    for filename in ('prompt.py', 'promptparser.py', 'model.py', 'experiment.py'):
        shutil.copy2(source/filename, destination/('upstream_'+filename))
    provenance = {'self_icl_commit': subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                  'bbh_commit': subprocess.check_output(['git','-C',str(bbh),'rev-parse','HEAD'],text=True).strip(),
                  'tasks_sha256': hashlib.sha256((destination/'tasks.json').read_bytes()).hexdigest()}
    save_json(destination/'provenance.json', provenance)
    return tasks


def work(record):
    usage = record['response']['usage']
    return {'call_id': record['call_id'], 'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': usage['completion_tokens'],
            'cost_usd': usage['cost'], 'finish_reason': record['finish_reason'], 'provider': record['response']['provider']}


async def run_case(api, directory, task_name, task, index, model, budgets, modes, real, input_tokens_per_shot):
    path = directory/'cases'/f'{model}__{task_name}__{index}.json'
    if path.exists(): return json.loads(path.read_text())
    case = task['examples'][index]
    query, target, description = case['input'], normalize(case['target']), task['description']
    choice = case['target'].startswith('(')
    rows, generations = [], []
    result = {'task': task_name, 'index': index, 'model': model, 'query': query, 'target': case['target'], 'rows': rows, 'generations': generations}

    async def predict(question, shots, mode):
        cot = mode == 'cot'
        prompt = prediction_prompt(description, question, shots, cot)
        prefix = '(' if model == 'gpt35' and choice and not cot else ''
        if prefix: prompt += ' ('
        tokens = 1024 if cot else (2 if model == 'gpt35' else 16)
        record = await api.complete(model, prompt, tokens, mode)
        return record, prefix

    async def evaluate(shots, mode, condition, budget, preparation):
        prediction, prefix = await predict(query, shots, mode)
        answer = extract(prediction['text'], mode == 'cot', choice, prefix)
        rows.append({'mode': mode, 'condition': condition, 'budget': budget, 'prediction': answer, 'correct': answer == target,
                     'answer_parse_failure': answer not in {normalize(item['target']) for item in task['examples']}, 'pipeline_failure': False, 'response': prediction['text'],
                     'evaluation': work(prediction), 'preparation': preparation, 'demonstrations': shots})

    await gather(*(evaluate([], mode, 'zero', 0, []) for mode in modes))
    if real:
        official = [(demo['Q'], demo['A']) for demo in task['official_demos']]
        assert query not in [demo[0] for demo in official], 'Test item overlaps official demonstration.'
        await evaluate(official, 'direct', 'official_real', 3, [])

    for k in budgets:
        if k == 0: continue
        if real and len(budgets)>2:
            # The first 32 shuffled BBH items are a disjoint labelled demonstration pool.
            gold_indices = task['permutation'][:k]
            assert index not in gold_indices
            gold = [(task['examples'][i]['input'], task['examples'][i]['target']) for i in gold_indices]
            await evaluate(gold, 'direct', 'real', k, [])
        generated = await api.complete(model, generation_prompt(description, query, k), 1024 if input_tokens_per_shot==0 else k*input_tokens_per_shot, 'inputs')
        parsed = split_inputs(generated['text'], k)
        generations.append({'budget': k, 'generation': work(generated), 'response': generated['text'], 'parse': parsed})
        if not parsed['valid']:
            for mode in modes:
                rows.append({'mode': mode, 'condition': 'self', 'budget': k, 'prediction': None, 'correct': False,
                             'answer_parse_failure': True, 'pipeline_failure': True, 'response': '', 'evaluation': None,
                             'preparation': [work(generated)], 'demonstrations': []})
            continue
        async def mode_run(mode):
            labelled = await gather(*(predict(question, [], mode) for question in parsed['inputs']))
            shots = [(question, (prefix+answer['text']).strip()) for question, (answer, prefix) in zip(parsed['inputs'], labelled, strict=True)]
            preparation = [work(generated)]+[work(answer) for answer, prefix in labelled]
            await evaluate(shots, mode, 'self', k, preparation)
        await gather(*(mode_run(mode) for mode in modes))
    save_json(path, result)
    return result


async def run(args):
    root = args.output.resolve()
    tasks = prepare_inputs(root) if args.phase == 'pilot' else json.loads((root/'inputs/tasks.json').read_text())
    selected = sorted(tasks) if args.tasks == ['all'] else args.tasks
    directory = root/args.phase
    (directory/'cases').mkdir(parents=True, exist_ok=True)
    manifest = {'phase': args.phase, 'models': {key: MODELS[key] for key in args.models}, 'tasks': selected, 'count_per_task': args.count,
                'start': args.start, 'budgets': args.budgets, 'modes': args.modes, 'real_controls': args.real,
                'input_tokens_per_shot': args.input_tokens_per_shot,
                'inputs_sha256': hashlib.sha256((root/'inputs/tasks.json').read_bytes()).hexdigest(),
                'case_indices': {name: tasks[name]['permutation'][args.start:args.start+args.count] for name in selected},
                'source_sha256': {name: hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest() for name in ('self_icl.py','self_icl_api.py')}}
    if (directory/'manifest.json').exists(): assert json.loads((directory/'manifest.json').read_text()) == manifest, 'Frozen protocol changed.'
    else:
        save_json(directory/'manifest.json', manifest)
        (directory/'started.txt').write_text(datetime.now(ZoneInfo('Europe/London')).isoformat()+'\n')
        (directory/'source').mkdir(exist_ok=True)
        for filename in ('self_icl.py', 'self_icl_api.py'): shutil.copy2(ROOT/'experiments'/filename, directory/'source'/filename)
    api = Router(root/'api', os.environ['OPENROUTER_API_KEY'], args.concurrency)
    cases = [(name, index, model) for name in selected for index in manifest['case_indices'][name] for model in args.models]
    progress = tqdm(total=len(cases), desc=args.phase, unit='case', mininterval=5)
    case_semaphore = asyncio.Semaphore(args.case_concurrency)
    async def one(name, index, model):
        async with case_semaphore:
            result = await run_case(api, directory, name, tasks[name], index, model, args.budgets, args.modes, args.real, args.input_tokens_per_shot)
            progress.update()
            return result
    try:
        result = await gather(*(one(*case) for case in cases))
        save_json(directory/'results.json', result)
    finally:
        progress.close()
        await api.close()
        print(json.dumps(api.spending()), flush=True)


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/self_icl')
    parser.add_argument('--phase', required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=['gpt35','qwen','gemini'])
    parser.add_argument('--tasks', nargs='+', default=['all'])
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--start', type=int, default=32)
    parser.add_argument('--budgets', type=int, nargs='+', default=[0,3])
    parser.add_argument('--modes', nargs='+', choices=['direct','cot'], default=['direct','cot'])
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--concurrency', type=int, default=24)
    parser.add_argument('--case-concurrency', type=int, default=12)
    parser.add_argument('--input-tokens-per-shot', type=int, default=0, help='0 uses the paper\'s fixed 1024-token input generation; scaling uses 1024 per requested input.')
    args = parser.parse_args()
    assert args.start>=32 and args.count>0 and max(args.budgets)<=32
    assert max(args.budgets)<=3 or args.input_tokens_per_shot>0
    asyncio.run(run(args))


if __name__ == '__main__': main()
