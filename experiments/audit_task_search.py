"""After-inference diagnostics; generated-sample scores never feed the learner."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from task_search_api import ROOT
from task_search_data import dataset
from task_search_scoring import score_answer


def normalize(text): return ' '.join(text.split())


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',required=True); args=parser.parse_args(); root=ROOT.resolve(); directory=root/args.phase
    pre=json.loads((directory/'preflight.json').read_text()); files=sorted((directory/'evaluations').glob('*.json'))
    assert len(files)==sum(r['status']=='ready' for r in pre),'Audit only after the frozen phase has finished.'
    tokenizer=AutoTokenizer.from_pretrained(json.loads((root/'runtime/model.json').read_text())['path']); records=[]
    for path in tqdm(files,desc='Post-inference sample audit',mininterval=10):
        result=json.loads(path.read_text()); world=json.loads((directory/'worlds'/path.name).read_text()); data=dataset(world)
        support={normalize(x['question']) for x in world['support']}; tests={normalize(x['question']) for x in world['tests']}
        generations=[]
        for arm,samples in result['generations'].items():
            seen=set()
            for item in samples:
                example=item['example']; question=normalize(example['question']); record={'arm':arm,'index':item['index'],'output_status':item['call']['status'],
                    'question_tokens':len(tokenizer.encode(example['question'])),'answer_tokens':len(tokenizer.encode(example['answer'])),
                    'duplicates_support':question in support,'duplicates_previous_generation':question in seen,'matches_test_question':question in tests}
                if arm=='self_fixed':
                    entry=world['practice'][item['index']]; assert entry['question']==example['question']; reward=score_answer(data,world['spec']['task'],example['answer'],entry)
                    record.update({'reward':reward,'correct':reward>=1-1e-9,'exact_teacher_answer':example['answer']==entry['answer'],'teacher_answer_tokens':len(tokenizer.encode(entry['answer']))})
                generations.append(record); seen.add(question)
        conditions=[]
        for condition in result['conditions']:
            calls=[call for call in condition['calls'] if 'usage' in call]
            conditions.append({'arm':condition['arm'],'n':condition['n'],'status':condition['status'],'output_statuses':dict(Counter(call['status'] for call in condition['calls'])),
                               'prompt_tokens':sum(call['usage']['prompt_tokens'] for call in calls),'completion_tokens':sum(call['usage']['completion_tokens'] for call in calls)})
        records.append({'spec':world['spec']['id'],'seed':world['seed'],'generations':generations,'conditions':conditions})
    (directory/'diagnostics.json').write_text(json.dumps({'timing':'After every evaluation completed; these diagnostics did not select, repair, or reweight generated samples.','episodes':records},indent=2))
    all_samples=[s for r in records for s in r['generations']]; fixed=[s for s in all_samples if s['arm']=='self_fixed']
    print(json.dumps({'episodes':len(records),'generated_samples':len(all_samples),'own_fixed_answer_accuracy':float(np.mean([s['correct'] for s in fixed])) if fixed else None,
                      'generation_statuses':dict(Counter(s['output_status'] for s in all_samples)),'generated_questions_matching_test':sum(s['matches_test_question'] for s in all_samples),'duplicate_generated_questions':sum(s['duplicates_previous_generation'] for s in all_samples)},indent=2))


if __name__=='__main__': main()
