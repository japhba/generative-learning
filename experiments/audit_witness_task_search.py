"""Recreate every world, audit complete requests, and independently rescore all predictions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tqdm.auto import tqdm

from task_search_api import ROOT
from task_search_data import make_world
from witness_task_search import register,valid


def main():
    root=ROOT.resolve();complete=json.loads((root/'witness_complete.json').read_text());register();phases=['witness_calibration','witness_screen']+(['witness_confirmation'] if complete['selected_tasks'] else []);zero=json.loads((root/'witness_zero_complete.json').read_text());phases+=['witness_zero_screen']+(['witness_zero_confirmation'] if zero['selected_tasks'] else [])
    subprocess.run([sys.executable,str(Path(__file__).with_name('verify_task_search.py')),'--phases',*phases],check=True)
    reports=[];all_keys=set()
    for phase in phases:
        directory=root/phase;design=json.loads((directory/'data_manifest.json').read_text());manifest=json.loads((directory/'inference_manifest.json').read_text());cache=root/'api'/manifest['engine_fingerprint'];records=json.loads((directory/'preflight.json').read_text());stats=Counter();quality=Counter();keys=set()
        for name,expected in design['source_sha256'].items():
            assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==expected
            assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==expected
        for record in tqdm(records,desc='Recreate and rescore '+phase,mininterval=5):
            path=directory/'worlds'/record['world'];assert hashlib.sha256(path.read_bytes()).hexdigest()==record['world_sha256'];world=json.loads(path.read_text())
            assert make_world(world['spec'],world['seed'],design['practice_count'],design['test_count'],design['support_count'])==world
            if record['status']!='ready': stats['preflight_unavailable']+=1;continue
            result=json.loads((directory/'evaluations'/record['world']).read_text());stats['episodes']+=1
            for arm,generations in result['generations'].items():
                assert arm=='self_fixed'
                for generated,entry in zip(generations,world['practice']):
                    call=generated['call'];stats['generation_'+call['status']]+=1;quality['samples']+=1;quality['correct']+=int(valid(generated['example']['answer'],entry));quality['exact_teacher_answer']+=int(generated['example']['answer']==entry['answer'])
                    if call['status']!='context_overflow':
                        raw=json.loads((cache/(call['key']+'.json')).read_text());choice=raw['response']['choices'][0];assert call['status']==('ok' if choice['finish_reason']=='stop' else 'truncated');assert call['value']['answer']==choice['message']['content'].strip();keys.add(call['key'])
            for condition in result['conditions']:
                for call in condition['calls']:
                    stats['evaluation_'+call['status']]+=1
                    if call['status']!='context_overflow':
                        raw=json.loads((cache/(call['key']+'.json')).read_text());choice=raw['response']['choices'][0];assert call['status']==('ok' if choice['finish_reason']=='stop' else 'truncated');assert call['value']['answer']==choice['message']['content'].strip();keys.add(call['key'])
                if condition['status']!='ok': stats['censored_cells']+=1;continue
                correct=[]
                for prediction,query in zip(condition['predictions'],world['tests']):
                    value=valid(prediction['answer'],query);assert value==prediction['correct'] and float(value)==prediction['reward'];correct.append(value)
                assert sum(correct)/len(correct)==condition['accuracy']
        report={'phase':phase,'counts':dict(stats),'generation_quality':dict(quality),'unique_requests':len(keys),'worlds_recreated_and_scores_verified':True,'source_only_request_audit_passed':True};(directory/'witness_verification.json').write_text(json.dumps(report,indent=2));reports.append(report);all_keys.update(keys)
    (root/'witness_verification.json').write_text(json.dumps(reports,indent=2));(root/'witness_request_keys.json').write_text(json.dumps(sorted(all_keys),indent=2));print(json.dumps(reports,indent=2))


if __name__=='__main__': main()
