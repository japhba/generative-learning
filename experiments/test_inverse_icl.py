"""Scientific controls for compositional inverse ICL; no network calls."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from inverse_icl_task import audit_generated, make_world, predictions, rehearsal_prompt
from run_inverse_icl import run_world


class RecordingAPI:
    def __init__(self): self.calls=[]
    async def complete(self,model,messages,max_tokens):
        self.calls.append((model,messages,max_tokens))
        text='A = 999999\nTHIS IS AN UNFILTERED INCORRECT EXAMPLE' if 'Generate forward examples' in messages[-1]['content'] else 'ANSWER 1: A'
        return {'call_id':hashlib.sha256(str(messages).encode()).hexdigest(),'text':text,'finish_reason':'stop','seconds':0,
                'response':{'usage':{'prompt_tokens':1,'completion_tokens':1,'cost':0}}}


class InverseTests(unittest.IsolatedAsyncioTestCase):
    def test_forward_composition_and_unique_inverse(self):
        for dimension in (8,10,12):
            world=make_world(4000+dimension,dimension)
            self.assertTrue(all(predictions(world,'\n'.join(f"ANSWER {item['id']}: {item['answer']}" for item in world['tests']))[i]['correct'] for i in range(len(world['tests']))))
            totals={row['total'] for row in world['tests']}
            self.assertFalse(totals & {row['total'] for row in world['full']})
            for item in world['tests']:
                mask=item['mask']; half=dimension//2
                left=sum(bit*weight for bit,weight in zip(mask[:half],world['weights'][:half],strict=True))
                right=sum(bit*weight for bit,weight in zip(mask[half:],world['weights'][half:],strict=True))
                self.assertEqual(left+right,item['total'])

    async def test_bad_generated_examples_are_retained_and_gold_answers_do_not_change_calls(self):
        world=make_world(4100,8,queries=1)
        changed=copy.deepcopy(world); changed['tests'][0]['answer']='WRONG'; changed['tests'][0]['mask']=[0]*8
        records=[]
        with tempfile.TemporaryDirectory() as temporary:
            for i,value in enumerate((world,changed)):
                path=Path(temporary)/str(i); (path/'cases').mkdir(parents=True)
                api=RecordingAPI()
                result=await run_world(api,path,value,'qwen_flash',[0,2,4],['generated_parts'],64,False,'reasoning')
                self.assertTrue(all('999999' in row['examples'] for row in result['rows'] if row['condition']=='generated_parts'))
                records.append(api.calls)
            self.assertEqual(records[0],records[1])
        alternate=copy.deepcopy(world); alternate['tests'][0]['total']+=1
        subsets=[row['subset'] for row in world['structured'][:4]]
        self.assertEqual(rehearsal_prompt(world,subsets),rehearsal_prompt(alternate,subsets))

    def test_diagnostics_score_errors_without_repair(self):
        world=make_world(4200,8)
        record=audit_generated(world,['A'],'A = 999999')
        self.assertEqual(record['correct_count'],0)
        self.assertEqual(record['records'][0]['reported_total'],999999)


if __name__=='__main__': unittest.main()
