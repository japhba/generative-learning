"""Check inverse-task scoring, information isolation, and unfiltered generation."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from compiled_inverse_icl import generation_prompt,make_world,run_world,score


class RecordingAPI:
    def __init__(self): self.calls=[]
    async def complete(self,model,messages,max_tokens):
        self.calls.append((model,messages,max_tokens))
        text='CALC: deliberately wrong\nCODE: 999' if 'nonnegative remainder' in messages[-1]['content'] else '1: 1'
        return {'text':text,'call_id':hashlib.sha256(str(messages).encode()).hexdigest(),'finish_reason':'stop','seconds':0,
                'response':{'usage':{'prompt_tokens':1,'completion_tokens':1,'cost':0}}}


class CompiledTests(unittest.IsolatedAsyncioTestCase):
    def test_encoder_has_an_exact_inverse(self):
        world=make_world(302000)
        codes=[(world['a']*z+world['b'])%257 for z in range(1,257)]
        self.assertEqual(len(set(codes)),256)
        inverse=pow(world['a'],-1,257)
        for z,code in enumerate(codes,start=1): self.assertEqual((inverse*(code-world['b']))%257,z)

    def test_scoring_accepts_surface_forms_and_rejects_ambiguous_answers(self):
        query=[{'id':1,'code':189,'label':2,'weight':.5}]
        for text in ('ANSWER 1: 2','1: label 2','1: ANSWER 2','1: code 189 -> label 2','**1: 2**'):
            self.assertTrue(score(query,text)[0]['correct'])
        self.assertFalse(score(query,'1: either 2 or 3')[0]['correct'])
        self.assertFalse(score(query,'1: code 189')[0]['correct'])

    async def test_wrong_codes_remain_and_test_answers_never_change_requests(self):
        world=make_world(302001)
        changed=copy.deepcopy(world)
        for row in changed['queries']: row['label']=256-row['label']
        requests=[]
        with tempfile.TemporaryDirectory() as temporary:
            for index,value in enumerate((world,changed)):
                directory=Path(temporary)/str(index); (directory/'cases').mkdir(parents=True)
                api=RecordingAPI()
                result=await run_world(api,directory,value,'qwen_flash',[0,2],8,[],True)
                self.assertEqual(len(result['generation']),2)
                self.assertTrue(all(not item['correct'] for item in result['generation']))
                self.assertIn('Code 999 -> label 1',next(row['examples'] for row in result['rows'] if row['condition']=='generated'))
                requests.append(api.calls)
            self.assertEqual(requests[0],requests[1])
        changed['queries'][0]['code']+=1
        self.assertEqual(generation_prompt(world,5),generation_prompt(changed,5))


if __name__=='__main__': unittest.main()
