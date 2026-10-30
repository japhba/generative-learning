"""Scientific isolation and budget tests; no paid requests."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from self_icl import extract, run_case, split_inputs
from self_icl_api import Router
from audit_self_icl import boolean_truth, option_text_answer
from decimal import Decimal


class RecordingRouter:
    def __init__(self, malformed=False): self.requests, self.malformed = [], malformed

    async def complete(self, model, prompt, max_tokens, stage):
        self.requests.append((model,prompt,max_tokens,stage))
        if stage=='inputs': text='one' if self.malformed else 'one\n\nNew instance 2:\nQ: two\n\nNew instance 3:\nQ: three'
        else: text='Therefore, the correct answer is (A).' if stage=='cot' else '(A)'
        return {'call_id':hashlib.sha256(prompt.encode()).hexdigest(), 'text':text, 'finish_reason':'stop',
                'response':{'usage':{'prompt_tokens':1,'completion_tokens':1,'cost':0},'provider':'mock'}}


class SelfICLTests(unittest.IsolatedAsyncioTestCase):
    async def test_heldout_target_never_changes_requests(self):
        requests=[]
        with tempfile.TemporaryDirectory() as temporary:
            for target in ('(A)','(B)'):
                directory=Path(temporary)/target; (directory/'cases').mkdir(parents=True)
                task={'examples':[{'input':'Same held-out question','target':target}], 'description':'Task.', 'official_demos':[]}
                api=RecordingRouter()
                result=await run_case(api,directory,'test',task,0,'qwen',[0,3],['direct','cot'],False,0)
                self.assertEqual(sum(row['correct'] for row in result['rows']),4 if target=='(A)' else 0)
                requests.append(api.requests)
        self.assertEqual(requests[0],requests[1])
        self.assertEqual(len(requests[0]),11)
        self.assertEqual(sum(call[3]=='inputs' for call in requests[0]),1)

    async def test_malformed_generation_is_counted_without_reroll(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary); (directory/'cases').mkdir()
            api=RecordingRouter(malformed=True)
            task={'examples':[{'input':'Question','target':'(A)'}], 'description':'Task.', 'official_demos':[]}
            result=await run_case(api,directory,'test',task,0,'qwen',[0,3],['direct','cot'],False,0)
            self.assertEqual(len(result['rows']),4)
            self.assertEqual(sum(row['pipeline_failure'] for row in result['rows']),2)
            self.assertEqual(len(api.requests),3)

    async def test_unresolved_charges_stay_reserved_on_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            api=Router(temporary,'test-key')
            api.event('reserve','uncertain',Decimal('4.50'))
            await api.close()
            api=Router(temporary,'test-key')
            self.assertEqual(Decimal(api.spending()['committed_usd']),Decimal('4.50'))
            with self.assertRaises(AssertionError): api.event('reserve','would-exceed-cap',Decimal('.51'))
            await api.close()

    def test_long_lists_and_cot_markdown(self):
        response='First question\n'+''.join(f'New instance {i}:\nQ: question {i}\n' for i in range(2,17))
        self.assertTrue(split_inputs(response,16)['valid'])
        self.assertEqual(extract('Therefore, the correct answer is **(B)**.',True,True),'B')

    def test_option_text_audit_does_not_infer_an_answer_from_ambiguous_text(self):
        question='Pick a date.\n(A) 01/01/2000\n(B) 02/02/2000'
        self.assertEqual(option_text_answer(question,'Therefore, the correct answer is 02/02/2000.',None,{'A','B'}),'B')
        self.assertEqual(option_text_answer(question,'Therefore, the correct answer is (A).','A',{'A','B'}),'A')
        self.assertIsNone(option_text_answer(question,'Therefore, the correct answer is a date in 2000.',None,{'A','B'}))
        ambiguous='Pick a date.\n(A) 01/01/2000\n(B) 01/01/2000'
        self.assertIsNone(option_text_answer(ambiguous,'Therefore, the correct answer is 01/01/2000.',None,{'A','B'}))

    def test_boolean_audit_respects_precedence_and_rejects_non_boolean_code(self):
        self.assertEqual(boolean_truth('Q: False and True or not False is'),'TRUE')
        self.assertEqual(boolean_truth('not (True or False) is'),'FALSE')
        with self.assertRaises(ValueError): boolean_truth('bool(1)')


if __name__=='__main__': unittest.main()
