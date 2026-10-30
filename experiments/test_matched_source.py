"""Semantic and treatment-invariant checks for the source-only search."""

import asyncio
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from matched_source_tasks import FAMILIES,base_text,canonical,extract_answer,generation_prompt,inference_prompt,make_world,score
from run_matched_source import run_world
from evaluate_matched_traces import trace_prompt
from run_matched_bbh import answer_label,eligible_indices,make_world as make_bbh_world,score_bbh
from replay_matched_bbh_labels import generated_label


class FakeAPI:
    def __init__(self,generation='ANSWER: BAD'): self.calls=[]; self.generation=generation
    async def complete(self,model,messages,limit):
        self.calls.append((model,messages,limit))
        text=self.generation if 'Solve this problem.' in messages[-1]['content'] else '1: BAD\n2: BAD'
        return {'call_id':str(len(self.calls)),'text':text,'finish_reason':'stop','seconds':0,
                'response':{'usage':{'prompt_tokens':10,'completion_tokens':10,'cost':0}}}


class MatchedSourceTests(unittest.TestCase):
    def test_all_families_have_disjoint_reproducible_questions(self):
        for family in FAMILIES:
            for level in (1,2):
                world=make_world(family,level,5123,64,24)
                self.assertEqual(world,make_world(family,level,5123,64,24))
                items=world['support']+world['practice']+world['tests']
                self.assertEqual(len({item['question'] for item in items}),92)
                self.assertTrue(all(canonical(item['answer']) for item in items))

    def test_inverse_base_and_parity_have_independent_checks(self):
        for level in (1,2):
            for family in ('affine_inverse','base_multiplication','parity_code'):
                world=make_world(family,level,2134,16,24); state=world['oracle_state']
                for item in world['support']+world['practice']+world['tests']:
                    if family=='affine_inverse':
                        x=int(item['question'].split('=')[1]); a,b,_,_=state['coefficients']; p=state['modulus']
                        self.assertEqual(int(item['answer']),pow(a,-1,p)*(x-b)%p)
                    elif family=='base_multiplication':
                        a,b=item['question'].split(' * '); base=state['base']
                        self.assertEqual(int(item['answer'],base),int(a,base)*int(b,base))
                    else:
                        bits=list(map(int,item['question']))
                        result=''.join(str(sum(a*b for a,b in zip(row,bits,strict=True))%2) for row in state['matrix'])
                        self.assertEqual(item['answer'],result)

    def test_bayesian_teacher_matches_log_odds(self):
        import re
        world=make_world('bayes_urn',2,7654,32,24)
        for item in world['support']+world['practice']+world['tests']:
            values=list(map(int,re.findall(r'\d+',item['question'])))
            prior,_,pa,_,pb,_,red,blue=values
            odds=np.log(prior/(10-prior))+red*np.log(pa/pb)+blue*np.log((10-pa)/(10-pb))
            if abs(odds)>1e-10: self.assertEqual(item['answer'],'A' if odds>0 else 'B')

    def test_numeric_and_symbolic_scoring_rejects_ambiguity(self):
        self.assertEqual(extract_answer('work\n**ANSWER:** AB, CD'),'AB,CD')
        self.assertIsNone(extract_answer('maybe 4'))
        queries=[{'id':1,'question':'x','answer':')]}'},{'id':2,'question':'y','answer':'A,B'}]
        self.assertTrue(all(item['correct'] for item in score(queries,'1: )]}\n2: A, B')))
        self.assertIsNone(score(queries,'1: )]}\n1: BAD')[0]['prediction'])

    def test_oracle_and_test_fields_cannot_enter_generation(self):
        world=make_world('hidden_affine',1,8761,4,2); changed=copy.deepcopy(world)
        changed['oracle_state']['coefficients']=[999]*4
        for item in changed['practice']+changed['tests']: item['answer']='SECRET'
        changed['tests'][0]['question']='UNSEEN TARGET'
        self.assertEqual(generation_prompt(world,world['practice'][0]),generation_prompt(changed,changed['practice'][0]))
        tests=copy.deepcopy(world['tests'])
        for item in tests: item['answer']='SECRET'
        self.assertEqual(inference_prompt(world,['BAD']*4,world['tests']),inference_prompt(world,['BAD']*4,tests))

    def test_paired_runner_keeps_wrong_answers_and_only_changes_answer_slots(self):
        world=make_world('hidden_affine',1,9123,4,2)
        config={'models':{'qwen_flash':{}},'generation_tokens':1024,'inference_tokens':512,'batch_size':2,'budgets':[0,4],'cross_teacher':False}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); (path/'cases').mkdir(); api=FakeAPI()
            result=asyncio.run(run_world(api,path,world,config))
        own=next(row for row in result['rows'] if row['source']=='qwen_flash')
        teacher=next(row for row in result['rows'] if row['source']=='oracle')
        self.assertEqual(own['answers'],['BAD']*4)
        self.assertEqual(teacher['answers'],[item['answer'] for item in world['practice']])
        for row in (own,teacher):
            call=next(call for call in api.calls if call[1][-1]['content']==inference_prompt(world,row['answers'],world['tests']))
            self.assertEqual(call[0],'qwen_flash'); self.assertEqual(call[2],512)
            for item in world['practice']: self.assertEqual(call[1][-1]['content'].count('Input: '+item['question']+'\n'),1)
        self.assertTrue(all(not item['correct'] for item in result['generations']['qwen_flash']))

    def test_unparsed_generation_cannot_smuggle_reasoning_into_answer_only_context(self):
        world=make_world('hidden_affine',1,9123,4,2)
        config={'models':{'qwen_flash':{}},'generation_tokens':1024,'inference_tokens':512,'batch_size':2,'budgets':[0,4],'cross_teacher':False}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); (path/'cases').mkdir(); api=FakeAPI('An unfinished private calculation')
            result=asyncio.run(run_world(api,path,world,config))
        own=next(row for row in result['rows'] if row['source']=='qwen_flash')
        self.assertEqual(own['answers'],['UNPARSED']*4)
        self.assertTrue(all(item['response']=='An unfinished private calculation' for item in result['generations']['qwen_flash']))
        self.assertNotIn('An unfinished private calculation',inference_prompt(world,own['answers'],world['tests']))

    def test_full_solution_source_changes_only_solution_slots(self):
        world=make_world('boolean_circuit',1,7612,4,2)
        own=trace_prompt(world,['LEARNER WORK']*4,world['tests'])
        teacher=trace_prompt(world,['TEACHER WORK']*4,world['tests'])
        self.assertEqual(own.replace('LEARNER WORK','SOLUTION'),teacher.replace('TEACHER WORK','SOLUTION'))
        self.assertEqual(trace_prompt(world,[],world['tests']),inference_prompt(world,[],world['tests']))
        changed=copy.deepcopy(world['tests'])
        for query in changed: query['answer']='SECRET'
        self.assertEqual(own,trace_prompt(world,['LEARNER WORK']*4,changed))

    def test_bbh_scoring_preserves_commas_and_rejects_ambiguous_answers(self):
        question='Pick a movie.\nOptions:\n(A) Monsters, Inc.\n(B) Another movie'
        self.assertEqual(answer_label(question,'Monsters, Inc.'),'A')
        self.assertEqual(answer_label(question,'(A)'),'A')
        self.assertIsNone(answer_label(question,'Monsters'))
        queries=[{'id':1,'question':question,'answer':'A'}]
        self.assertTrue(score_bbh(queries,'1: Monsters, Inc.')[0]['correct'])
        self.assertIsNone(score_bbh(queries,'1: A\n1: B')[0]['prediction'])
        duplicate_task={'examples':[{'input':question,'target':target} for target in ('A','A')]}
        self.assertEqual(eligible_indices(duplicate_task),[0])
        duplicate_task['examples'][1]['target']='B'
        self.assertEqual(eligible_indices(duplicate_task),[])

    def test_bbh_phase_partitions_and_generation_exclude_test_information(self):
        tasks={'toy':{'description':'Choose A.','examples':[{'input':f'Question {i}\n(A) First\n(B) Second','target':'(A)'} for i in range(120)]}}
        worlds=[make_bbh_world(tasks,'toy',8317,pool,16,12) for pool in ('screen','confirmation')]
        ids=[{item['dataset_index'] for role in ('support','practice','tests') for item in world[role]} for world in worlds]
        self.assertFalse(ids[0]&ids[1])
        original=worlds[0]; changed=copy.deepcopy(original)
        for item in changed['practice']+changed['tests']: item['answer']='SECRET'
        changed['tests'][0]['question']='HIDDEN TEST'
        self.assertEqual(generation_prompt(original,original['practice'][0]),generation_prompt(changed,changed['practice'][0]))

    def test_bare_label_replay_never_repairs_a_generated_label(self):
        question='Is the argument valid or invalid?'
        for text in ('INVALID','reasoning\nINVALID','ANSWER: INVALID'):
            self.assertEqual(generated_label(question,text),'INVALID')
        self.assertEqual(generated_label(question,'VALID'),'VALID')
        self.assertIsNone(generated_label(question,'I cannot decide'))


if __name__=='__main__': unittest.main()
