"""Scientific controls that must hold before interpreting a source difference."""

import asyncio
import json
from pathlib import Path
import unittest

from run_task_search import run_world
from task_search_data import final_prompt,fixed_prompt,free_prompt,make_world,specification,dataset
from task_search_scoring import arithmetic_score,safe_symbolic,score_answer
from task_search_prompts import answer_prompt
from audit_task_search_constraints import answer_valid


class Controls(unittest.TestCase):
    def test_incorrect_probability_prefix_cannot_receive_full_credit(self):
        entry={'metadata':{'rational':{'numerator':1,'denominator':2}}}
        self.assertFalse(answer_valid('coin_flip','0',entry))
        self.assertFalse(answer_valid('coin_flip','0.49',entry))
        self.assertTrue(answer_valid('coin_flip','1/2',entry))
        self.assertTrue(answer_valid('coin_flip','0.50001',entry))

    def test_palindrome_answer_requires_partitions_not_a_flat_list(self):
        entry={'metadata':{'string':'aab'}}
        self.assertTrue(answer_valid('palindrome_partitioning','[["a","a","b"],["aa","b"]]',entry))
        self.assertFalse(answer_valid('palindrome_partitioning','["aab"]',entry))
        self.assertFalse(answer_valid('palindrome_partitioning','[["aa","b"]]',entry))

    def test_arithmetic_uses_question_numbers_and_exact_target(self):
        self.assertEqual(arithmetic_score('6+6+6+6',[7,6,4,7],24),0)
        self.assertEqual(arithmetic_score('7*6*4/7',[7,6,4,7],24),1)
        self.assertEqual(arithmetic_score('8/(3-8/3)',[8,3,8,3],24),1)
        self.assertEqual(arithmetic_score('49/2',[49,2],24),0)
        self.assertEqual(arithmetic_score('2**4+8',[2,4,8],24),0)

    def test_symbolic_code_is_rejected(self):
        self.assertFalse(safe_symbolic('__import__("os").system("false")'))
        self.assertFalse(safe_symbolic('x.__class__'))
        self.assertTrue(safe_symbolic('sin(x) + x**3/3 + C'))

    def test_malformed_factorization_and_grid_are_zero(self):
        for task,answer in [('prime_factorization','Let me check. The answer is 3.'),('game_of_life','{"answer":"wrong"}')]:
            world=make_world(specification(task,'default'),799,practice_count=1,test_count=1)
            self.assertEqual(score_answer(dataset(world),task,answer,world['tests'][0]),0)

    def test_alternative_coloring_is_valid(self):
        world=make_world(specification('graph_color','default'),719,practice_count=1,test_count=1)
        entry=world['tests'][0]; coloring=json.loads(entry['answer']); colors=sorted(set(coloring.values()))
        alternate={node:colors[(colors.index(color)+1)%len(colors)] for node,color in coloring.items()}
        self.assertNotEqual(coloring,alternate)
        self.assertEqual(score_answer(dataset(world),'graph_color',json.dumps(alternate),entry),1)

    def test_no_test_access_and_wrong_generation_not_filtered(self):
        world=make_world(specification('countdown','default'),101,practice_count=2,test_count=2)
        for i,entry in enumerate(world['tests']): entry['question']='HELDOUT_SENTINEL_'+str(i)
        class FakeModel:
            async def call(self,prompt,output_schema,seed,**kwargs):
                if output_schema is None:
                    if 'temperature' in kwargs and kwargs['temperature']>0: assert 'HELDOUT_SENTINEL' not in prompt
                    value={'answer':'wrong'}
                else:
                    assert 'HELDOUT_SENTINEL' not in prompt
                    value={'question':'duplicate','answer':'wrong'}
                return {'status':'ok','key':'fake','value':value}
        result=asyncio.run(run_world(FakeModel(),world,{'arms':['teacher','self_fixed','self_free'],'sample_counts':[0,1,2]}))
        self.assertEqual(len(result['conditions']),7)
        for arm in ['self_fixed','self_free']:
            self.assertEqual([x['example']['answer'] for x in result['generations'][arm]],['wrong','wrong'])
        fixed=[x['example'] for x in result['generations']['self_fixed']]
        self.assertEqual([x['question'] for x in fixed],[x['question'] for x in world['practice']])
        teacher_prompt=answer_prompt(world,world['practice'],world['tests'][0]); own_prompt=answer_prompt(world,fixed,world['tests'][0])
        self.assertEqual(teacher_prompt.split('Solve this new problem.')[1],own_prompt.split('Solve this new problem.')[1])


if __name__=='__main__': unittest.main()
