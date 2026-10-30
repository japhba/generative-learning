"""One-problem inference in each benchmark task's native answer format."""

from task_search_data import initial_context


def answer_prompt(world,examples,query):
    additional='\n\nAdditional examples:\n'+'\n\n'.join(f"Problem: {item['question']}\nAnswer: {item['answer']}" for item in examples) if examples else ''
    return initial_context(world)+additional+'\n\nSolve this new problem. Return only the answer in the format requested by the problem, without working or explanations.\nProblem: '+query['question']+'\nAnswer:'
