"""Constraint scoring, including strict arithmetic resource use and safe expressions."""

import ast
import json
import re
from collections import Counter
from fractions import Fraction


SYMBOLIC_TASKS={'intermediate_integration','polynomial_multiplication','simple_integration'}
FUNCTIONS={'sin','cos','tan','asin','acos','atan','sinh','cosh','tanh','exp','log','sqrt','Abs','abs','Rational'}


def arithmetic_score(text,numbers,target):
    try: tree=ast.parse(text.strip(),mode='eval')
    except (SyntaxError,ValueError): return 0.
    nodes=list(ast.walk(tree))
    allowed=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.UAdd,ast.USub)
    if len(nodes)>256 or any(not isinstance(node,allowed) for node in nodes): return 0.
    literals=[node.value for node in nodes if isinstance(node,ast.Constant)]
    if any(type(value) not in (int,float) for value in literals) or Counter(literals)!=Counter(numbers): return 0.
    def value(node):
        if isinstance(node,ast.Constant): return Fraction(str(node.value))
        if isinstance(node,ast.UnaryOp): return -value(node.operand) if isinstance(node.op,ast.USub) else value(node.operand)
        a,b=value(node.left),value(node.right)
        if isinstance(node.op,ast.Add): return a+b
        if isinstance(node.op,ast.Sub): return a-b
        if isinstance(node.op,ast.Mult): return a*b
        return a/b
    try: return float(value(tree.body)==Fraction(str(target)))
    except (ZeroDivisionError,ValueError,OverflowError): return 0.


def safe_symbolic(text):
    try: tree=ast.parse(text.strip(),mode='eval')
    except (SyntaxError,ValueError): return False
    nodes=list(ast.walk(tree)); allowed=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Constant,ast.Name,ast.Load,ast.Call,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.UAdd,ast.USub)
    if len(nodes)>256 or any(not isinstance(node,allowed) for node in nodes): return False
    for node in nodes:
        if isinstance(node,ast.Name) and not (len(node.id)==1 and node.id.isalpha() or node.id in FUNCTIONS|{'pi'}): return False
        if isinstance(node,ast.Constant) and (type(node.value) not in (int,float) or abs(node.value)>1e6): return False
        if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in FUNCTIONS or node.keywords): return False
    return True


def score_answer(data,task,answer,entry):
    if task=='prime_factorization' and any(re.fullmatch(r'[+-]?\d+',part.strip()) is None for part in answer.split('×') if part.strip()): return 0.
    if task=='game_of_life':
        try: matrix=json.loads(answer)
        except (json.JSONDecodeError,TypeError): return 0.
        if not isinstance(matrix,list) or any(not isinstance(row,list) for row in matrix): return 0.
    if task in {'puzzle24','countdown'}:
        return arithmetic_score(answer,entry['metadata']['numbers'],24 if task=='puzzle24' else entry['metadata']['target'])
    if task in SYMBOLIC_TASKS and not safe_symbolic(answer): return 0.
    return float(data.score_answer(answer,entry))


def teacher_answer(task,entry):
    import json
    if task=='graph_color': return json.dumps(entry['metadata']['possible_answer'],sort_keys=True)
    if task=='propositional_logic': return entry['metadata']['example_answer']
    if task=='rubiks_cube': return entry['metadata']['example_correct_answer']
    return entry['answer']
