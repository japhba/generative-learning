"""One-token numerical choices, preserving the full vocabulary policy."""

from hashlib import sha256
import json
from random import Random


CELLS = [{"name":"largest_2digits","rank":1,"digits":2,"shared":0},
         {"name":"largest_close_12digits","rank":1,"digits":12,"shared":10},
         {"name":"second_6digits","rank":2,"digits":6,"shared":4},
         {"name":"second_close_12digits","rank":2,"digits":12,"shared":10}]
LETTERS = "ABCD"


def problems(seed, count, cell):
    rng, result = Random(seed), []
    for _ in range(count):
        if cell["shared"]:
            prefix = rng.randrange(10**(cell["shared"]-1),10**cell["shared"])
            values = [prefix*10**(cell["digits"]-cell["shared"])+v for v in rng.sample(range(10**(cell["digits"]-cell["shared"])),4)]
        else: values = rng.sample(range(10**(cell["digits"]-1),10**cell["digits"]),4)
        label = sorted(range(4),key=lambda i:values[i])[-cell["rank"]]
        row = {"values":values,"rank":cell["rank"],"cell":cell["name"],"label":LETTERS[label]}
        row["id"] = sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:24]
        result.append(row)
    return result


def messages(problem):
    order = "largest" if problem["rank"]==1 else "second-largest"
    lines = "\n".join(f"{letter}: {value}" for letter,value in zip(LETTERS,problem["values"],strict=True))
    return [{"role":"system","content":"Follow the requested answer format exactly."},
            {"role":"user","content":f"Which entry has the {order} integer?\n{lines}\nReply with just its letter: A, B, C, or D."}]


def encode(student, rows):
    labels = {letter:student.tokenizer.encode(letter,add_special_tokens=False) for letter in LETTERS}
    assert all(len(tokens)==1 for tokens in labels.values())
    return [{**row,"input_ids":student.tokenizer.apply_chat_template(messages(row),tokenize=True,add_generation_prompt=True),
             "target_id":labels[row["label"]][0]} for row in rows]
