"""Short arithmetic/counting tasks with a fixed unfinished answer prefix."""

from hashlib import sha256
import json
from random import Random


CELLS=[{"name":"last_digit_2","family":"last_digit","digits":2},
       {"name":"last_digit_6","family":"last_digit","digits":6},
       {"name":"count_8","family":"count","length":8,"max_count":8},
       {"name":"count_16","family":"count","length":16,"max_count":9}]
SYMBOLS="0123456789"


def problems(seed,count,cell):
    rng,result=Random(seed),[]
    for _ in range(count):
        if cell["family"]=="last_digit":
            a,b=[rng.randrange(10**cell["digits"]) for _ in range(2)]
            row={"cell":cell["name"],"family":"last_digit","a":a,"b":b,"label":str((a+b)%10)}
        else:
            target=rng.randrange(10);number=rng.randrange(cell["max_count"]+1)
            places=set(rng.sample(range(cell["length"]),number));other=[v for v in range(10) if v!=target]
            values=[target if i in places else rng.choice(other) for i in range(cell["length"])]
            row={"cell":cell["name"],"family":"count","target":target,"values":values,"label":str(number)}
        row["id"]=sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:24];result.append(row)
    return result


def messages(row):
    if row["family"]=="last_digit":
        question=f"What is the last decimal digit of {row['a']} + {row['b']}? Give just that one digit."
    else:
        values=" ".join(map(str,row["values"]))
        question=f"How many copies of the digit {row['target']} occur in this list?\n{values}\nGive just the count as one digit."
    return [{"role":"system","content":"Answer the question accurately."},{"role":"user","content":question}]


def encode(student,rows):
    ids={s:student.tokenizer.encode(s,add_special_tokens=False) for s in SYMBOLS}
    assert all(len(v)==1 for v in ids.values())
    result=[]
    for row in rows:
        prefix=student.tokenizer.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True)+"Answer: "
        result.append({**row,"input_ids":student.tokenizer.encode(prefix,add_special_tokens=False),"target_id":ids[row["label"]][0]})
    return result
