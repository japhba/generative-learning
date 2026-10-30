"""A separately declared redesign after the label-choice screen failed."""

import argparse
import json
import os
from pathlib import Path

from bestn_precision import exact_matmul
from run_elicitation_pilot import ROOT,MODELS,digest,save,setup
from scaling_choice_model import ChoiceStudent
from scaling_digit_task import CELLS,SYMBOLS,encode,problems
from train_bestn import stamp


OUTPUT=ROOT/"results/scaling_s4g/digit/calibration"
SOURCES=("calibrate_scaling_digit.py","scaling_digit_task.py","scaling_choice_model.py","bestn_precision.py","bestn_llm.py","train_elicitation.py","run_elicitation_pilot.py","elicitation_task.py","elicitation_answer.py")


def prepare(output,cells,seed,purpose):
    output.mkdir(parents=True,exist_ok=False);(output/"source").mkdir()
    for name in SOURCES:(output/"source"/name).write_bytes((ROOT/"experiments"/name).read_bytes())
    save(output/"protocol.json",{"created_at_uk":stamp(),"model":MODELS["general"],"cells":cells,"seed":seed,"problems_per_cell":64,
         "source_sha256":{name:digest(output/"source"/name) for name in SOURCES},"why_redesign":purpose,
         "selection":"First listed cell with correct-token probability in[.3,.9], digit mass at least.95, and greedy accuracy at least.6. No learning curves have been measured on these cells.",
         "policy":"Full vocabulary, temperature1, exactly one generated token. The same fixed Answer prefix appears for all sources and observations. No answer vocabulary mask or new head."})


def run(output,allocation):
    import torch
    setup();os.environ["HF_XET_HIGH_PERFORMANCE"]="1";assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4);exact_matmul()
    protocol=json.loads((output/"protocol.json").read_text())
    assert protocol["source_sha256"]=={name:digest(ROOT/"experiments"/name) for name in SOURCES}
    student=ChoiceStudent();student.answer_symbols=SYMBOLS;records=[]
    for index,cell in enumerate(protocol["cells"]):
        rows=problems(protocol["seed"]+index,64,cell);save(output/f"{cell['name']}_tasks.json",rows)
        result=student.evaluate(encode(student,rows));record={"cell":cell,**result}
        save(output/f"{cell['name']}.json",record);records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!="records"}),flush=True)
    eligible=[r for r in records if .3<=r["accuracy"]<=.9 and r["valid_mass"]>=.95 and r["greedy_accuracy"]>=.6]
    save(output/"complete.json",{"finished_at_uk":stamp(),"allocation":allocation,"selected":eligible[0]["cell"] if eligible else None,
         "summary":[{k:v for k,v in r.items() if k!="records"} for r in records]})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","run"));parser.add_argument("--allocation")
    parser.add_argument("--output",type=Path,default=OUTPUT);parser.add_argument("--specification",type=Path)
    args=parser.parse_args()
    if args.command=="prepare":
        specification=json.loads(args.specification.read_text())
        prepare(args.output,specification["cells"],specification["seed"],specification["purpose"])
    else:run(args.output,args.allocation)
