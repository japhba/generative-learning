"""Select an informative short-output task before any learning-curve search."""

import argparse
import json
import os

import numpy as np

from bestn_precision import exact_matmul
from run_elicitation_pilot import ROOT, MODELS, digest, save, setup
from scaling_choice_model import ChoiceStudent
from scaling_choice_task import CELLS, encode, problems
from train_bestn import stamp


OUTPUT=ROOT/"results/scaling_s4g/choice/calibration"
SOURCES=("calibrate_scaling_choice.py","scaling_choice_model.py","scaling_choice_task.py","bestn_precision.py","bestn_llm.py","train_elicitation.py","run_elicitation_pilot.py","elicitation_task.py","elicitation_answer.py")


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=False);(OUTPUT/"source").mkdir()
    for name in SOURCES:(OUTPUT/"source"/name).write_bytes((ROOT/"experiments"/name).read_bytes())
    save(OUTPUT/"protocol.json",{"created_at_uk":stamp(),"model":MODELS["general"],"cells":CELLS,
         "source_sha256":{name:digest(OUTPUT/"source"/name) for name in SOURCES},"problems_per_cell":64,"seed":98000,
         "selection":"First listed cell with mean full-vocabulary correct-token probability between.3and.9, valid-letter mass at least.95, and greedy accuracy at least.6. If none qualifies, report the failed screen before a redesign.",
         "purpose":"Retain best-of-two self-generation with correctness feedback while eliminating Monte Carlo evaluation noise and the nuisance entropy of long equivalent traces. This one-token task does not test multi-step reasoning exploration.",
         "policy":"Natural full-vocabulary softmax. No temperature change, vocabulary restriction, extra classifier head, or new parameters."})


def run(allocation):
    import torch
    setup();os.environ["HF_XET_HIGH_PERFORMANCE"]="1";assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4);exact_matmul()
    protocol=json.loads((OUTPUT/"protocol.json").read_text())
    assert protocol["source_sha256"]=={name:digest(ROOT/"experiments"/name) for name in SOURCES}
    student=ChoiceStudent();records=[]
    for index,cell in enumerate(CELLS):
        rows=problems(98000+index,64,cell);save(OUTPUT/f"{cell['name']}_tasks.json",rows)
        result=student.evaluate(encode(student,rows));record={"cell":cell,**result}
        save(OUTPUT/f"{cell['name']}.json",record);records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!="records"}),flush=True)
    selected=[r for r in records if .3<=r["accuracy"]<=.9 and r["valid_mass"]>=.95 and r["greedy_accuracy"]>=.6]
    save(OUTPUT/"complete.json",{"finished_at_uk":stamp(),"allocation":allocation,"selected":selected[0]["cell"] if selected else None,
         "summary":[{k:v for k,v in r.items() if k!="records"} for r in records]})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","run"));parser.add_argument("--allocation")
    args=parser.parse_args()
    if args.command=="prepare":prepare()
    else:run(args.allocation)
