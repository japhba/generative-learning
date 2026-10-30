"""Retrospective three-mode reduction using initial class-gradient overlaps."""

import json
import os
from pathlib import Path

import numpy as np

from bestn_llm import dot
from bestn_precision import exact_matmul
from run_elicitation_pilot import ROOT,digest,save,setup
from scaling_choice_model import ChoiceStudent
from scaling_conditional_gate import probabilities
from scaling_digit_task import SYMBOLS,encode
from train_bestn import stamp


OUTPUT=ROOT/"results/scaling_s4g/class_geometry"


def main():
    import torch
    setup();os.environ["HF_XET_HIGH_PERFORMANCE"]="1";assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4);exact_matmul()
    OUTPUT.mkdir(exist_ok=False)
    save(OUTPUT/"protocol.json",{"created_at_uk":stamp(),"source_sha256":digest(Path(__file__)),
         "status":"Retrospective diagnostic chosen after seeing class-specific learning phases. All coefficients are computed from the initial model, but model structure was chosen after observing the curve; this is not an independent predictive confirmation.",
         "question":"Do overlaps among initial success gradients for correct-count0,1,2 explain their interference, and how much final parameter displacement lies in that initial three-dimensional span?",
         "reduction":"Three coupled conditional Boltzmann success gates, with fixed empirical per-question initial biases. Kernel entries are initial class-gradient overlaps divided by initial Bernoulli sensitivities. Constant-kernel population updates use the actual learner's frozen rate; no curve parameter is optimized.",
         "scope":"This groups questions by their true answer for diagnostic coarse-graining; it is not an implemented autonomous classifier with only three parameters. It does not prove feature learning or new information from generation."})
    (OUTPUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    cohort=ROOT/"results/scaling_s4g/digit/pilot/99101"
    numeric=json.loads((cohort/"forecast.json").read_text());student=ChoiceStudent();student.answer_symbols=SYMBOLS
    initial=[p.detach().clone() for p in student.model.parameters()]
    rows={split:encode(student,json.loads((cohort/f"{split}.json").read_text())) for split in ("observer","evaluation")}
    gradient={};bias={};labels={};sensitivity={};prior={}
    for split,field in (("observer","train"),("evaluation","evaluation")):
        labels[split]=np.array([int(r["label"]) for r in rows[split]])
        p=np.array([np.exp(r["correct_logprob"]) for r in numeric[field]["records"]])
        bias[split]=np.log(p)-np.log1p(-p)
        sensitivity[split]=np.array([np.mean(p[labels[split]==c]*(1-p[labels[split]==c])) for c in range(3)])
        prior[split]=np.array([np.mean(labels[split]==c) for c in range(3)])
        gradient[split]=[]
        for count in range(3):
            _,g=student.success_gradient([r for r in rows[split] if int(r["label"])==count])
            for value in g:value.neg_()
            gradient[split].append(g)
    train_gram=np.array([[dot(a,b) for b in gradient["observer"]] for a in gradient["observer"]])
    cross_gram=np.array([[dot(a,b) for b in gradient["observer"]] for a in gradient["evaluation"]])
    eval_gram=np.array([[dot(a,b) for b in gradient["evaluation"]] for a in gradient["evaluation"]])
    train_kernel=train_gram/np.outer(sensitivity["observer"],sensitivity["observer"])
    cross_kernel=cross_gram/np.outer(sensitivity["evaluation"],sensitivity["observer"])
    shifts={split:np.zeros(3) for split in rows};curve=[]
    for step in range(1025):
        p={split:probabilities(bias[split]+shifts[split][labels[split]]) for split in rows}
        curve.append({"step":step,"accuracy":float(p["evaluation"].mean()),"by_class":[float(p["evaluation"][labels["evaluation"]==c].mean()) for c in range(3)]})
        force=prior["observer"]*np.array([np.mean(p["observer"][labels["observer"]==c]*(1-p["observer"][labels["observer"]==c])) for c in range(3)])
        shifts["observer"]+=numeric["eta"]*train_kernel@force
        shifts["evaluation"]+=numeric["eta"]*cross_kernel@force
    archive=json.loads((cohort/"current/step1024_attempt0.pt.archive.json").read_text())
    state=torch.load(archive["persistent"],map_location="cpu",weights_only=False)
    student.model.load_state_dict(state["model"],strict=True);del state
    displacement=[];norm_squared=0.
    with torch.no_grad():
        for parameter,original in zip(student.model.parameters(),initial,strict=True):
            delta=parameter-original;norm_squared+=float(torch.sum(delta.square(),dtype=torch.float64))
        for g in gradient["observer"]:
            displacement.append(sum(float(torch.sum((parameter-original)*value,dtype=torch.float64)) for parameter,original,value in zip(student.model.parameters(),initial,g,strict=True)))
    projected=np.array(displacement)@np.linalg.solve(train_gram,np.array(displacement))
    result={"finished_at_uk":stamp(),"allocation":os.environ["SLURM_JOB_ID"],"train_gram":train_gram.tolist(),"eval_train_gram":cross_gram.tolist(),
            "train_cosines":(train_gram/np.sqrt(np.outer(np.diag(train_gram),np.diag(train_gram)))).tolist(),
            "eval_train_cosines":(cross_gram/np.sqrt(np.outer(np.diag(eval_gram),np.diag(train_gram)))).tolist(),
            "initial_gain":float(numeric["eta"]*prior["evaluation"]@cross_gram@prior["observer"]),
            "train_kernel":train_kernel.tolist(),"cross_kernel":cross_kernel.tolist(),"curve":curve,
            "final_displacement_squared":norm_squared,"projection_squared":float(projected),"three_mode_displacement_cosine":float(np.sqrt(projected/norm_squared))}
    save(OUTPUT/"results.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="curve"},indent=2))


if __name__=="__main__":main()
