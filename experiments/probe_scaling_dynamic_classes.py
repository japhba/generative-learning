"""Measure changing class-gradient interactions at saved LLM checkpoints."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bestn_llm import dot
from bestn_precision import exact_matmul
from run_elicitation_pilot import ROOT,digest,save,setup
from scaling_choice_model import ChoiceStudent
from scaling_digit_task import SYMBOLS,encode
from train_bestn import stamp


def main():
    import torch
    setup();os.environ["HF_XET_HIGH_PERFORMANCE"]="1";assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4);exact_matmul()
    output=ROOT/"results/scaling_s4g/class_geometry/dynamic";output.mkdir(exist_ok=False)
    save(output/"protocol.json",{"created_at_uk":stamp(),"source_sha256":digest(Path(__file__)),"steps":[32,128,512,1024],
         "status":"Retrospective diagnostic, after the constant initial three-mode reduction failed. No weights train in this probe.",
         "measurement":"At saved pilot checkpoints, compute exact class-conditional success gradients on the same512 observer questions. Their Gram matrix gives cooperation/interference and the population best-of-two update's instantaneous effect on each class within this empirical observer distribution.",
         "limit":"This identifies changing optimization geometry. Such changes need not imply new hidden features; they can occur in nonlinear probabilities with fixed underlying features."})
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    cohort=ROOT/"results/scaling_s4g/digit/pilot/99101"
    initial=json.loads((ROOT/"results/scaling_s4g/class_geometry/results.json").read_text())
    numeric=json.loads((cohort/"forecast.json").read_text())
    student=ChoiceStudent();student.answer_symbols=SYMBOLS
    rows=encode(student,json.loads((cohort/"observer.json").read_text()))
    labels=np.array([int(r["label"]) for r in rows]);prior=np.bincount(labels,minlength=3)/len(rows)
    p=np.array([np.exp(r["correct_logprob"]) for r in numeric["train"]["records"]])
    gram=np.array(initial["train_gram"])
    records=[{"step":0,"cosines":initial["train_cosines"],"gram":initial["train_gram"],"class_success":[float(p[labels==c].mean()) for c in range(3)],
              "predicted_class_gain_per_update":(numeric["eta"]*gram@prior).tolist()}]
    for step in (32,128,512,1024):
        marker=json.loads((cohort/f"current/step{step:03d}_attempt0.pt.archive.json").read_text())
        state=torch.load(marker["persistent"],map_location="cpu",weights_only=False)
        student.model.load_state_dict(state["model"],strict=True);del state
        gradients=[];success=[]
        for count in range(3):
            probability,g=student.success_gradient([r for r in rows if int(r["label"])==count]);gradients.append(g);success.append(probability)
        gram=np.array([[dot(a,b) for b in gradients] for a in gradients]);del gradients
        records.append({"step":step,"cosines":(gram/np.sqrt(np.outer(np.diag(gram),np.diag(gram)))).tolist(),"gram":gram.tolist(),
                        "class_success":success,"predicted_class_gain_per_update":(numeric["eta"]*gram@prior).tolist()})
        save(output/f"step{step:04d}.json",records[-1]);print(json.dumps(records[-1]),flush=True)
    save(output/"results.json",{"finished_at_uk":stamp(),"allocation":os.environ["SLURM_JOB_ID"],"records":records})
    fig,ax=plt.subplots()
    for a,b in ((0,1),(0,2),(1,2)):
        ax.plot([r["step"] for r in records],[r["cosines"][a][b] for r in records],"o-",label=f"Count {a} vs {b}")
    ax.axhline(0,color="gray",linestyle="--")
    ax.set(xlabel="SGD updates",ylabel="Cosine between class-success gradients",title="The LLM's learning directions interact differently over time")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-class-interactions.png");plt.close(fig)


if __name__=="__main__":main()
