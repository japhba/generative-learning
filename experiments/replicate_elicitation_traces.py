"""Prospective three-seed confirmation of the unchanged full-trace main contrast."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from elicitation_task import make_problems
from run_elicitation_pilot import ROOT, MODELS, digest, save, setup
from train_elicitation import counters, add_generation, append, read_rows
from train_elicitation_traces import TraceStudent, targets, evaluate


ARMS = ("frozen_reasoning", "on_policy_reasoning")
SEEDS = (71000,71001,71002)
CHECKPOINTS = (0,16,64,192,384)
SOURCES = ("replicate_elicitation_traces.py", "train_elicitation_traces.py", "train_elicitation.py",
           "elicitation_task.py", "elicitation_answer.py", "run_elicitation_pilot.py")


def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    train,evaluation = make_problems(71100,32,("arithmetic",)),make_problems(71200,48,("arithmetic",))
    previous=[]
    for name in ("elicitation_consolidation", "elicitation_traces"):
        for split in ("train", "evaluation"): previous.extend(json.loads((ROOT/f"results/{name}/{split}.json").read_text()))
    for path in ("calibration.json", "general_confirmation/confirmation.json"):
        previous.extend(json.loads((ROOT/"results/elicitation_pilot"/path).read_text()))
    assert not {p["id"] for p in train+evaluation}&{p["id"] for p in previous}
    assert not {p["id"] for p in train}&{p["id"] for p in evaluation}
    save(output/"train.json",train)
    save(output/"evaluation.json",evaluation)
    (output/"source").mkdir()
    for name in SOURCES: (output/"source"/name).write_bytes((ROOT/"experiments"/name).read_bytes())
    protocol={"created_at_utc":datetime.now(timezone.utc).isoformat(),"model":MODELS["general"],"seeds":SEEDS,
              "train_seed":71100,"evaluation_seed":71200,"train_unique":96,"epochs":4,"evaluation_size":144,
              "arms":ARMS,"checkpoints":CHECKPOINTS,"batch_size":16,"learning_rate":1e-6,"updates_per_batch":1,
              "source_sha256":{name:digest(output/"source"/name) for name in SOURCES},
              "train_sha256":digest(output/"train.json"),"evaluation_sha256":digest(output/"evaluation.json"),
              "change":"No change to the pilot's main-pair learner, target rule, prompts, rollout budgets, optimizer, normalization, or checkpoints. Fresh practice/evaluation cohorts and three new optimization seeds. Raw one-rollout control is not repeated.",
              "primary":"Final current-minus-frozen reasoning accuracy across three paired optimization seeds and 144 new questions. No interim scoring or hyperparameter changes. No exponent fitting.",
              "resources":"Three institutional H100 workers on gpu_lowp. Save and SHA-256 archive final seed71000 models only. No paid APIs or Vast."}
    save(output/"protocol.json",protocol)
    for seed in SEEDS:
        directory=output/str(seed)
        directory.mkdir()
        save(directory/"protocol.json",{**protocol,"seed":seed})
        for name in ("train.json", "evaluation.json"): (directory/name).symlink_to(Path("..")/name)
        (directory/"source").symlink_to(Path("../source"))


def run(output,seed):
    import torch
    assert seed in SEEDS
    cache=setup()/"elicitation_trace_confirmation"/str(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    protocol=json.loads((output/"protocol.json").read_text())
    assert protocol["source_sha256"]=={name:digest(ROOT/"experiments"/name) for name in SOURCES}
    train=json.loads((output/"train.json").read_text())
    evaluation=json.loads((output/"evaluation.json").read_text())
    directory=output/str(seed)
    rng=np.random.default_rng(seed+100000)
    order=np.concatenate([rng.permutation(96) for _ in range(4)]).tolist()
    sequence=[train[i] for i in order]
    save(directory/"order.json",order)
    frozen=directory/"frozen_candidates"
    frozen.mkdir()
    student=TraceStudent()
    save(directory/"objective_check.json",student.verify_objective(train[0]))
    save(directory/"environment.json",{"job_id":os.environ["SLURM_JOB_ID"],"device":torch.cuda.get_device_name(),
         "torch":torch.__version__,"cuda":torch.version.cuda,"parameter_count":sum(p.numel() for p in student.model.parameters())})
    initial={key:value.detach().cpu().clone() for key,value in student.model.state_dict().items()}
    for step,begin in enumerate(tqdm(range(0,384,16),desc=f"{seed} frozen candidates")):
        student.generate(sequence[begin:begin+16],"reason",4,1.,512,32,seed+200000+step,frozen/f"round_{step}.jsonl")
    for arm in ARMS:
        student.model.load_state_dict(initial,strict=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        optimizer=torch.optim.AdamW(student.model.parameters(),lr=1e-6,weight_decay=0.,fused=True)
        current=directory/arm
        current.mkdir()
        count=counters()
        evaluate(student,evaluation,train,current,0,count)
        for step,begin in enumerate(tqdm(range(0,384,16),desc=f"{seed} {arm}")):
            problems=sequence[begin:begin+16]
            if arm=="frozen_reasoning":
                path=frozen/f"round_{step}.jsonl"
                usage=json.loads(path.with_suffix(".usage.json").read_text())
            else:
                path=current/f"round_{step}.jsonl"
                usage=student.generate(problems,"reason",4,1.,512,32,seed+200000+step,path)
            target,selected=targets(read_rows(path),problems,arm)
            count["practice_presentations"]+=16
            count["valid_targets"]+=sum(bool(t) for t in target)
            add_generation(count,usage)
            loss=student.update(problems,target,optimizer,count)
            append(current/"updates.jsonl",{"step":step,"budget":begin+16,"selected":selected,"loss":loss,
                   "candidate_path":str(path),"candidate_sha256":digest(path),"counters":dict(count)})
            if count["practice_presentations"] in CHECKPOINTS: evaluate(student,evaluation,train,current,count["practice_presentations"],count)
        if seed==SEEDS[0]:
            destination=cache/arm
            destination.mkdir(parents=True,exist_ok=False)
            student.model.save_pretrained(destination)
            student.tokenizer.save_pretrained(destination)
            save(current/"saved_model.json",{"path":str(destination),"files":{p.name:digest(p) for p in destination.iterdir() if p.is_file()}})
        save(current/"complete.json",{"counters":count,"finished_at_utc":datetime.now(timezone.utc).isoformat()})
        del optimizer
        torch.cuda.empty_cache()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--output",type=Path,default=ROOT/"results/elicitation_trace_confirmation")
    parser.add_argument("--seed",type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.command=="prepare": prepare(args.output)
    else: run(args.output,args.seed)
