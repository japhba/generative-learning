"""Fresh one-seed follow-up preserving the model's actual reasoning trajectories.

This changes both the representation taught and update strength relative to the
answer-only pilot. It tests a revised recipe, not a factorial attribution of the
earlier failure to a single cause. No gold labels or rewards enter training.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm

from elicitation_answer import parse_answer
from elicitation_task import consensus, make_problems, messages
from run_elicitation_pilot import ROOT, MODELS, digest, save, setup
from train_elicitation import Student, counters, add_generation, append, read_rows


ARMS = ("frozen_reasoning", "on_policy_reasoning", "raw_reasoning_self")
CHECKPOINTS = (0,16,64,192,384)
SOURCES = ("train_elicitation_traces.py", "train_elicitation.py", "elicitation_task.py", "elicitation_answer.py", "run_elicitation_pilot.py")


class TraceStudent(Student):
    def update(self, problems, targets, optimizer, count):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        self.model.train()
        inputs, labels = [], []
        for p, target in zip(problems,targets,strict=True):
            prefix = self.tokenizer.apply_chat_template(messages(p,"reason"), tokenize=True, add_generation_prompt=True)
            inputs.append(torch.tensor(prefix+target,dtype=torch.long))
            labels.append(torch.tensor([-100]*len(prefix)+target,dtype=torch.long))
        ids = pad_sequence(inputs,batch_first=True,padding_value=self.tokenizer.pad_token_id).cuda()
        attention = pad_sequence([torch.ones(len(r),dtype=torch.long) for r in inputs],batch_first=True,padding_value=0).cuda()
        labels = pad_sequence(labels,batch_first=True,padding_value=-100).cuda()
        valid = labels[:,1:] != -100
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            hidden = self.model.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state[:,:-1]
            logits = self.model.lm_head(hidden[valid]).float()
            loss = torch.nn.functional.cross_entropy(logits,labels[:,1:][valid],reduction="sum")/(128*len(problems))
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        count["training_seconds"] += time.monotonic()-start
        count["optimizer_updates"] += 1
        count["training_presentations"] += len(problems)
        count["supervised_tokens"] += int(valid.sum())
        count["training_input_tokens"] += int(attention.sum())
        count["padded_training_tokens"] += ids.numel()
        return float(loss.detach())


def targets(candidates, problems, arm):
    width = 1 if arm == "raw_reasoning_self" else 4
    assert len(candidates) == width*len(problems)
    result, records = [], []
    for i,problem in enumerate(problems):
        group = [{**r,"answer":parse_answer(r["text"])} for r in candidates[i*width:(i+1)*width]]
        assert [r["id"] for r in group] == [problem["id"]]*width
        vote = consensus(group)
        if arm == "raw_reasoning_self": selected = 0
        else: selected = vote["selected_index"]
        target = [] if selected is None else group[selected]["token_ids"]
        result.append(target)
        records.append({"id":problem["id"],"vote":vote,"selected_index":selected,"target_ids":target,
                        "target_text":"" if selected is None else group[selected]["text"]})
    return result,records


def evaluate(student, evaluation, train, directory, budget, count):
    usage = {}
    for mode,maximum in (("direct",96),("reason",512)):
        usage[mode] = student.generate(evaluation,mode,1,0.,maximum,16,1690000,directory/f"eval_{mode}_n{budget}.jsonl")
    usage["practice"] = student.generate(train,"reason",1,0.,512,16,1680000,directory/f"practice_reason_n{budget}.jsonl")
    append(directory/"checkpoints.jsonl",{"budget":budget,"counters":dict(count),"evaluation_usage":usage})


def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    train,evaluation=make_problems(69100,32,("arithmetic",)),make_problems(69200,48,("arithmetic",))
    old=[]
    for path in ["results/elicitation_consolidation/train.json","results/elicitation_consolidation/evaluation.json",
                 "results/elicitation_pilot/calibration.json","results/elicitation_pilot/general_confirmation/confirmation.json"]:
        old.extend(json.loads((ROOT/path).read_text()))
    assert not {r["id"] for r in train+evaluation}&{r["id"] for r in old}
    assert not {r["id"] for r in train}&{r["id"] for r in evaluation}
    save(output/"train.json",train)
    save(output/"evaluation.json",evaluation)
    (output/"source").mkdir()
    for name in SOURCES: (output/"source"/name).write_bytes((ROOT/"experiments"/name).read_bytes())
    save(output/"protocol.json",{"created_at_utc":datetime.now(timezone.utc).isoformat(),"model":MODELS["general"],
         "seed":69000,"train_seed":69100,"evaluation_seed":69200,"train_unique":96,"epochs":4,"evaluation_size":144,
         "arms":ARMS,"checkpoints":CHECKPOINTS,"batch_size":16,"learning_rate":1e-6,"updates_per_batch":1,
         "source_sha256":{name:digest(output/"source"/name) for name in SOURCES},
         "train_sha256":digest(output/"train.json"),"evaluation_sha256":digest(output/"evaluation.json"),
         "change":"Retain the actual selected reasoning token sequence; train and evaluate reasoning under its original prompt. LR1e-6 and one update per batch. This is a revised recipe with multiple changes, not a causal isolation of trace preservation.",
         "selection":"Same four-sample internal plurality, no confidence/correctness filtering. The raw-reasoning control learns from one unfiltered current-model trajectory, including incomplete attempts.",
         "objective":"Full FP32 weights/AdamW state, BF16 autocast, no decay, clip1; summed completion NLL divided by fixed128*batch_size. Standard existing LM head.",
         "primary":"Final reasoning accuracy: on-policy minus frozen-source, plus each versus its common baseline. One exploratory optimization seed; no claim of seed replication or scaling exponents.",
         "cost":"All forward/backward/generated tokens reported. Main pair matched in questions, rollout counts, and updates; actual lengths differ. No paid APIs or external labels."})


def run(output):
    import torch
    cache=setup()/"elicitation_traces"
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(69000)
    torch.cuda.manual_seed_all(69000)
    protocol=json.loads((output/"protocol.json").read_text())
    assert protocol["source_sha256"]=={name:digest(ROOT/"experiments"/name) for name in SOURCES}
    train=json.loads((output/"train.json").read_text())
    evaluation=json.loads((output/"evaluation.json").read_text())
    rng=np.random.default_rng(169000)
    order=np.concatenate([rng.permutation(96) for _ in range(4)]).tolist()
    sequence=[train[i] for i in order]
    save(output/"order.json",order)
    frozen=output/"frozen_candidates"
    frozen.mkdir()
    student=TraceStudent()
    save(output/"objective_check.json",student.verify_objective(train[0]))
    save(output/"environment.json",{"job_id":os.environ["SLURM_JOB_ID"],"device":torch.cuda.get_device_name(),
         "torch":torch.__version__,"cuda":torch.version.cuda,"parameter_count":sum(p.numel() for p in student.model.parameters())})
    initial={key:value.detach().cpu().clone() for key,value in student.model.state_dict().items()}
    for step,begin in enumerate(tqdm(range(0,384,16),desc="Frozen reasoning samples")):
        student.generate(sequence[begin:begin+16],"reason",4,1.,512,32,269000+step,frozen/f"round_{step}.jsonl")
    for arm in ARMS:
        student.model.load_state_dict(initial,strict=True)
        torch.manual_seed(69000)
        torch.cuda.manual_seed_all(69000)
        optimizer=torch.optim.AdamW(student.model.parameters(),lr=1e-6,weight_decay=0.,fused=True)
        directory=output/arm
        directory.mkdir()
        count=counters()
        evaluate(student,evaluation,train,directory,0,count)
        for step,begin in enumerate(tqdm(range(0,384,16),desc=arm)):
            problems=sequence[begin:begin+16]
            if arm=="frozen_reasoning":
                path=frozen/f"round_{step}.jsonl"
                usage=json.loads(path.with_suffix(".usage.json").read_text())
            else:
                path=directory/f"round_{step}.jsonl"
                usage=student.generate(problems,"reason",1 if arm=="raw_reasoning_self" else 4,1.,512,32,269000+step,path)
            target,selected=targets(read_rows(path),problems,arm)
            count["practice_presentations"]+=16
            count["valid_targets"]+=sum(bool(t) for t in target)
            add_generation(count,usage)
            loss=student.update(problems,target,optimizer,count)
            append(directory/"updates.jsonl",{"step":step,"budget":begin+16,"selected":selected,"loss":loss,
                                              "candidate_path":str(path),"candidate_sha256":digest(path),"counters":dict(count)})
            if count["practice_presentations"] in CHECKPOINTS:
                evaluate(student,evaluation,train,directory,count["practice_presentations"],count)
        destination=cache/arm
        destination.mkdir(parents=True,exist_ok=False)
        student.model.save_pretrained(destination)
        student.tokenizer.save_pretrained(destination)
        save(directory/"saved_model.json",{"path":str(destination),"files":{p.name:digest(p) for p in destination.iterdir() if p.is_file()}})
        save(directory/"complete.json",{"counters":count,"finished_at_utc":datetime.now(timezone.utc).isoformat()})
        del optimizer
        torch.cuda.empty_cache()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--output",type=Path,default=ROOT/"results/elicitation_traces")
    args=parser.parse_args()
    if args.command=="prepare": prepare(args.output)
    else: run(args.output)
