"""Arm 3: LoRA fine-tuning of one student on one source's matched solutions (train_venv). Saves adapter, loss curve, merged weights."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer

SENTINEL='XQZSENTINELXQZ'


def encode(tok,messages,limit):
    render=lambda ms,**kw:tok.apply_chat_template(ms,tokenize=False,enable_thinking=False,**kw)
    prompt=render(messages[:-1],add_generation_prompt=True)
    suffix=render(messages[:-1]+[{'role':'assistant','content':SENTINEL}]).split(SENTINEL)[1]  # the template's end-of-turn markup
    p=tok(prompt,add_special_tokens=False).input_ids;r=tok(messages[-1]['content']+suffix,add_special_tokens=False).input_ids
    ids=(p+r)[:limit];labels=([-100]*len(p)+r)[:limit];return ids,labels


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base');parser.add_argument('--data');parser.add_argument('--out');parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-4);parser.add_argument('--batch',type=int,default=16);parser.add_argument('--attention-only',action='store_true');parser.add_argument('--seed',type=int,default=0);args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(args.seed);random.seed(args.seed)
    tok=AutoTokenizer.from_pretrained(args.base);rows=json.loads(Path(args.data).read_text())
    examples=[encode(tok,r['messages'],3072) for r in rows];print('examples',len(examples),'mean tokens',sum(len(e[0]) for e in examples)/len(examples),flush=True)
    model,info=AutoModelForCausalLM.from_pretrained(args.base,dtype=torch.bfloat16,device_map='cuda',output_loading_info=True)
    assert not info['missing_keys'],info['missing_keys'][:10]  # a text-only class must find every language-model weight
    print('unexpected (ignored, e.g. vision tower):',len(info['unexpected_keys']),flush=True);model.gradient_checkpointing_enable();model.enable_input_require_grads()
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj'] if args.attention_only else 'all-linear',task_type='CAUSAL_LM'));model.print_trainable_parameters()
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.)
    steps=args.epochs*math.ceil(len(examples)/args.batch);warmup=10;schedule=lambda s:min(1,(s+1)/warmup)*.5*(1+math.cos(math.pi*min(s,steps)/steps))
    log=[];step=0;pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;started=time.time();first=0
    done=[e for e in range(args.epochs) if (out/f'adapter_epoch{e+1}/state.pt').exists()]
    if done:  # resume after the last completed epoch: adapter weights, optimizer, step, log, RNG
        state=torch.load(out/f'adapter_epoch{done[-1]+1}/state.pt',weights_only=False);set_peft_model_state_dict(model,load_file(out/f'adapter_epoch{done[-1]+1}/adapter_model.safetensors'))
        optimizer.load_state_dict(state['optimizer']);step=state['step'];log=state['log'];torch.set_rng_state(state['rng']);first=done[-1]+1;print('resumed after epoch',first,flush=True)
    for epoch in range(first,args.epochs):
        order=list(range(len(examples)));random.Random(epoch+1000*args.seed).shuffle(order)  # seed 0 reproduces the original order
        for b in range(0,len(order),args.batch):
            batch=[examples[i] for i in order[b:b+args.batch]];tokens=sum(sum(l!=-100 for l in e[1]) for e in batch);total=0.
            for e in batch:  # one sequence per micro-step (no padding); logits only where a solution token is predicted
                ids=torch.tensor([e[0]],device='cuda');labels=torch.tensor(e[1],device='cuda');keep=(labels[1:]!=-100).nonzero().squeeze(1)
                logits=model(input_ids=ids,logits_to_keep=keep).logits[0].float()
                loss=torch.nn.functional.cross_entropy(logits,labels[keep+1],reduction='sum')/tokens
                loss.backward();total+=loss.item()
            for g in optimizer.param_groups: g['lr']=args.lr*schedule(step)
            torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();optimizer.zero_grad();step+=1
            log.append({'step':step,'epoch':epoch,'loss':total,'lr':args.lr*schedule(step-1),'seconds':time.time()-started});print(log[-1],flush=True)
        model.save_pretrained(out/f'adapter_epoch{epoch+1}');(out/'loss.json').write_text(json.dumps(log,indent=1))
        torch.save({'optimizer':optimizer.state_dict(),'step':step,'log':log,'rng':torch.get_rng_state()},out/f'adapter_epoch{epoch+1}/state.pt')
    merged=model.merge_and_unload();merged.save_pretrained(out/'merged',safe_serialization=True);tok.save_pretrained(out/'merged')
    (out/'done.json').write_text(json.dumps({'steps':step,'seconds':time.time()-started,'examples':len(examples)}))


if __name__=='__main__': main()
