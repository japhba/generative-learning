"""Arm 3 evaluation with vLLM offline (runtime venv): fresh operator episodes (answer-only with "Answer:" prefill, and
step-by-step), transfer to alien units (answer-only), and retention on GSM-symbolic (step-by-step). Writes raw outputs."""
import argparse
import json
from pathlib import Path

from vllm import LLM,SamplingParams

SETS=Path('/ceph/scratch/jbauer/mlayout/msft_eval_sets.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('--out');parser.add_argument('--mm',action='store_true');parser.add_argument('--sets',default=str(SETS));args=parser.parse_args()
    sets=json.loads(Path(args.sets).read_text());llm=LLM(model=args.model,max_model_len=8192,gpu_memory_utilization=.85,seed=0,**({'language_model_only':True} if args.mm else {}))
    kw={'chat_template_kwargs':{'enable_thinking':False}};out={}
    for name,items in sets.items():
        prefill=name.endswith('answer_only')
        messages=[it['messages']+([{'role':'assistant','content':'Answer:'}] if prefill else []) for it in items]
        params=SamplingParams(temperature=0.,max_tokens=12 if prefill else (1024 if name.startswith('gsm_test') else 2048),seed=0)
        results=llm.chat(messages,params,continue_final_message=prefill,add_generation_prompt=not prefill,**kw)
        out[name]=[{'id':it['id'],'answer':it['answer'],'text':r.outputs[0].text,'finish':r.outputs[0].finish_reason} for it,r in zip(items,results)]
        print(name,len(items),flush=True)
    Path(args.out).write_text(json.dumps(out))


if __name__=='__main__': main()
