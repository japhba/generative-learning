"""Supplemental exact endpoint evaluation over all IID command sequences."""

import argparse
import gc
import json
from pathlib import Path

from tqdm.auto import tqdm

from train_feedback import sha256
from train_planning import configure_environment, initialize, load_model
from tracking_exact import OFFSETS, exact_metrics, oracle_policy
from tracking_task import LETTERS, prompt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    configure_environment()
    import torch
    grid=[{'u':u,'e':e} for e in range(-8,9) for u in range(-2,3)]
    results=[]
    for directory in tqdm(args.runs,desc='Exact endpoint policies'):
        config=json.loads((directory/'config.json').read_text())
        initialize(config['seed'])
        for arm in config['arms']:
            final=json.loads((directory/f'{arm}_final.json').read_text())
            checkpoint=Path(final['checkpoint'])
            assert sha256(checkpoint/'model.safetensors')==final['model_sha256']
            model,tokenizer=load_model(str(checkpoint))
            model.eval()
            model.requires_grad_(False)
            tokenizer.padding_side='left'
            tokens=[tokenizer.encode(' '+letter,add_special_tokens=False) for letter in LETTERS]
            assert all(len(value)==1 for value in tokens)
            actions=[]
            for begin in range(0,len(grid),128):
                rows=grid[begin:begin+128]
                count=len(rows)
                padded=rows+[{'u':-2,'e':-8}]*(128-count)
                batch=tokenizer([prompt(row) for row in padded],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(**batch,use_cache=False,logits_to_keep=1).logits[:,-1,[value[0] for value in tokens]].float()
                    actions.extend((logits.argmax(-1)-3).cpu().tolist()[:count])
            policy={(row['u'],row['e']):action for row,action in zip(grid,actions,strict=True)}
            assert all(-8<=e+action-u<=8 for (u,e),action in policy.items()), 'The initial error domain is not closed under this policy.'
            # This detects batching/numerical disagreements with the already
            # recorded protocol instead of silently treating a different lookup
            # policy as the evaluated model.
            predictions=json.loads((directory/f"{arm}_n{config['labels']}_predictions.json").read_text())
            checked=0
            for row in predictions:
                error=row['episode']['initial_error']
                for command,action in zip(row['episode']['commands'],row['actions'],strict=True):
                    assert policy[command,error]==action-3, 'Fixed-grid action differs from the recorded greedy trajectory.'
                    error+=action-3-command
                    checked+=1
            results.append({'model':config['model'],'seed':config['seed'],'arm':arm,'model_sha256':final['model_sha256'],
                            'metrics':{stratum:exact_metrics(policy,offsets) for stratum,offsets in OFFSETS.items()},
                            'recorded_actions_matched':checked,'grid_actions':actions,'closed_error_domain':[-8,8]})
            del model,tokenizer,batch,logits
            gc.collect()
            torch.cuda.empty_cache()
    summary={'grid':grid,'results':results,'oracle_metrics':{stratum:exact_metrics(oracle_policy(),offsets) for stratum,offsets in OFFSETS.items()},
             'interpretation':'Exact dynamic programming for each saved deterministic policy lookup, over IID uniform commands and prescribed initial-error distribution for16steps. The85input grid in e=-8..8 is padded to128rows, verified closed under every action, and must match every recorded final test action. Supplemental endpoint check under this numerical batching convention; original sampled-episode primary results retained. An earlier unnecessary wider grid changed padding and flipped a near-tied expert action; its failure is retained.',
             'source_sha256':sha256(Path(__file__))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f'Computed exact metrics for {len(results)} final policies over all command sequences.')


if __name__=='__main__': main()
