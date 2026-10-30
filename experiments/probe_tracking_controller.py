"""Fixed-grid behavioral control gains; these are not raw transformer weights."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from train_feedback import sha256
from train_planning import configure_environment, initialize, load_model
from tracking_task import LETTERS, prompt, target_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    configure_environment()
    import torch
    grid = [{'u':u,'e':e,'target':target_action(u,e)} for u in range(-2,3) for e in range(-3,4) if abs(u-e)<=3]
    design = np.array([[1.,row['u'],row['e']] for row in grid])
    assert np.linalg.matrix_rank(design)==3
    results=[]
    for directory in tqdm(args.runs,desc='Probe model control gains'):
        config=json.loads((directory/'config.json').read_text())
        initialize(config['seed'])
        for arm in ('warm','expert','on_policy','random','frozen_policy'):
            final=json.loads((directory/f'{arm}_final.json').read_text())
            checkpoint=Path(final['checkpoint'])
            assert sha256(checkpoint/'model.safetensors')==final['model_sha256']
            model,tokenizer=load_model(str(checkpoint))
            model.eval()
            model.requires_grad_(False)
            tokenizer.padding_side='left'
            tokens=[tokenizer.encode(' '+letter,add_special_tokens=False) for letter in LETTERS]
            assert all(len(value)==1 for value in tokens)
            batch=tokenizer([prompt(row) for row in grid],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(**batch,use_cache=False,logits_to_keep=1).logits[:,-1,[value[0] for value in tokens]].float()
                logp=logits.log_softmax(-1).cpu().numpy()
            p=np.exp(logp)
            greedy=p.argmax(-1)-3
            expected=p@np.arange(-3,4)
            targets=np.array([row['target']-3 for row in grid])
            q=np.full_like(p,0.1/7)
            q[np.arange(len(grid)),targets+3]+=0.9
            greedy_coef=np.linalg.lstsq(design,greedy,rcond=None)[0]
            expected_coef=np.linalg.lstsq(design,expected,rcond=None)[0]
            results.append({'run':str(directory.resolve()),'model':config['model'],'seed':config['seed'],'arm':arm,
                            'model_sha256':final['model_sha256'],'parameter_count':sum(value.numel() for value in model.parameters()),
                            'grid_greedy_accuracy':float(np.mean(greedy==targets)),
                            'greedy_control_coefficients':dict(zip(('intercept','command','error'),greedy_coef.tolist(),strict=True)),
                            'expected_action_coefficients':dict(zip(('intercept','command','error'),expected_coef.tolist(),strict=True)),
                            'greedy_linear_fit_mse':float(np.mean((design@greedy_coef-greedy)**2)),
                            'mean_teacher_to_student_kl':float(np.mean(np.sum(q*(np.log(q)-logp),axis=1))),
                            'greedy_actions':greedy.tolist(),'action_probabilities':p.tolist()})
            del model,tokenizer,batch,logits
            gc.collect()
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    summary={'grid':grid,'results':results,'interpretation':'OLS behavioral gains on29fixed unsaturated inputs. Ideal greedy coefficients(intercept,command,error)=(0,1,-1); label-smoothing target expected gains=(0,.9,-.9). These are local behavioral response coefficients, not identifiable raw transformer weights. All final arms and warm starts retained; no training or checkpoint selection.',
             'source_sha256':sha256(Path(__file__))}
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f'Probed {len(results)} checkpoints on {len(grid)} fixed inputs; saved {args.output}')


if __name__=='__main__': main()
