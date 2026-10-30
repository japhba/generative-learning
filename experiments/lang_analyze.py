"""Pilot analysis: practice uplift by source relation (own / same-language peer / other language), recognition probe, and
application conditional on recognition."""
from collections import defaultdict
import re

import numpy as np

from lang_search import COUNTS,FAMILIES,DIRECTORY,LANGUAGE,NAMES,POOL,TESTS,conditions,correct,eval_path,gen_path,probe_correct,probe_path,read,write

SHORT={'llama31_8b':'Llama-3.1-8B (en)','olmo3_7b':'OLMo-3-7B (en)','swallow_8b':'Swallow-8B (ja)','gswallow_9b':'Gemma2-Swallow-9B (ja)','glm4_9b':'GLM-4-9B (zh)','internlm3_8b':'InternLM3-8B (zh)'}


def language_of(text):
    kana=len(re.findall(r'[぀-ヿ]',text));han=len(re.findall(r'[一-鿿]',text));latin=len(re.findall(r'[A-Za-z]',text))
    return 'ja' if kana>=10 else 'zh' if han>=10 else 'en' if latin>=20 else '?'


def relation(r,s): return 'own' if r==s else 'same language' if LANGUAGE[r]==LANGUAGE[s] else 'other language'


def boot(v,seed=0): v=np.asarray(v,float);i=np.random.default_rng(seed).integers(0,len(v),(10000,len(v)));return np.percentile(v[i].mean(1),[2.5,97.5])


def main():
    design=read(DIRECTORY/'design.json');done=read(DIRECTORY/'complete.json');worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in design['worlds']}
    print('language of attempts (share in the intended language):')
    for m in NAMES:
        langs=[language_of(s['text']) for n in worlds for i in range(POOL) for s in read(gen_path(m,n,i))['samples']]
        acc={f:np.mean([correct(s['text'],worlds[n]['pool'][i]) for n in worlds if worlds[n]['family']==f for i in range(POOL) for s in read(gen_path(m,n,i))['samples']]) for f in FAMILIES}
        print('  %-20s intended %.0f%% %s | attempt accuracy %s'%(SHORT[m],100*np.mean([l==LANGUAGE[m] for l in langs]),dict(zip(*np.unique(langs,return_counts=True))),{f:round(100*v) for f,v in acc.items()}))
    out={}
    for family in FAMILIES:
        eps=[n for n in done['usable'] if worlds[n]['family']==family]
        if not eps: print(family,'no usable episodes');continue
        acc=lambda r,n,c:np.mean([correct(read(eval_path(r,n,c,i))['text'],worlds[n]['tests'][i]) for i in range(TESTS)])
        rec=lambda r,n,c:float(probe_correct(read(probe_path(r,n,c))['text'],worlds[n]))
        print(f'\n===== {family}: {len(eps)} episodes')
        rows={};uplift=defaultdict(list);recog=defaultdict(list);cond=defaultdict(list)
        for r in NAMES:
            base=[acc(r,n,'initial') for n in eps];ans=[acc(r,n,'answers') for n in eps]
            line=f"{SHORT[r]:20s} none {100*np.mean(base):3.0f}  answers {100*np.mean(ans):3.0f}  probe(none) {100*np.mean([rec(r,n,'initial') for n in eps]):3.0f}"
            for k in COUNTS:
                for rel in ['own','same language','other language']:
                    srcs=[s for s in NAMES if relation(r,s)==rel]
                    a=[np.mean([acc(r,n,f'src:{s}:{k}') for s in srcs]) for n in eps];p=[np.mean([rec(r,n,f'src:{s}:{k}') for s in srcs]) for n in eps]
                    uplift[k,rel]+=[x-b for x,b in zip(a,base)];recog[k,rel]+=p
                    for s in srcs:
                        for n in eps: cond[k,rel,rec(r,n,f'src:{s}:{k}')].append(acc(r,n,f'src:{s}:{k}'))
                    line+=f" | N={k} {rel[:5]} {100*np.mean(a):3.0f} (probe {100*np.mean(p):3.0f})"
            print(line)
        print('mean over readers: uplift over no practice, recognition, and test accuracy given the probe was right / wrong')
        for k in COUNTS:
            for rel in ['own','same language','other language']:
                u=uplift[k,rel];lo,hi=boot(u)
                print(f"  N={k} {rel:15s} uplift {100*np.mean(u):+5.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}]  recognised {100*np.mean(recog[k,rel]):3.0f}%  acc|recognised {100*np.mean(cond[k,rel,1.0]) if cond[k,rel,1.0] else float('nan'):3.0f}% (n={len(cond[k,rel,1.0])})  acc|not {100*np.mean(cond[k,rel,0.0]) if cond[k,rel,0.0] else float('nan'):3.0f}% (n={len(cond[k,rel,0.0])})")
        out[family]={f'{k}|{rel}':{'uplift':float(np.mean(uplift[k,rel])),'recognised':float(np.mean(recog[k,rel]))} for k in COUNTS for rel in ['own','same language','other language']}
    write(DIRECTORY/'analysis.json',out)


if __name__=='__main__': main()
