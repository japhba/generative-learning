"""PROC analysis: deployment vs recognition of a demonstrated procedure by source relation (own / same-language peer /
other language), with reader and source main effects removed."""
import numpy as np

from edge_analyze import sign_flip_two
from lang_analyze import language_of
SHORT={'olmo3_7b':'OLMo-3-7B (en)','phi4_mini':'Phi-4-mini (en)','gswallow_9b':'Gemma2-Swallow-9B (ja)','abeja_qwen7b':'ABEJA-Qwen2.5-7B (ja)','glm4_9b':'GLM-4-9B (zh)','internlm3_8b':'InternLM3-8B (zh)'}
from proc_search import COUNTS,DIRECTORY,LANGUAGE,NAMES,POOL,SAMPLES,TESTS,chosen,correct,eval_path,gen_path,probe_path,read,shows_steps,write

def relation(r,s): return 'own' if r==s else 'same language' if LANGUAGE[r]==LANGUAGE[s] else 'other language'


M=len(NAMES);REL=np.array([[relation(r,s) for s in NAMES] for r in NAMES]);RELATIONS=['own','same language','other language']


def boot(v,seed=20261007): v=np.asarray(v,float);i=np.random.default_rng(seed).integers(0,len(v),(20000,len(v)));return [float(x) for x in np.percentile(v[i].mean(1),[2.5,97.5])]
def summary(v): v=np.asarray(v,float);return {'mean':float(v.mean()),'ci':boot(v),'p_two_sided':sign_flip_two(v,100000),'n':len(v)}
def fmt(s): return '%+.1f pp [%+.1f, %+.1f] p=%.3f (n=%d)'%(100*s['mean'],100*s['ci'][0],100*s['ci'][1],s['p_two_sided'],s['n'])
def dc(A): return A-A.mean(1,keepdims=True)-A.mean(0,keepdims=True)+A.mean()
def contrast(A,a,b): D=dc(A);return float(D[REL==a].mean()-D[REL==b].mean())


def main():
    design=read(DIRECTORY/'design.json');done=read(DIRECTORY/'complete.json');worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in design['worlds']}
    print('writers (told the procedure): correct and showing every step / language')
    for m in NAMES:
        for L in sorted({w['length'] for w in worlds.values()}):
            s=[(correct(x['text'],worlds[n]['pool'][i]) and shows_steps(x['text'],worlds[n]['pool'][i]),correct(x['text'],worlds[n]['pool'][i]),language_of(x['text'])==LANGUAGE[m]) for n in worlds if worlds[n]['length']==L for i in range(POOL) for x in read(gen_path(m,n,i))['samples']]
            print('  %-22s L=%d  correct+steps %3.0f%%  correct %3.0f%%  own language %3.0f%%'%(SHORT[m],L,*(100*np.mean([x[k] for x in s]) for k in range(3))))
    out={}
    for L in sorted({w['length'] for w in worlds.values()})+['all']:
        eps=[n for n in done['usable'] if L=='all' or worlds[n]['length']==L]
        if not eps: print('L',L,'no usable episodes');continue
        dep=lambda r,n,c:float(np.mean([correct(read(eval_path(r,n,c,i))['text'],worlds[n]['tests'][i]) for i in range(TESTS)]))
        rec=lambda r,n,c:float(np.mean([chosen(read(probe_path(r,n,c,j))['text'])==worlds[n]['probes'][j]['answer'] for j in range(worlds[n]['length'])]))
        print(f'\n===== L={L}: {len(eps)} usable episodes')
        base={c:{r:np.mean([dep(r,n,c) for n in eps]) for r in NAMES} for c in ['answers','stated']};recb={r:np.mean([rec(r,n,'answers') for n in eps]) for r in NAMES}
        print('  deployment: answers-only',{SHORT[r][:10]:round(100*v) for r,v in base['answers'].items()},'| procedure stated',{SHORT[r][:10]:round(100*v) for r,v in base['stated'].items()})
        print('  recognition with answers only',{SHORT[r][:10]:round(100*v) for r,v in recb.items()})
        for k in COUNTS:
            Dp={n:np.array([[dep(r,n,f'src:{s}:{k}') for s in NAMES] for r in NAMES]) for n in eps};Rc={n:np.array([[rec(r,n,f'src:{s}:{k}') for s in NAMES] for r in NAMES]) for n in eps}
            res={'deploy_language':summary([contrast(Dp[n],'same language','other language') for n in eps]),'recog_language':summary([contrast(Rc[n],'same language','other language') for n in eps]),
                 'deploy_self':summary([contrast(Dp[n],'own','same language') for n in eps]),'recog_self':summary([contrast(Rc[n],'own','same language') for n in eps]),
                 'deploy_level':{x:float(np.mean([Dp[n][REL==x].mean() for n in eps])) for x in RELATIONS},'recog_level':{x:float(np.mean([Rc[n][REL==x].mean() for n in eps])) for x in RELATIONS}}
            out[f'L{L}_N{k}']=res
            print(f'  N={k}: deployment',{x:round(100*v) for x,v in res['deploy_level'].items()},'| recognition',{x:round(100*v) for x,v in res['recog_level'].items()})
            print(f'        same-language peer - other language: deployment {fmt(res["deploy_language"])} | recognition {fmt(res["recog_language"])}')
            print(f'        own - same-language peer:            deployment {fmt(res["deploy_self"])} | recognition {fmt(res["recog_self"])}')
    write(DIRECTORY/'analysis.json',out)


if __name__=='__main__': main()
