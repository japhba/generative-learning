"""Arm 2 statistics: own minus foreign accuracy when readers reason at test, after 1 or 6 matched practice solutions."""
import json

import numpy as np

import mlayout3_search as M3
from mcot_search import CONDITIONS,COUNTS,DIRECTORY,NAMES,TESTS,path_of
from mlayout_analyze import SHORT,bootstrap,sign_flip
from mlayout_search import final_answer,read,write
from mlayout_tasks import correct


def main():
    assert (DIRECTORY/'complete.json').exists();matched=read(M3.DIRECTORY/'matched.json');inc=[n for n in matched if matched[n]['included']];M=len(NAMES)
    worlds={n:read(M3.DIRECTORY/'worlds'/f'{n}.json') for n in inc}
    acc=lambda n,r,c:np.mean([correct(final_answer(read(path_of(r,n,c,i))['text']),worlds[n]['tests'][i]) for i in range(TESTS)])
    out={'episodes':len(inc)}
    for k in COUNTS:
        F={n:np.array([[acc(n,r,f'full:{s}:{k}') for s in NAMES] for r in NAMES]) for n in inc}
        S=[np.mean([F[n][a,a]-np.mean([F[n][a,b] for b in range(M) if b!=a]) for a in range(M)]) for n in inc]
        out[f'N{k}']={'S':float(np.mean(S)),'S_ci':bootstrap(S),'p_one_sided':sign_flip(S),'pooled':np.mean([F[n] for n in inc],0).tolist(),
                      'per_reader':{r:[float(np.mean([F[n][a,a]-np.mean([F[n][a,b] for b in range(M) if b!=a]) for n in inc])),*bootstrap([F[n][a,a]-np.mean([F[n][a,b] for b in range(M) if b!=a]) for n in inc])] for a,r in enumerate(NAMES)},
                      'answers':{r:float(np.mean([acc(n,r,f'answers:{k}') for n in inc])) for r in NAMES}}
    out['initial']={r:float(np.mean([acc(n,r,'initial') for n in inc])) for r in NAMES};write(DIRECTORY/'analysis.json',out)
    print('episodes',len(inc),'| initial (reasoning, no practice):',{SHORT[r]:round(100*v) for r,v in out['initial'].items()})
    for k in COUNTS:
        o=out[f'N{k}'];print(f"N={k}: S = {100*o['S']:+.1f} pp [{100*o['S_ci'][0]:+.1f}, {100*o['S_ci'][1]:+.1f}] p={o['p_one_sided']:.3f} | per reader",{SHORT[r]:round(100*v[0],1) for r,v in o['per_reader'].items()})
        for a,r in enumerate(NAMES): print('   %-14s'%SHORT[r]+' '.join(('[%3.0f]' if b==a else ' %3.0f ')%(100*o['pooled'][a][b]) for b in range(M))+'  | answers-only practice %3.0f'%(100*o['answers'][r]))


if __name__=='__main__': main()
