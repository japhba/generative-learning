"""Audit the completed factorial cohort and estimate paired source preferences."""
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from specificity_experiment import DIRECTORY,MODELS,SOURCES,COUNTS,correct,hashes
from specificity_api import GEMMA
from task_search_api import digest
from task_search_data import seed_for
from task_search_prompts import answer_prompt

OUT=Path(__file__).resolve().parents[1]/'reports'
LABELS={'factor_k0':'Factorization · K = 0','factor_k4':'Factorization · K = 4','bezout_k0':'Bézout · K = 0'}
NAMES={'qwen':'Qwen','gemma':'Gemma','teacher':'Teacher','initial':'Initial only'}


def estimate(values,draws):
    v=np.asarray(values);samples=v[draws].mean(axis=1)
    return {'mean':float(v.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'episode_values':v.tolist()}


def signflip(counts):
    counts=np.asarray(counts,dtype=int)
    signs=2*((np.arange(2**len(counts))[:,None]>>np.arange(len(counts)))&1)-1
    return float(np.mean(signs@counts>=counts.sum()))


def extract_last_array(text):
    arrays=[];decoder=json.JSONDecoder()
    for index,character in enumerate(text):
        if character!='[': continue
        try: value,length=decoder.raw_decode(text[index:])
        except json.JSONDecodeError: continue
        if isinstance(value,list): arrays.append(text[index:index+length])
    return arrays[-1] if arrays else ''


def analysis():
    design=json.loads((DIRECTORY/'design.json').read_text());assert design['source_sha256']==hashes()
    for stage in ['generate','evaluate']:
        for model in MODELS: assert (DIRECTORY/f'{stage}_{model}_complete.json').exists(),'Finish the whole cohort before viewing accuracy.'
    worlds={r['name']:json.loads((DIRECTORY/'worlds'/r['name']).read_text()) for r in design['worlds']}
    assert all(digest(worlds[r['name']])==r['sha256'] for r in design['worlds'])
    raw_directories={}
    for model in MODELS:
        meta=json.loads((DIRECTORY/f'{model}_inference.json').read_text())
        raw_directories[model]=(DIRECTORY.parent if model=='qwen' else DIRECTORY)/'api'/meta['engine_fingerprint']
    statuses=Counter();providers=Counter();unique=set();cost=0.;tokens=Counter();audit_count=0
    generations={};scores={};sensitive={};quality=[]
    def audit(record,prompt,seed,temperature):
        nonlocal audit_count,cost
        model=record['reader'];call=record['call'];key=call['key'];raw=json.loads((raw_directories[model]/f'{key}.json').read_text());request=raw['request'];response=raw['response']
        expected={**({'model':'qwen35-9b','chat_template_kwargs':{'enable_thinking':False}} if model=='qwen' else GEMMA),'messages':[{'role':'user','content':prompt}],'temperature':temperature,'top_p':.9 if temperature else 1.,'max_tokens':256,'seed':seed}
        assert request==expected and digest(request)==key
        choice=response['choices'][0];message=choice['message'];assert choice['finish_reason'] in {'stop','length'}
        assert call['status']==('ok' if choice['finish_reason']=='stop' else 'truncated')
        assert record['answer']==message['content'].strip()==call['value']['answer']
        assert not message['reasoning']
        if model=='gemma':
            assert response['provider']=='DekaLLM' and response['model']==GEMMA['model']
            assert response['usage']['completion_tokens_details']['reasoning_tokens']==0
        statuses[(model,'generation' if temperature else 'evaluation',call['status'])]+=1;audit_count+=1
        if (model,key) not in unique:
            unique.add((model,key));tokens[(model,'prompt')]+=response['usage']['prompt_tokens'];tokens[(model,'completion')]+=response['usage']['completion_tokens']
            if model=='gemma': cost+=response['usage']['cost'];providers[response['provider']]+=1
    for name,world in tqdm(worlds.items(),desc='Audit all saved generation requests'):
        stem=Path(name).stem
        for model in MODELS:
            records=[]
            for index,entry in enumerate(world['practice']):
                record=json.loads((DIRECTORY/'generations'/model/f'{stem}__{index:02}.json').read_text())
                assert record['world']==name and record['index']==index and record['source']==model
                audit(record,answer_prompt(world,[],entry),seed_for(world['spec'],f"{world['seed']}|specificity|generation|k{len(world['support'])}",index),.7)
                records.append(record)
            generations[(name,model)]=records
    for name,world in tqdm(worlds.items(),desc='Audit all saved final requests'):
        stem=Path(name).stem
        for model in MODELS:
            for source,n in [('initial',0)]+[(s,n) for n in COUNTS[1:] for s in SOURCES]:
                if source=='initial': examples=[]
                elif source=='teacher': examples=world['practice'][:n]
                else: examples=[{'question':e['question'],'answer':g['answer']} for e,g in zip(world['practice'][:n],generations[(name,source)][:n])]
                for index,entry in enumerate(world['tests']):
                    record=json.loads((DIRECTORY/'evaluations'/model/f'{stem}__{source}__n{n}__{index:02}.json').read_text())
                    assert record['world']==name and record['index']==index and record['source']==source and record['n']==n
                    audit(record,answer_prompt(world,examples,entry),world['seed']+index,0.)
                    key=(world['setting'],model,source,n)
                    if key not in scores: scores[key]=np.zeros((16,12),dtype=int);sensitive[key]=np.zeros((16,12),dtype=int)
                    scores[key][world['episode'],index]=int(record['call']['status']=='ok' and correct(record['answer'],entry))
                    sensitive[key][world['episode'],index]=int(record['call']['status']=='ok' and correct(extract_last_array(record['answer']),entry))
    for setting in design['settings']:
        selected=[(name,w) for name,w in worlds.items() if w['setting']==setting]
        for source in SOURCES:
            values=[];counts=Counter();characters=[];completion_tokens=[]
            for name,world in selected:
                for index,entry in enumerate(world['practice']):
                    answer=entry['answer'] if source=='teacher' else generations[(name,source)][index]['answer']
                    valid=correct(answer,entry);values.append(int(valid));characters.append(len(answer))
                    if source!='teacher': completion_tokens.append(generations[(name,source)][index]['call']['usage']['completion_tokens'])
                    try: parsed=json.loads(answer)
                    except json.JSONDecodeError: counts['malformed_json']+=1;continue
                    if isinstance(parsed,list) and all(type(x) is int for x in parsed):
                        counts['integer_array']+=1
                        if 'operation' in entry['metadata']:
                            counts['two_or_more_ones']+=parsed.count(1)>=2
                            counts['valid_one_one']+=valid and parsed.count(1)==1
                            counts['valid_no_ones']+=valid and parsed.count(1)==0
                        elif len(parsed)==2:
                            counts['first_coordinate_nonnegative']+=parsed[0]>=0
            quality.append({'setting':setting,'source':source,'n':len(values),'valid':sum(values),'accuracy':float(np.mean(values)),'mean_characters':float(np.mean(characters)),'mean_completion_tokens':float(np.mean(completion_tokens)) if source!='teacher' else None,'strategies':dict(counts)})
    draws=np.random.default_rng(520912).integers(0,16,size=(20000,16));cells=[];endpoints=[]
    for setting in design['settings']:
        for model in MODELS:
            for source,n in [('initial',0)]+[(s,n) for n in COUNTS[1:] for s in SOURCES]:
                array=scores[(setting,model,source,n)]
                cells.append({'setting':setting,'reader':model,'source':source,'n':n,'correct':int(array.sum()),'total':int(array.size),'accuracy':estimate(array.mean(axis=1),draws),'last_array_accuracy':float(sensitive[(setting,model,source,n)].mean())})
        q=scores[(setting,'qwen','qwen',32)]-scores[(setting,'qwen','gemma',32)]
        g=scores[(setting,'gemma','gemma',32)]-scores[(setting,'gemma','qwen',32)]
        joint_counts=(q+g).sum(axis=1);h=joint_counts/24
        result={'setting':setting,'own_preference':{},'own_minus_initial':{},'own_minus_teacher':{},'interaction_half':estimate(h,draws),'p_one_sided':signflip(joint_counts)}
        for model,difference in [('qwen',q),('gemma',g)]:
            result['own_preference'][model]=estimate(difference.mean(axis=1),draws)
            for other in ['initial','teacher']:
                contrast=scores[(setting,model,model,32)]-scores[(setting,model,other,0 if other=='initial' else 32)]
                result['own_minus_'+other][model]=estimate(contrast.mean(axis=1),draws)
        result['identical_answer_fraction']=float(np.mean([a['answer']==b['answer'] for name,w in worlds.items() if w['setting']==setting for a,b in zip(generations[(name,'qwen')],generations[(name,'gemma')])]))
        endpoints.append(result)
    last=0.
    for rank,result in enumerate(sorted(endpoints,key=lambda r:r['p_one_sided'])):
        last=max(last,min(1.,(3-rank)*result['p_one_sided']));result['p_holm']=last
    assert audit_count==14592
    result={'design':design,'cells':cells,'endpoints':endpoints,'generation_quality':quality,'audit':{'request_uses':audit_count,'unique_requests':len(unique),'providers':dict(providers),'cost_usd':cost,'tokens':{'|'.join(k):v for k,v in tokens.items()},'statuses':{'|'.join(k):v for k,v in statuses.items()},'prompt_reconstruction':'every full request exactly matched; only saved rehearsal answers differ across sources','context_overflow':0},'format_sensitivity':'Descriptive last complete JSON array, final truncations still incorrect; primary scores unchanged.'}
    (DIRECTORY/'analysis.json').write_text(json.dumps(result,indent=2))
    (OUT/'task-search-specificity-results.json').write_text(json.dumps(result,indent=2))
    plot(result)
    print(json.dumps({'endpoints':[{k:v for k,v in r.items() if k not in ['own_minus_initial','own_minus_teacher']} for r in endpoints],'audit':result['audit']},indent=2))


def plot(result):
    cells={(r['setting'],r['reader'],r['source'],r['n']):r for r in result['cells']}
    for setting in result['design']['settings']:
        fig,axes=plt.subplots(1,2,sharey=True,layout='constrained')
        for model,ax in zip(MODELS,axes):
            baseline=cells[(setting,model,'initial',0)]['accuracy']
            for source in SOURCES:
                values=[baseline]+[cells[(setting,model,source,n)]['accuracy'] for n in COUNTS[1:]]
                mean=np.array([v['mean'] for v in values])*100;ci=np.array([v['ci95'] for v in values])*100
                ax.plot(range(4),mean,marker='o',label=NAMES[source]+(' (own)' if source==model else ''))
                ax.fill_between(range(4),ci[:,0],ci[:,1],alpha=.15)
            ax.set_xticks(range(4),COUNTS);ax.set_xlabel('Rehearsal pairs N');ax.set_title(NAMES[model]+' reads');ax.set_ylim(0,100);ax.legend()
        axes[0].set_ylabel('Correct answers (%)');fig.suptitle(LABELS[setting])
        fig.savefig(OUT/f'task-search-specificity-{setting}-curves.png');plt.close(fig)
        matrix=np.array([[cells[(setting,r,s,32)]['accuracy']['mean'] for s in SOURCES] for r in MODELS])*100
        fig,ax=plt.subplots(layout='constrained');heat=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        ax.set_xticks(range(3),[NAMES[s] for s in SOURCES]);ax.set_yticks(range(2),[NAMES[r] for r in MODELS]);ax.set_xlabel('Source of 32 rehearsal answers');ax.set_ylabel('Reader');ax.set_title(LABELS[setting])
        for i,r in enumerate(MODELS):
            for j,s in enumerate(SOURCES): ax.text(j,i,f'{matrix[i,j]:.1f}%'+('\n(own)' if r==s else ''),ha='center',va='center',color='white' if matrix[i,j]>50 else 'black')
        fig.colorbar(heat,ax=ax,label='Correct answers (%)');fig.savefig(OUT/f'task-search-specificity-{setting}-matrix.png');plt.close(fig)
    fig,ax=plt.subplots(layout='constrained')
    for i,endpoint in enumerate(result['endpoints']):
        for j,model in enumerate(MODELS):
            stats=endpoint['own_preference'][model];mean=100*stats['mean'];ci=np.array(stats['ci95'])*100
            ax.errorbar(mean,3*i+j,xerr=np.array([[mean-ci[0]],[ci[1]-mean]]),fmt='o',color=f'C{j}',label=NAMES[model] if i==0 else None)
    ax.set_yticks([3*i+j for i in range(3) for j in range(2)],[LABELS[e['setting']]+' · '+NAMES[m] for e in result['endpoints'] for m in MODELS]);ax.invert_yaxis();ax.axvline(0,color='black',linestyle='--');ax.set_xlabel('Own − other model’s source (percentage points)');ax.set_title('Does each reader prefer its own examples?');ax.legend()
    fig.savefig(OUT/'task-search-specificity-preferences.png');plt.close(fig)


if __name__=='__main__': analysis()
