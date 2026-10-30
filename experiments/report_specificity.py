"""Build the reader/source comparison and full examples for the canonical report."""
from html import escape
import json
from pathlib import Path
from analyze_specificity import LABELS,NAMES
from specificity_experiment import DIRECTORY,MODELS,SOURCES,correct
from task_search_prompts import answer_prompt

OUT=Path(__file__).resolve().parents[1]/'reports'


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def interval(result):
    low,high=result['ci95'];return f'{100*result["mean"]:+.1f} [{100*low:+.1f}, {100*high:+.1f}]'


def answer_cell(answer,entry,finished=True):
    status='Correct' if finished and correct(answer,entry) else 'Incorrect'
    return '<pre>'+escape(answer)+'</pre><span class="small">'+status+(' · output limit reached' if not finished else '')+'</span>'


def main():
    result=json.loads((DIRECTORY/'analysis.json').read_text());cells={(r['setting'],r['reader'],r['source'],r['n']):r for r in result['cells']}
    assert all(not all(e['own_preference'][m]['mean']>0 for m in MODELS) for e in result['endpoints'])
    lead='No tested setting shows both models preferring their own samples at N=32. The clearest benefit transfers from Qwen to Gemma.'
    section='<section id="specificity"><h2>Does each model benefit especially from its own examples?</h2><p class="lead"><strong>'+lead+'</strong></p>'
    section+='<p>Qwen3.5-9B and Gemma 4 26B A4B each read both models’ saved answers, plus exact teacher answers, to the <strong>same rehearsal questions</strong>. Only those answer strings change across sources within a reader. No author labels, correctness feedback, filtering, or weight updates.</p>'
    section+='<p><strong>Factorization:</strong> return three positive integers with the requested product; every prompt forbids two or more 1s. <strong>Bézout:</strong> given coprime a,b ∈ [3,19], return integers [x,y] with ax+by=1. K is the initial exact-example count.</p>'
    rows=[]
    for setting in result['design']['settings']:
        for reader in MODELS:
            row=[LABELS[setting]+'<br>'+NAMES[reader]+' reads']
            for source,n in [('initial',0),('qwen',32),('gemma',32),('teacher',32)]:
                c=cells[(setting,reader,source,n)];value=f'{100*c["accuracy"]["mean"]:.1f}% <span class="small">({c["correct"]}/192)</span>'
                row.append('<strong>'+value+'</strong>' if reader==source else value)
            rows.append(row)
    section+=table(['Reader / setting','Initial only<br>N = 0','Qwen source<br>N = 32','Gemma source<br>N = 32','Teacher source<br>N = 32'],rows)
    section+='<p class="small">Fresh fixed cohort: 16 episodes × 12 tests per setting/reader. Bold marks own source. All 32 rehearsal slots, including truncated prefixes, are replayed. Thinking disabled; generation temperature 0.7, evaluation 0; 256-token output cap in both roles.</p>'
    section+=table(['Own − other source<br>percentage points','Qwen reads','Gemma reads','Mean preference H','Holm p'],[[LABELS[e['setting']],interval(e['own_preference']['qwen']),interval(e['own_preference']['gemma']),interval(e['interaction_half']),f'{e["p_holm"]:.3f}'] for e in result['endpoints']])
    section+='<p class="small">Brackets: 95% paired episode-bootstrap intervals. H is half the reader × source interaction; a universal source main effect cancels. N=32 was fixed in advance; three tests are Holm-corrected. Sign-flip inference assumes symmetric episode contrasts. The positive H at K=4 comes entirely from Qwen: Gemma is near the floor and has no own-source preference.</p>'
    section+='<p><strong>What transfers:</strong> 108/115 correct Gemma answers with Qwen rehearsal use one 1, which remains valid. <strong>Limit:</strong> Gemma often produces visible working despite the disabled reasoning channel; 150/192 initial-only factorization responses hit the 256-token cap. Source correctness, length and truncation are not matched.</p>'
    section+='<p><strong>Mechanistic control:</strong> a separate toy has two equally correct generators and two restricted recognizers. One shared squared-error sleep update learns a native latent branch but averages conflicting branches from the other source. Giving both readers both features removes the asymptotic specificity. This illustrates the hypothesis; it does not identify the LLM mechanism.</p>'
    section+='<details><summary>All learning curves, source diagnostics, and 10 rehearsal + 10 eval examples per setting</summary><p>Settings were selected from earlier results before this fresh cohort. They cover two task families. All N=0,2,8,32 are shown; no power-law exponent is fitted.</p><img src="task-search-specificity-preferences.png" alt="Own-source preferences for both readers">'
    for setting in result['design']['settings']:
        section+='<h2>'+LABELS[setting]+'</h2><div class="columns"><img src="task-search-specificity-'+setting+'-matrix.png" alt="Reader by source accuracy matrix"><img src="task-search-specificity-'+setting+'-curves.png" alt="Learning curves for both readers and all sources"></div>'
        e=next(e for e in result['endpoints'] if e['setting']==setting)
        section+='<p>Own − initial: Qwen '+interval(e['own_minus_initial']['qwen'])+'; Gemma '+interval(e['own_minus_initial']['gemma'])+'. Own − teacher: Qwen '+interval(e['own_minus_teacher']['qwen'])+'; Gemma '+interval(e['own_minus_teacher']['gemma'])+'. Identical Qwen/Gemma rehearsal strings: '+f'{e["identical_answer_fraction"]:.1%}'+'.</p>'
        quality=[q for q in result['generation_quality'] if q['setting']==setting]
        section+=table(['Rehearsal source','Correct / total','Mean characters','Format / strategy counts'],[[NAMES[q['source']],f'{q["valid"]}/{q["n"]} ({q["accuracy"]:.1%})',f'{q["mean_characters"]:.1f}',escape(json.dumps(q['strategies']))] for q in quality])
        section+='<p class="small">Validity and representation were checked only after inference. They never affected which samples the readers saw. Correctness and length are not matched between generated sources.</p>'
        name=f'{setting}__00.json';stem=Path(name).stem;w=json.loads((DIRECTORY/'worlds'/name).read_text())
        own={m:[json.loads((DIRECTORY/'generations'/m/f'{stem}__{i:02}.json').read_text()) for i in range(32)] for m in MODELS}
        section+='<details><summary>First episode: first 10 rehearsals, complete saved text</summary><p>Original order; all 32 were replayed. Correctness annotations are for this report only.</p>'
        section+=table(['Assigned question','Teacher','Qwen generation','Gemma generation'],[[escape(w['practice'][i]['question']),answer_cell(w['practice'][i]['answer'],w['practice'][i])]+[answer_cell(own[m][i]['answer'],w['practice'][i],own[m][i]['call']['status']=='ok') for m in MODELS] for i in range(10)])+'</details>'
        section+='<details><summary>First episode: first 10 tests, all final responses side by side</summary>'
        for reader in MODELS:
            rows=[]
            for index,entry in enumerate(w['tests'][:10]):
                row=[escape(entry['question'])]
                for source,n in [('initial',0),('qwen',32),('gemma',32),('teacher',32)]:
                    r=json.loads((DIRECTORY/'evaluations'/reader/f'{stem}__{source}__n{n}__{index:02}.json').read_text());row.append(answer_cell(r['answer'],entry,r['call']['status']=='ok'))
                rows.append(row)
            section+='<h2>'+NAMES[reader]+' reads</h2>'+table(['Held-out question','Initial only','Qwen source','Gemma source','Teacher source'],rows)
        section+='</details><details><summary>Exact generation and final prompt bodies</summary><p>Both readers receive these same user-message bodies. Provider chat templates and tokenization differ. Generation is a separate offline call with no additional examples.</p>'
        prompt_records=[('generation',answer_prompt(w,[],w['practice'][0]))]
        for source in SOURCES:
            examples=w['practice'] if source=='teacher' else [{'question':e['question'],'answer':g['answer']} for e,g in zip(w['practice'],own[source])]
            prompt_records.append((source,answer_prompt(w,examples,w['tests'][0])))
        for label,prompt in prompt_records:
            filename=f'task-search-specificity-{setting}-{label}-prompt.txt';(OUT/filename).write_text(prompt)
            title='Offline generation prompt' if label=='generation' else NAMES[label]+' final prompt'
            section+='<details><summary>'+title+'</summary><pre>'+escape(prompt)+'</pre></details><p><a href="'+filename+'">Full '+label+' prompt text</a></p>'
        section+='</details>'
    audit=result['audit'];section+='<h2>Checks and limits</h2><p>'+f'{audit["request_uses"]:,} request uses reconstructed exactly; no context overflow. Experimental Gemma calls billed USD {audit["cost_usd"]:.3f}. All came from DekaLLM with zero reported reasoning tokens.'+'</p><pre>'+escape(json.dumps(audit,indent=2))+'</pre>'
    section+='<p>The remote serving revision and kernels are not pinned. A source interaction could reflect correctness sensitivity, verbosity, formatting, or task difficulty as well as representation compatibility. Descriptive last-complete-JSON-array sensitivity scores are in <a href="task-search-specificity-results.json">the full results</a>.</p></details>'
    section+='<details><summary>A one-weight sleep toy with symmetric specificity</summary><p>Input (m,c), m&gt;0 and c∈{0,1}. Any z with |z|=m is correct. Source A returns m; source B returns (−1)<sup>c</sup>m. Both are 100% correct on the same inputs. Reader A has φ=m; reader B has φ=(−1)<sup>c</sup>m. Each predicts ẑ=wφ, starting at w=0.25.</p><pre>Same loss: L = ½((wφ − z)/m)²\nSame update: w ← w + η[(φ/m)(z/m) − w]\nOwn source: (φ/m)(z/m) = 1\nOther source: (φ/m)(z/m) = ±1 with mean 0</pre><p>Own examples drive w toward 1. Cross-source labels conflict in the restricted feature space and drive the mean toward 0. No source-specific bonus or correctness judgment appears in the update.</p><img src="task-search-specificity-sleep-toy.png" alt="Equal-quality sources; restricted readers favor their own source; rich readers fit either source"><p>Both features [m,(−1)<sup>c</sup>m] let either reader fit either source. The original feature alignment is assumed, not learned. This is a sleep-phase recognition loss with fixed generators, not a complete wake–sleep model. Native error is (1−w₀)²(1−η)<sup>2N</sup>, checked against 4,096 simulations: exponential relaxation, not a power law.</p><p>The generator knows the latent it sampled, providing a sleep target without external feedback, as in <a href="https://www.cs.toronto.edu/~hinton/absps/ws.pdf">Hinton et al.’s wake–sleep algorithm</a>. <a href="https://aclanthology.org/2023.emnlp-main.968/">Self-ICL</a> replays pseudo-examples in a frozen model; it does not establish this weight-space mechanism or reader × source specificity.</p></details></section>'
    (OUT/'specificity-section.html').write_text(section)
    print('Built specificity section: 30 rehearsal examples and 60 complete paired evaluation rows.')


if __name__=='__main__': main()
