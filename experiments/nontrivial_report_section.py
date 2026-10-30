"""Render complete, prespecified example rows for the constrained follow-up."""
from collections import Counter
from html import escape
import json
from pathlib import Path

from nontrivial_task_runner import solution
from task_search_prompts import answer_prompt


def extra_section(directory,report):
    cells={(r['initial_count'],r['arm'],r['rehearsal_count']):r for r in report['cells']}
    differences={(r['initial_count'],r['rehearsal_count']):r for r in report['paired_differences']}
    rows=[]
    for k in [0,4,16]:
        baseline=cells[k,'self_fixed',0]; teacher=cells[k,'teacher',32]; own=cells[k,'self_fixed',32]; delta=differences[k,32]
        rows.append(f'<tr><td>{k}</td><td>{baseline["accuracy"]:.1%}</td><td>{teacher["accuracy"]:.1%}</td><td>{own["accuracy"]:.1%}</td><td>{100*delta["own_minus_teacher"]:+.1f} [{100*delta["ci95_teacher"][0]:+.1f}, {100*delta["ci95_teacher"][1]:+.1f}]</td><td>{100*delta["own_minus_initial"]:+.1f} [{100*delta["ci95_initial"][0]:+.1f}, {100*delta["ci95_initial"][1]:+.1f}]</td></tr>')
    parts=['<h3>Constrained task: endpoint accuracy at N = 32</h3>', '<table><thead><tr><th>Initial K</th><th>Initial only</th><th>Teacher</th><th>Own</th><th>Own − teacher, points [95% CI]</th><th>Own − initial, points [95% CI]</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>']
    own4=cells[4,'self_fixed',32]; teacher4=cells[4,'teacher',32]
    parts.append(f'<p>At K = 4, N = 32, {round(256*own4["valid_one_unit_fraction"])} of the {own4["correct"]} correct own-context test answers use exactly one 1. The corresponding teacher count is {round(256*teacher4["valid_one_unit_fraction"])} of {teacher4["correct"]}. Arrays containing two or more 1s occur in {round(256*own4["two_ones_fraction"])}/256 own-context answers and {round(256*teacher4["two_ones_fraction"])}/256 teacher-context answers; all are scored incorrect.</p>')
    quality=report['generation_quality']; truncated_generations=sum(Counter(r['statuses'])['truncated'] for r in quality); truncated_calls=Counter(report['verification']['call_statuses'])['truncated']
    forbidden=sum(r['forbidden_identity'] for r in quality)
    parts.append(f'<p class="small">Across 3,072 own rehearsal slots, {sum(r["correct"] for r in quality)} answers satisfy the new rule ({sum(r["valid_one_unit"] for r in quality)} use exactly one 1); {forbidden} answers have the correct product but use the forbidden identity form. In total, {sum(r["two_ones"] for r in quality)} rehearsal arrays contain two or more 1s, regardless of product correctness. No correctness feedback or rejection sampling was used. {truncated_generations} rehearsal calls hit the output limit and retained empty answer slots; {truncated_calls-truncated_generations} final calls were truncated and scored incorrect. Full raw outputs are saved.</p>')
    name='k4__product_decomposition_k3_d3__default__1093732622.json'
    world=json.loads((directory/'worlds'/name).read_text()); result=json.loads((directory/'evaluations'/name).read_text())
    generated=result['generations']['self_fixed']; endpoints={c['arm']:c for c in result['conditions'] if c['n']==32}
    def shown(text,entry):
        valid,identity=solution(text,entry); label='correct' if valid else 'incorrect: forbidden two-1 identity' if identity else 'incorrect'
        return '<pre>'+escape(text)+'</pre><span class="small">'+label+'</span>'
    practice=[]; tests=[]
    for i,entry in enumerate(world['practice'][:10]):
        practice.append(f'<tr><td>Q({entry["metadata"]["target"]})</td><td>{shown(entry["answer"],entry)}</td><td>{shown(generated[i]["example"]["answer"],entry)}</td></tr>')
    for i,entry in enumerate(world['tests'][:10]):
        tests.append(f'<tr><td>Q({entry["metadata"]["target"]})<br>One valid answer: {escape(entry["answer"])}</td><td>{shown(endpoints["teacher"]["predictions"][i]["answer"],entry)}</td><td>{shown(endpoints["self_fixed"]["predictions"][i]["answer"],entry)}</td></tr>')
    parts+=['<details><summary>Ten constrained rehearsal examples and ten paired evaluations</summary><p>Same illustrative episode as above: 1093732622, K = 4; first ten entries in original order. Q(T) is the original product question with the new constraint appended. All answer strings below are complete; correctness annotations were added after inference.</p>', '<table><thead><tr><th>Rehearsal question</th><th>Teacher answer</th><th>Own generation</th></tr></thead><tbody>'+''.join(practice)+'</tbody></table>', '<table><thead><tr><th>Held-out question</th><th>Teacher context, N = 32</th><th>Own context, N = 32</th></tr></thead><tbody>'+''.join(tests)+'</tbody></table>']
    out=Path(__file__).resolve().parents[1]/'reports'
    for arm,examples in [('teacher',world['practice']),('own',[g['example'] for g in generated]),('generation',[])]:
        query=world['practice'][0] if arm=='generation' else world['tests'][0]
        (out/f'task-search-constrained-{arm}-prompt.txt').write_text(answer_prompt(world,examples,query))
    parts.append('<p>Full constrained prompts: <a href="task-search-constrained-generation-prompt.txt">generation</a> · <a href="task-search-constrained-teacher-prompt.txt">teacher evaluation</a> · <a href="task-search-constrained-own-prompt.txt">own evaluation</a>.</p></details>')
    return '\n'.join(parts)
