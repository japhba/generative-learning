"""PROC-LAYOUT, exploratory: where does a deployment solution first go wrong? For each test solution, the first
intermediate value of the procedure that does not appear in it, by step operation, after an example in the reader's
own layout (OO, PO) vs the peer's layout (OP, PP). Reads the stored evaluations; no inference."""
import collections, glob, json, re

B = '/var/tmp/jbauer/generative_task_search_20260910'
OWN_LAYOUT = {'OO', 'PO'}
READERS = ['olmo3_7b', 'glm4_9b', 'phi4_mini', 'internlm3_8b', 'abeja_qwen7b', 'gswallow_9b']


def has(text, v): return re.search(rf'(?<![\d,]){v}(?![\d,])', text.replace(',', '').replace('，', '')) is not None


def load():
    rows = []
    for study in ['proc_main', 'proc_main2']:
        worlds = {f.split('/')[-1][:-5]: json.load(open(f)) for f in glob.glob(f'{B}/{study}/worlds/*.json')}
        for f in glob.glob(f'{B}/proc_layout/evaluations/{study}/*/*.json'):
            d = json.load(open(f)); w = worlds[d['world']]; it = w['tests'][d['index']]
            first = next((j for j, v in enumerate(it['values']) if not has(d['text'], v)), None)
            op = 'none' if first is None else w['procedure'][first]['op']
            rows.append(dict(reader=d['reader'], layout='own' if d['cell'] in OWN_LAYOUT else 'peer', op=op))
    return rows


def table(rows, label):
    n = collections.Counter(r['layout'] for r in rows); c = collections.Counter((r['layout'], r['op']) for r in rows)
    print(f'\n{label}: % of solutions whose first missing value is at a step of this op (none = every value shown)')
    for op in ['none', 'swapends', 'sortdesc', 'sortasc', 'rev', 'add', 'sub', 'mul']:
        o, p = 100 * c[('own', op)] / n['own'], 100 * c[('peer', op)] / n['peer']
        print(f'  {op:9s} own layout {o:5.1f}  peer layout {p:5.1f}  peer - own {p - o:+5.1f}')


if __name__ == '__main__':
    rows = load(); table(rows, 'all readers')
    for r in READERS: table([x for x in rows if x['reader'] == r], r)
