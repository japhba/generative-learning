"""Re-score PROC-REC extra step-identification sets and verification chains: strict study parser vs lenient (bare leading answer)."""
import json, glob, re, collections
B = '/var/tmp/jbauer/generative_task_search_20260910/proc_rec'
CHOICE = re.compile(r'(?:answer|答え|答案)[ \t]*[*_]*[ \t]*[:：][ \t]*[*_(（]*[ \t]*([A-DＡ-Ｄ])', re.I)
BARE = re.compile(r'^[\s*_(（]*([A-DＡ-Ｄ])(?:[\s.)）:：。、]|$)')
YESNO = re.compile(r'(?:answer|答え|答案)[ \t]*[*_]*[ \t]*[:：][ \t]*[*_“"「]*[ \t]*(yes|no|はい|いいえ|是|否)', re.I)
BAREYN = re.compile(r'^[\s*_“"「]*(yes|no|はい|いいえ|是|否)', re.I)
T = str.maketrans('ＡＢＣＤ', 'ABCD'); YES = {'yes', 'はい', '是'}
PEER = {'olmo3_7b': 'phi4_mini', 'phi4_mini': 'olmo3_7b', 'gswallow_9b': 'abeja_qwen7b', 'abeja_qwen7b': 'gswallow_9b', 'glm4_9b': 'internlm3_8b', 'internlm3_8b': 'glm4_9b'}
acc = collections.defaultdict(list)
for setname in ['proc_main', 'proc_main2']:
    items = json.load(open(f'{B}/{setname}/items.json'))
    for f in glob.glob(f'{B}/{setname}/probes/*/*.json'):
        p = json.load(open(f)); c = p['condition']
        if not c.startswith('src:'): continue
        _, s, k = c.split(':'); r = p['reader']
        if s not in (r, PEER[r]): continue
        who = 'own' if s == r else 'peer'; kind = p['kind']; t = p['text']
        if kind.startswith('set'):
            a, j = map(int, re.findall(r'\d+', kind)); ans = items[p['world']]['sets'][a][j]['answer']
            m = CHOICE.findall(t); st = m[-1].translate(T).upper() if m else ''; b = BARE.match(t); le = st or (b.group(1).translate(T).upper() if b else '')
            acc[(r, who, 'step', 'strict')].append(st == ans); acc[(r, who, 'step', 'lenient')].append(le == ans)
        else:
            ans = items[p['world']]['chains'][int(kind[6:])]['correct']
            m = YESNO.findall(t); st = m[-1].lower() if m else ''; b = BAREYN.match(t); le = st or (b.group(1).lower() if b else '')
            acc[(r, who, 'verify', 'strict')].append(st != '' and (st in YES) == ans); acc[(r, who, 'verify', 'lenient')].append(le != '' and (le in YES) == ans)
            acc[(r, who, 'verify', 'parsed')].append(le != '')
pc = lambda key: 100 * sum(acc[key]) / len(acc[key])
for r in PEER:
    print(f"{r:13s} step id strict own {pc((r,'own','step','strict')):5.1f} peer {pc((r,'peer','step','strict')):5.1f} | lenient own {pc((r,'own','step','lenient')):5.1f} peer {pc((r,'peer','step','lenient')):5.1f} || verify strict {pc((r,'own','verify','strict')):5.1f}/{pc((r,'peer','verify','strict')):5.1f} lenient {pc((r,'own','verify','lenient')):5.1f}/{pc((r,'peer','verify','lenient')):5.1f} parsed {pc((r,'own','verify','parsed')):5.1f}%")
