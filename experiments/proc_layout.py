"""PROC-LAYOUT (PROCLAYOUT_PROTOCOL.md): does the own advantage in deploying a demonstrated procedure follow the layout
or the authorship (wording) of the worked example?

Stages: rewrite (OpenRouter, cached) -> check (manipulation checks, full spot checks) -> evaluate (six vLLM readers)
-> analyze.
"""
import argparse
import asyncio
from itertools import product
import json
import os
from pathlib import Path
import random
import re
import time

from dotenv import load_dotenv
import httpx
import numpy as np
from tqdm.auto import tqdm

import proc_search as P
from mlayout_search import now, read, write
from task_search_api import ROOT, digest

BASE = ROOT.resolve(); DIRECTORY = BASE / 'proc_layout'; STUDIES = ['proc_main', 'proc_main2']
PAIRS = [('olmo3_7b', 'phi4_mini'), ('gswallow_9b', 'abeja_qwen7b'), ('glm4_9b', 'internlm3_8b')]
PEER = {a: b for a, b in PAIRS} | {b: a for a, b in PAIRS}
STRONG = ['olmo3_7b', 'phi4_mini', 'glm4_9b', 'internlm3_8b']  # recognise every step (PROC-REC)
REWRITER = {'model': 'anthropic/claude-sonnet-5.5', 'provider': 'anthropic'}; BUDGET = 8.0
CELLS = ['OO', 'OP', 'PO', 'PP']  # (wording, layout) relative to the reader: O = own, P = same-language peer
TESTS, TOKENS = 8, 1024

PROMPT = """Below are two worked solutions, TEXT A and TEXT B, of the same calculation. They contain the same numbers but are laid out differently.

Rewrite TEXT A so that its LAYOUT is exactly that of TEXT B, while its WORDING stays that of TEXT A.

LAYOUT (take it from TEXT B): the overall structure and markup (headings, numbering, bullet points, bold, code blocks, line breaks and blank lines); which elements each step contains and in which order (for example whether the digits are listed one by one, whether a number is written as a spaced digit string, before/after lines, arrows, a separate result line, whether a calculation is written as an equation or in a code block); whether there is an introduction or a closing summary; the format of the final answer line.

WORDING (keep it from TEXT A): the sentences and phrases. For each element of TEXT B's layout, reuse TEXT A's own sentence or phrase for that element verbatim if TEXT A has one; if TEXT A has none, write the element in new words in the style of TEXT A's phrasing. Never copy a sentence or phrase from TEXT B; only its layout, its numbers and the names of the steps may be shared.

Rules:
- Each step must contain exactly the elements of the corresponding step of TEXT B, in the same order: no more, no fewer. Drop every sentence, line or detail of TEXT A that has no counterpart in TEXT B. For example, if TEXT B does not list the digits of a number one by one, the rewrite must not list them either; if TEXT B does, the rewrite must.
- Keep the language of TEXT A. Every number must be correct, and the result of every step must appear.
- Finish with a final answer line in TEXT B's format giving the final answer.
- First, inside <plan> and </plan>, go through TEXT B line by line: for each line name its element (for example 'heading of step 2', 'the digits listed one by one', 'result line of step 2') and give the phrase of TEXT A you will use for it, or NEW. Then write the rewritten text inside <text> and </text>, with nothing after it.

=== TEXT A ===
{a}
=== END OF TEXT A ===

=== TEXT B ===
{b}
=== END OF TEXT B ==="""


def episodes():
    out = []
    for s in STUDIES:
        m = read(BASE / s / 'matched.json')
        out += [(s, n) for n in sorted(m) if len(m[n]) >= max(P.COUNTS)]
    return out


def world(s, n): return read(BASE / s / 'worlds' / f'{n}.json')
def demo(s, n):
    """The first matched pool input (the N = 1 example in PROC) and each source's matched worked example for it."""
    q = read(BASE / s / 'matched.json')[n][0]
    texts = {m: read(BASE / s / 'generations' / m / f"{n}__{q['index']:02}.json")['samples'][q['sample'][m]]['text'] for m in P.NAMES}
    return world(s, n)['pool'][q['index']], texts


def rewrite_path(s, n, w, l): return DIRECTORY / 'rewrites' / s / n / f'{w}__in__{l}.json'
def eval_path(s, n, r, c, i): return DIRECTORY / 'evaluations' / s / r / f'{n}__{c}__{i:02}.json'
def original_path(s, n, r, src, i): return BASE / s / 'evaluations' / r / f'{n}__src-{src}-1__{i:02}.json'
def valid(text, item): return P.correct(text, item) and P.shows_steps(text, item)
def cell_sources(r, c): return (r if c[0] == 'O' else PEER[r]), (r if c[1] == 'O' else PEER[r])


class Rewriter:
    """Pinned OpenRouter model, cached by request digest, with a study-level billed-cost ceiling."""
    def __init__(self, concurrency=6):
        load_dotenv(Path.home() / '.env')
        self.config = {'model': REWRITER['model'], 'provider': {'only': [REWRITER['provider']], 'allow_fallbacks': False}, 'usage': {'include': True}}
        self.directory = DIRECTORY / 'api' / digest(self.config); self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'engine.json').write_text(json.dumps(self.config, indent=2))
        self.client = httpx.AsyncClient(timeout=300, headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
        self.semaphore = asyncio.Semaphore(concurrency)
        self.spent = sum(json.loads(p.read_text())['response']['usage']['cost'] for p in self.directory.glob('*.json') if len(p.stem) == 64)

    async def call(self, prompt, temperature, seed):
        request = {**self.config, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': temperature, 'max_tokens': 4096, 'seed': seed}
        path = self.directory / f'{digest(request)}.json'
        if path.exists(): return json.loads(path.read_text())['response']
        async with self.semaphore:
            assert self.spent < BUDGET, f'rewrite budget ${BUDGET} reached; cached work is kept'
            for attempt in range(10):
                try: response = await self.client.post('https://openrouter.ai/api/v1/chat/completions', json=request)
                except (httpx.TimeoutException, httpx.TransportError): await asyncio.sleep(min(2 ** attempt, 30)); continue
                if response.status_code in {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}: await asyncio.sleep(min(2 ** attempt, 60) * (.5 + random.random())); continue
                response.raise_for_status(); body = response.json()
                assert body['provider'] == 'Anthropic', body['provider']
                self.spent += body['usage']['cost']; write(path, {'request': request, 'response': body, 'time': time.time()}); return body
            raise RuntimeError('rewriter kept failing')


async def rewrite(eps):
    rw = Rewriter()
    async def one(s, n, w, l, item, texts):
        path = rewrite_path(s, n, w, l)
        if path.exists(): return
        if w == l:  # own wording in own layout is the original example (identity rewrites in the pilot returned it verbatim)
            write(path, {'study': s, 'world': n, 'wording': w, 'layout': l, 'text': texts[w], 'valid': valid(texts[w], item), 'attempts': 0, 'identity': True}); return
        attempts = []
        for k, temp in enumerate([0., .3, .3, .3]):
            body = await rw.call(PROMPT.format(a=texts[w], b=texts[l]), temp, k)
            raw = body['choices'][0]['message']['content']; m = re.search(r'<text>(.*?)</text>', raw, re.S); text = m.group(1).strip() if m else ''; attempts.append(text)
            if body['choices'][0].get('finish_reason') == 'stop' and valid(text, item): break
        write(path, {'study': s, 'world': n, 'wording': w, 'layout': l, 'text': text, 'valid': valid(text, item), 'attempts': len(attempts), 'rewriter': REWRITER})
    jobs = []
    for s, n in eps:
        item, texts = demo(s, n)
        jobs += [one(s, n, w, l, item, texts) for a, b in PAIRS for w, l in product((a, b), (a, b))]
    for f in tqdm(asyncio.as_completed(jobs), total=len(jobs), desc='rewrites', mininterval=10): await f
    await rw.client.aclose(); print(f'spent ${rw.spent:.3f}')


def pair_ok(s, n, r): return all(read(rewrite_path(s, n, w, l))['valid'] for w, l in product((r, PEER[r]), (r, PEER[r])))


# ---------- manipulation checks ----------
def signature(text):
    """Line types: H heading, B bullet, N numbered, C code, S spaced digit string, R rule, Q quote, T text; blank runs as E."""
    out, code = [], False
    for line in text.splitlines():
        t = line.strip()
        if t.startswith('```'): code = not code; out.append('C'); continue
        if code: out.append('C'); continue
        if not t: out.append('E') if (not out or out[-1] != 'E') else None; continue
        out.append('H' if t.startswith('#') else 'R' if re.fullmatch(r'[-*_=]{3,}', t) else 'Q' if t.startswith('>') else 'B' if re.match(r'[-*•・]\s', t)
                   else 'N' if re.match(r'\d+[.)、]\s*\S', t) and not re.fullmatch(r'[\d\s.,→=+\-*×÷()]+', t) else 'S' if re.search(r'(?:\d[ ,、]\s*){3,}\d', t) else 'T')
    return ''.join(out)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def scaffold(text): return len(re.findall(r'(?:\d[ ,、，]\s*){3,}\d', text))  # digits written out one by one
def wording_side(x, a, b):
    """Of the rewrite's 4-grams found in exactly one source, the share found in a (1: wording entirely from a)."""
    g, ga, gb = grams(x), grams(a), grams(b); ua, ub = len(g & (ga - gb)), len(g & (gb - ga)); return ua / max(ua + ub, 1)
def layout_sim(x, y): sx, sy = signature(x), signature(y); return 1 - levenshtein(sx, sy) / max(len(sx), len(sy), 1)
def grams(text): t = re.sub(r'[\d\s*#`>_\-–—|.,:;!?()\[\]{}"\'“”‘’→=+×÷/\\。、，：；！？（）「」『』【】]+', '', text.lower()); return {t[i:i + 4] for i in range(len(t) - 3)}
def wording_share(x, y): gx = grams(x); return len(gx & grams(y)) / max(len(gx), 1)


def check(eps, show):
    rows = []
    for s, n in eps:
        _, texts = demo(s, n)
        for a, b in PAIRS:
            for w, l in product((a, b), (a, b)):
                t = read(rewrite_path(s, n, w, l))
                o = b if w == a else a  # the other source
                rows.append(dict(s=s, n=n, pair=f'{a}/{b}', transplant=w != l, valid=t['valid'], attempts=t['attempts'],
                                 layout_to_layout_src=layout_sim(t['text'], texts[l]), layout_to_wording_src=layout_sim(t['text'], texts[w]),
                                 words_from_wording_src=wording_share(t['text'], texts[w]), words_from_other=wording_share(t['text'], texts[o]), wording_side=wording_side(t['text'], texts[w], texts[o]),
                                 length_ratio_to_layout_src=len(t['text']) / len(texts[l]), digit_lists=scaffold(t['text']), digit_lists_layout_src=scaffold(texts[l]), digit_lists_wording_src=scaffold(texts[w])))
    write(DIRECTORY / 'checks.json', rows)
    for tr in [False, True]:
        r = [x for x in rows if x['transplant'] == tr]
        print(f"{'transplants' if tr else 'identity'} n={len(r)} valid {np.mean([x['valid'] for x in r]):.3f} retried {np.mean([x['attempts'] > 1 for x in r]):.3f} | "
              + '  '.join(f'{k} {np.mean([x[k] for x in r]):.2f}' for k in ['layout_to_layout_src', 'layout_to_wording_src', 'words_from_wording_src', 'words_from_other', 'wording_side', 'length_ratio_to_layout_src', 'digit_lists', 'digit_lists_layout_src', 'digit_lists_wording_src']))
    for s, n in show:  # full spot check of one episode, every text in full
        item, texts = demo(s, n); print(f'\n######## {s} {n} input {item["input"]} values {item["values"]}')
        for a, b in PAIRS[:1] if len(show) > 1 else PAIRS:
            for m in (a, b): print(f'\n======== ORIGINAL {m} ========\n{texts[m]}')
            for w, l in product((a, b), (a, b)):
                t = read(rewrite_path(s, n, w, l)); print(f'\n======== REWRITE wording {w} in layout {l} (valid {t["valid"]}, attempts {t["attempts"]}) ========\n{t["text"]}')


# ---------- readers ----------
async def evaluate(eps):
    from mlayout_api import Served
    models = {m: Served(m, 64) for m in P.NAMES}; write(DIRECTORY / 'inference.json', {m: {'engine': x.engine} for m, x in models.items()})
    async def one(s, n, r, c, i, item):
        path = eval_path(s, n, r, c, i)
        if path.exists(): return
        w = world(s, n); t = P.TEXT[P.LANGUAGE[r]]; wsrc, lsrc = cell_sources(r, c)
        context = t['header'] + t['example'].format(x=item['input'], text=read(rewrite_path(s, n, wsrc, lsrc))['text'])
        msg = [{'role': 'user', 'content': context + '\n\n' + t['test'].format(y=w['tests'][i]['input'])}]
        write(path, {'reader': r, 'study': s, 'world': n, 'cell': c, 'index': i, **(await models[r].call(msg, w['seed'] + i, 0., TOKENS))[0]})
    jobs = []
    for s, n in eps:
        item, _ = demo(s, n)
        jobs += [one(s, n, r, c, i, item) for r in P.NAMES if pair_ok(s, n, r) for c in CELLS for i in range(TESTS)]
    for f in tqdm(asyncio.as_completed(jobs), total=len(jobs), desc='readers', mininterval=15): await f
    for x in models.values(): await x.close()
    write(DIRECTORY / 'complete.json', {'time_uk': now(), 'episodes': len(eps)})


# ---------- analysis ----------
def signflip(x, draws=100000, seed=0):
    x = np.asarray(x); obs = abs(x.mean()); s = np.random.default_rng(seed).choice([-1, 1], (draws, len(x)))
    return float((np.abs((s * x).mean(1)) >= obs - 1e-12).mean())


def boot(x, draws=10000, seed=1):
    x = np.asarray(x); b = x[np.random.default_rng(seed).integers(0, len(x), (draws, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def summary(x): return dict(mean=float(np.mean(x)), ci=boot(x), p=signflip(x), n=len(x))
def holm(p): p = np.asarray(p); o = np.argsort(p); adj = np.maximum.accumulate([(len(p) - k) * p[i] for k, i in enumerate(o)]); out = np.empty(len(p)); out[o] = np.minimum(adj, 1); return out


def analyze(eps):
    acc = {}  # (s, n, r) -> {cell: accuracy}
    for s, n in eps:
        w = world(s, n)
        for r in P.NAMES:
            if not pair_ok(s, n, r): continue
            a = {c: np.mean([P.correct(read(eval_path(s, n, r, c, i))['text'], w['tests'][i]) for i in range(TESTS)]) for c in CELLS}
            a |= {k: np.mean([P.correct(read(original_path(s, n, r, src, i))['text'], w['tests'][i]) for i in range(TESTS)]) for k, src in [('own_orig', r), ('peer_orig', PEER[r])]}
            acc[s, n, r] = a
    contrasts = {'L_own_layout': lambda a: ((a['OO'] + a['PO']) - (a['OP'] + a['PP'])) / 2, 'W_own_wording': lambda a: ((a['OO'] + a['OP']) - (a['PO'] + a['PP'])) / 2,
                 'interaction': lambda a: a['OO'] - a['OP'] - a['PO'] + a['PP'], 'own_after_rewrite_OO_PP': lambda a: a['OO'] - a['PP'], 'own_original': lambda a: a['own_orig'] - a['peer_orig'],
                 'rewrite_artefact_own': lambda a: a['OO'] - a['own_orig'], 'rewrite_artefact_peer': lambda a: a['PP'] - a['peer_orig']}
    def per_episode(f, readers=P.NAMES, subset=None):
        out = []
        for s, n in eps:
            if subset and (s, n) not in subset: continue
            v = [f(acc[s, n, r]) for r in readers if (s, n, r) in acc]
            if v: out.append(100 * np.mean(v))
        return out
    res = {k: summary(per_episode(f)) for k, f in contrasts.items()}
    p = holm([res['L_own_layout']['p'], res['W_own_wording']['p']]); res['L_own_layout']['p_holm'], res['W_own_wording']['p_holm'] = float(p[0]), float(p[1])
    res['cells_mean_accuracy'] = {c: 100 * float(np.mean([a[c] for a in acc.values()])) for c in CELLS + ['own_orig', 'peer_orig']}
    res['per_reader'] = {r: {k: summary(per_episode(contrasts[k], [r])) for k in ['L_own_layout', 'W_own_wording', 'own_after_rewrite_OO_PP', 'own_original']} for r in P.NAMES}
    res['strong_readers'] = {k: summary(per_episode(contrasts[k], STRONG)) for k in ['L_own_layout', 'W_own_wording', 'own_after_rewrite_OO_PP']}
    for s in STUDIES: res[s] = {k: summary(per_episode(contrasts[k], subset={e for e in eps if e[0] == s})) for k in ['L_own_layout', 'W_own_wording']}
    for L in (3, 4): res[f'L{L}'] = {k: summary(per_episode(contrasts[k], subset={e for e in eps if e[1].startswith(f'L{L}')})) for k in ['L_own_layout', 'W_own_wording']}
    res['pairs_dropped'] = [(s, n, r) for s, n in eps for r in P.NAMES if (s, n, r) not in acc]
    write(DIRECTORY / 'analysis.json', res); print(json.dumps({k: v for k, v in res.items() if k != 'per_reader'}, indent=1))
    for r, v in res['per_reader'].items(): print(r, '  '.join(f"{k} {x['mean']:+.1f} [{x['ci'][0]:+.1f},{x['ci'][1]:+.1f}]" for k, x in v.items()))
    figure(acc, res)


def figure(acc, res, out=Path(__file__).resolve().parents[1] / 'reports' / 'proc-layout.png'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    BLUE, ORANGE, INK = '#2a78d6', '#eb6834', '#52514e'
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained', gridspec_kw={'width_ratios': [1, 1.6]})
    for k, (word, col) in enumerate([('O', BLUE), ('P', ORANGE)]):
        m = [100 * np.mean([a[word + lay] for a in acc.values()]) for lay in 'OP']
        a1.bar(np.arange(2) + (k - .5) * .38, m, .36, color=col, label=f"{'own' if word == 'O' else 'peer'} wording")
        for x, v in zip(np.arange(2) + (k - .5) * .38, m): a1.text(x, v + .5, f'{v:.1f}', ha='center', va='bottom', color=INK)
    a1.set_xticks([0, 1], ['own layout', 'peer layout']); a1.set_ylabel('test accuracy (%)'); a1.set_ylim(0, 100); a1.legend(frameon=False, loc='lower right')
    a1.set_title('A · accuracy after one worked example, all readers', loc='left'); a1.grid(axis='y', alpha=.25, lw=.5)
    names = list(res['per_reader']) + ['all readers']
    for j, (key, col, lab) in enumerate([('L_own_layout', BLUE, 'own layout (L)'), ('W_own_wording', ORANGE, 'own wording (W)')]):
        vals = [res['per_reader'][r][key] for r in names[:-1]] + [res[key]]
        y = np.arange(len(names)) + (j - .5) * .3
        a2.errorbar([v['mean'] for v in vals], y, xerr=np.array([[v['mean'] - v['ci'][0], v['ci'][1] - v['mean']] for v in vals]).T, fmt='o', color=col, ms=8, lw=2, capsize=0, label=lab)
    a2.axvline(0, color=INK, lw=1); a2.set_yticks(np.arange(len(names)), names); a2.invert_yaxis(); a2.set_xlabel('effect on test accuracy (pp, 95% CI over episodes)')
    a2.set_title('B · own layout vs own wording, per reader', loc='left'); a2.legend(frameon=False); a2.grid(axis='x', alpha=.25, lw=.5)
    fig.savefig(out, dpi=130); print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('stage', choices=['rewrite', 'check', 'evaluate', 'analyze'])
    ap.add_argument('--pilot', type=int, default=0, help='only the first k episodes of each study'); ap.add_argument('--show', default='')
    a = ap.parse_args(); eps = episodes()
    if a.pilot: eps = [e for s in STUDIES for e in [x for x in eps if x[0] == s][:a.pilot]]
    if a.stage == 'rewrite': asyncio.run(rewrite(eps))
    elif a.stage == 'check': check(eps, [tuple(x.split(':')) for x in a.show.split(',') if x])
    elif a.stage == 'evaluate': asyncio.run(evaluate(eps))
    else: analyze(eps)
