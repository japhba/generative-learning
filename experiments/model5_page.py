"""Build tex/model_5_lateral_verdict.html (OverLyX article) from experiments/model5_page.html: fill in the numbers of the runs model5_cut_check.py, model5_cut_practice.py, model5_cut_depth.py,
model5_cut_twin.py, model5_cut_twin_codes.py and model5_cut_mechanism.py (the .npz and .json files they and their _figures.py scripts write), check the claims of the text against them, typeset
every \\(...\\) and \\[...\\] with KaTeX at build time and inline KaTeX's stylesheet and fonts. Figures are referenced from tex/figures/.
Served at http://127.0.0.1:8095/practice-docs/model_5_lateral_verdict.html (reports/practice-docs -> tex/). The note this one replaced is tex/model_5_appendix.tex.
Usage: python model5_page.py <directory with the data files> <out.html>"""

from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np
from zoneinfo import ZoneInfo

from report_katex import stylesheet, typeset

ROOT = Path(__file__).resolve().parents[1]


def f(x): return f'{x:.2f}' if abs(x) >= 0.095 else f'{x:.3f}'                                                                      # a loss in nats


def s(x): return 'never' if not np.isfinite(x) else f'{round(x, -2):,.0f}'                                                         # a number of samples


def pct(x): return '100%' if x > 0.9995 else f'{100 * x:.1f}%' if x > 0.995 else f'{100 * x:.0f}%'


def sg(x): return f'{x:+.2f}'.replace('-', '−') if abs(x) >= 0.005 else '0.00'                                                      # a change per 1,000 samples


def row(label, cells): return f'<tr><td>{label}</td>' + ''.join(f'<td class="num">{c}</td>' for c in cells) + '</tr>'


def main():
    D = Path(sys.argv[1]); J = lambda name: json.loads((D / name).read_text()); P, Dp, T, C, M = (J(f'cut_{k}.json') for k in ('practice_numbers', 'depth_numbers', 'twin_numbers', 'twin_codes', 'mechanism_numbers'))
    pf, sc, ck = (np.load(D / f'cut_{k}.npz') for k in ('practice_fresh', 'practice_scan', 'check')); W = M['worlds']; n = dict(worlds=W, built=datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z'))

    # one level
    p1, cols = P['numbers']['1024'], ('2048', '4096', '8192', '16384', '65536'); arms1, i1024 = list(pf['arms']), list(pf['ns']).index(1024); fin = lambda a, k: pf['res'][i1024, :, 0, arms1.index(a), -1, k]
    n['one_rows'] = '\n'.join([row(lab, [f(p1[t][k]) for t in cols]) for lab, k in (('Wake only', 'w'), ('Wake and sleep, best split', 'best_ws'), ("Wake read by the writer's own posterior", 'wtied'), ('Wake, sleep and practice', 'wsp'))]
                              + [row('Worlds in which practice is below wake and sleep', [p1[t]['worlds_below_ws'] for t in cols])])
    n |= dict(one_w=f(p1['65536']['w']), one_wrong256=pct(P['judge_accepts_wrong']['256']), one_wsp256=f(P['numbers']['256']['262144']['wsp']), one_ws256=f(P['numbers']['256']['262144']['best_ws']), one_512_worlds=P['numbers']['512']['262144']['worlds_below_ws'],
              sp_div=int((fin('SP', 0) > 0.35).sum()), wp_agree=pct(float(np.median(fin('WP', 1)))))
    R, E, cks = sc['res'][0], sc['etas'], sc['checks']; stay = lambda L: (lambda ok: cks[ok.argmax()] if ok.any() else np.inf)(np.array([(L[i:] < 0.35).all() for i in range(len(L))]))   # learned prior: the best steps of each arm
    def best(arm): t = [np.array([stay(l) for l in R[:, i, list(sc['arms']).index(arm), :, 0]]) * 128 for i in range(len(E)) if E[i, 3] == 1 and not np.isnan(R[:, i, list(sc['arms']).index(arm)]).all()]; return min(t, key=lambda x: (-np.isfinite(x).sum(), np.median(x)))
    lw, lp = best('WS'), best('WSP'); n |= dict(lp_ws=s(np.median(lw)), lp_wsp=s(np.median(lp)))
    assert all(p1[t]['worlds_below_ws'] >= 15 and p1[t]['wsp'] < p1[t]['best_ws'] for t in cols[1:]) and p1['2048']['wsp'] > p1['2048']['best_ws'] and P['numbers']['256']['262144']['wsp'] > P['numbers']['256']['262144']['best_ws'] and np.median(lp) < np.median(lw)

    # two levels
    A = Dp['4096']['arms']; sizes = list(Dp); tot = lambda a, k: sum(Dp[z]['arms'][a][k] for z in sizes); at = Dp['4096']['at']
    n['two_rows'] = '\n'.join(row(lab, [s(A[a]['t1']), s(A[a]['t2']), s(A[a]['r1']) if a != 'Wtied' else 'n/a', s(A[a]['r2']) if a != 'Wtied' else 'n/a', f"{A[a]['bound']} of {W}"]) for lab, a in (('Wake only', 'W'), ('Wake and sleep', 'WS'), ('Wake, sleep and practice', 'WSP'), ("Wake read by the writer's own posterior", 'Wtied')))
    n |= dict(t1_ws=s(A['WS']['t1']), t2_ws=s(A['WS']['t2']), t1_wsp=s(A['WSP']['t1']), t2_wsp=s(A['WSP']['t2']), lead1=f"{A['WS']['t1'] / A['WSP']['t1']:.1f}", lead2=f"{A['WS']['t2'] / A['WSP']['t2']:.0f}", nb_ws=tot('WS', 'level2_not_before'), nb_ws_n=tot('WS', 'reach2'),
              nb_wsp=tot('WSP', 'level2_not_before'), nb_wsp_n=tot('WSP', 'reach2'), d_ws32=f(at['32768']['best_ws']), d_wsp32=f(at['32768']['wsp']), d_ws262=f(at['262144']['best_ws']), d_wsp262=f(at['262144']['wsp']), b_wsp=tot('WSP', 'bound'), b_twin=tot('twin', 'bound'),
              b_ws=min(tot('WS', 'bound'), tot('WS31', 'bound')), b_ws31=max(tot('WS', 'bound'), tot('WS31', 'bound')), g_wsp13=A['WSP13']['good'], wp2=A['WP']['bound'], b_tied=A['Wtied']['bound'])
    assert A['WSP']['bound'] == W and n['nb_wsp'] == n['nb_wsp_n'] and n['nb_ws'] == n['nb_ws_n'] and A['WSP']['t1'] < A['WS']['t1'] and A['WS']['t2'] / A['WSP']['t2'] > 2 * A['WS']['t1'] / A['WSP']['t1'] and A['WSP']['t2'] >= A['WSP']['t1'] and len(sizes) == 3 and not np.isfinite(A['W']['t1']) and A['WP']['bound'] == 0

    # whose samples
    Tw = T['4096']; lab = dict(own_cause='Own, credited to the units that wrote them', own_read='Own, credited by reading', cross_read="A twin's, twin at the same stage", ahead_read="A twin's, twin ahead")
    n['twin_rows'] = '\n'.join(row(lab[a], [f"{Tw[a]['t1']:.0f}", f"{Tw[a]['t2']:.0f}", f"{Tw[a]['bound']} of {W}"]) for a in lab); cr = Tw['cross_read']
    n |= dict(tw_ratio1=f"{cr['ratio1']:.1f}", tw_ratio2=f"{cr['ratio2']:.1f}", tw_later1=cr['later1'], tw_later2=cr['later2'], tw_n=cr['worlds'], ah_ratio2=f"{Tw['ahead_read']['ratio2']:.1f}", read_ratio2=f"{Tw['own_read']['t2'] / Tw['own_cause']['t2']:.1f}", tv1_16=f(Tw['tv']['16'][0]),
              tv1_128=f(Tw['tv']['128'][0]), tv2_64=f(Tw['tv']['64'][1]), co_fin=C['finished'], co_diff=C['different'], co_twin=f(C['twin_when_different']), co_own=f(C['own_when_different']), oc_t2=f"{Tw['own_cause']['t2']:.0f}", or_t2=f"{Tw['own_read']['t2']:.0f}", tw_bound=cr['bound'], or_bound=Tw['own_read']['bound'])
    assert cr['ratio2'] > cr['ratio1'] > 1 and cr['later2'] >= 12 and Tw['ahead_read']['ratio2'] > cr['ratio2'] and Tw['tv']['64'][1] > Tw['tv']['64'][0] and Tw['tv']['16'][0] > Tw['tv']['16'][1] and C['twin_when_different'] < 0.6 < C['own_when_different'] and abs(C['own_when_same'] - C['twin_when_same']) < 0.05

    # inside the network
    Mw, Ms, O = M['probes']['WSP'], M['probes']['WS'], M['order']; names = (('wake', 'A teacher string, read by the reader'), ('wake_tied', 'A teacher string, read as well as possible'), ('practice', 'An own sample, credited to the units that wrote it'), ('accepted', 'The same, verdicts of accepted samples only'),
                                                                      ('refused', 'The same, verdicts of refused samples only'), ('own_read', 'An own sample, credited by reading'), ('twin_read', "A twin's sample, credited by reading"))
    n['probe_rows'] = '\n'.join(row(lab, [sg(Mw[r]['level1'][2]), sg(Mw[r]['level2'][1]), sg(Ms[r]['level1'][2]), sg(Ms[r]['level2'][1])]) for r, lab in names); g, a128, a64, ab = M['gating']['WSP'], M['agree']['WSP']['128'], M['agree']['WSP']['64'], M['ablation']
    n |= dict(p_wake2=sg(Mw['wake']['level2'][1]), p_wake2_pos=pct(Mw['wake']['level2_positive'][1]), p_tied2=sg(Mw['wake_tied']['level2'][1]), p_prac2=sg(Mw['practice']['level2'][1]), p_ref2=sg(Mw['refused']['level2'][1]), p_acc2=sg(Mw['accepted']['level2'][1]), p_ref1=sg(Mw['refused']['level1'][2]),
              p_acc1=sg(Mw['accepted']['level1'][2]), p_ref2_late=sg(Mw['refused']['level2'][2]), p_acc2_late=sg(Mw['accepted']['level2'][2]), p_ref1_early=sg(Mw['refused']['level1'][1]), p_acc1_early=sg(Mw['accepted']['level1'][1]), l_wake=sg(Mw['wake']['loss'][2]), l_ref=sg(Mw['refused']['loss'][2]), p_or2=sg(Mw['own_read']['level2'][1]), p_tw2=sg(Mw['twin_read']['level2'][1]), p_tied2_ws=sg(Ms['wake_tied']['level2'][1]), p_wake2_ws=sg(Ms['wake']['level2'][1]),
              p_prac2_ws=sg(Ms['practice']['level2'][1]), neg_ratio=f"{Mw['refused']['level2'][1] / Mw['accepted']['level2'][1]:.0f}", u32=f"{O['WSP']['units_at'][1]:.0f}", code1=O['WSP']['code_first'], code2=O['WSP']['code_second'], ws_stuck=O['WS']['stuck'], g0=sg(g[0]), g1=sg(g[1]), g2=sg(g[2]),
              ag2_t=f(a128['agree2_teacher']), ag2_o=f(a128['agree2_own']), ag1_t=f(a128['agree1_teacher']), ag1_o=f(a128['agree1_own']), w64=pct((a64['valid2'] - 0.5) / 0.5), r64=pct(a64['agree2_teacher']), read1_start=f"{M['agree']['WSP']['0']['read1']:.2f}", read1_end=f(M['agree']['WSP']['2048']['read1']),
              read2_end=f(M['agree']['WSP']['2048']['read2']), j_shown256=pct(M['judge']['256']['shown']), j_acc256=pct(M['judge']['256']['valid_accepted']), j_wrong256=pct(M['judge']['256']['invalid_accepted']), j_shown512=pct(M['judge']['512']['shown']),
              ab_wsp=max(v['WSP']['bound'] for v in ab.values()), ab_ws=max(v['WS']['bound'] for v in ab.values()), b_wsp_here=O['WSP']['bound'], b_ws_here=O['WS']['bound'])
    assert Mw['wake']['level2'][1] < 0 and abs(Mw['wake_tied']['level2'][1]) < 0.1 and Mw['practice']['level2'][1] > 0.5 and Mw['refused']['level2'][1] > 1.5 * Mw['accepted']['level2'][1] and Mw['refused']['level2'][2] > 3 * Mw['accepted']['level2'][2] and Mw['refused']['level1'][2] > 1.5 * Mw['accepted']['level1'][2] and abs(Mw['refused']['level1'][1] - Mw['accepted']['level1'][1]) < 0.05 and Mw['wake']['loss'][2] < 0 < Mw['refused']['loss'][2] and Mw['wake']['level1'][2] < 0 < Mw['practice']['level1'][2]
    assert g[0] < 0.02 < g[1] < g[2] and abs(a128['agree2_teacher'] - a128['agree2_own']) < 0.05 and a128['agree1_teacher'] >= a128['agree1_own'] - 0.02 and (a64['valid2'] - 0.5) / 0.5 > 3 * a64['agree2_teacher'] and O['WSP']['duplicates'] == 0 and O['WS']['duplicates_among_stuck'] == O['WS']['stuck'] > 0
    assert O['WSP']['code_first'] + O['WSP']['code_second'] == O['WSP']['bound'] == W and M['judge']['512']['invalid_accepted'] == 0 and M['judge']['512']['valid_accepted'] == 1 and Mw['twin_read']['level2'][1] < 0.2 * Mw['own_read']['level2'][1] and Ms['wake_tied']['level2'][1] > 0.1 > Ms['wake']['level2'][1]
    assert n['ab_wsp'] < O['WSP']['bound'] and n['ab_ws'] < O['WS']['bound'] and O['WSP']['units_at'][2] == 4

    # the stronger baseline (one-level world)
    ns, wr, sco, ky = list(ck['ns']), list(ck['writers']), list(ck['scores']), list(ck['keys']); ix = [ns.index(z) for z in (256, 512, 1024, 2048)]; med = lambda w, k: [float(np.median(ck['res'][i, :, wr.index(w), sco.index(k)])) for i in ix]; ref = [float(np.median(ck['aux'][i, :, ky.index('cut_pool_ref')])) for i in ix]
    n['check_rows'] = '\n'.join([row('Judge refuses pairs that do not agree', [pct(x) for x in ref]), row('Blocks agree, all units from the reader', [pct(x) for x in med('all2', 'agree')]), row('Loss, all units from the reader', [f(x) for x in med('all2', 'loss')]),
                                 row('Loss, true classes and true feature', [f(x) for x in med('true', 'loss')]), row('Loss, no unit for the agreement', [f(x) for x in med('none', 'loss')])])
    assert ref[2] > 0.99 and med('all2', 'agree')[2] > 0.99 and med('all2', 'loss')[3] < 1.2 * med('true', 'loss')[3] and med('none', 'loss')[3] > 0.6

    html = (ROOT / 'experiments/model5_page.html').read_text(); keys = sorted(set(re.findall(r'\{\{(\w+)\}\}', html))); assert not set(keys) - set(n), set(keys) - set(n)
    for key in keys: html = html.replace('{{' + key + '}}', str(n[key]))
    head, rest = html.split('<main>', 1); body, tail = rest.split('</main>', 1); body = typeset(body); html = head.replace('/*__KATEX_CSS__*/', stylesheet()) + '<main>' + body + '</main>' + tail
    assert '\\(' not in re.sub(r'<annotation.*?</annotation>', '', body, flags=re.S) and '{{' not in html and 'katex-error' not in html
    Path(sys.argv[2]).write_text(html); print(sys.argv[2]); print(f'{len(html) / 1e6:.2f} MB'); print({k: v for k, v in n.items() if not k.endswith('_rows')})


if __name__ == '__main__': main()
