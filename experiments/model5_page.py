"""Build tex/model_5_lateral_verdict.html (OverLyX article) from experiments/model5_page.html: fill in the numbers from numbers.json
(model5_figures.py) and own_benefit_numbers.json, propagation_numbers.json, network_numbers.json and source_numbers.json next to it, typeset every \\(...\\) and \\[...\\] with KaTeX at build time (scripts are left alone) and inline KaTeX's stylesheet.
Figures are referenced from tex/figures/. Served at http://127.0.0.1:8095/practice-docs/model_5_lateral_verdict.html (reports/practice-docs -> tex/).
Usage: python model5_page.py <numbers.json> <out.html>"""

from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np
from zoneinfo import ZoneInfo

from report_katex import stylesheet, typeset

ROOT = Path(__file__).resolve().parents[1]


def f(x): return f'{x:.2f}' if x >= 0.095 else (f'{x:.3f}' if x >= 0.001 else f'{x:.5f}'.rstrip('0'))                             # a loss in nats


def pct(x): return f'{100 * x:.0f}%' if x >= 0.095 else f'{100 * x:.1f}%'


def pct3(x): return f'{100 * x:.2f}%'


def main():
    v = json.loads(Path(sys.argv[1]).read_text()); a, e, s, g, d, h = (v[k] for k in ('ablation', 'efficiency', 'source', 'gradient', 'depth', 'heatmap'))
    rows = []
    for n in ('1024', '2048', '4096', '8192', '16384', '32768'):
        r = e[n]; rows.append(f"<tr><td class=\"num\">{int(n):,}</td><td class=\"num\">{pct(1 - np.exp(-int(n) / 1024))}</td><td class=\"num\">{f(r['table'])}</td><td class=\"num\">{f(r['practice0.8'])}</td>"
                              f"<td class=\"num\">{f(r['recall']) if r['recall'] < 20 else '&gt; 20'}</td><td class=\"num\">{f(r['bottleneck'])}</td><td class=\"num\">{r['locked']}</td></tr>")
    gk = lambda arm, th, key: [g[f'{st}|{th}|{arm}'][key] for st in ('noise0.1', 'noise1.0', 'random')]
    best = np.array(h['kl']).min(0); i13, i22 = h['total'].index(8192), len(h['total']) - 1
    nums = dict(worlds=a['worlds'], start=f(a['start']), full=f(a['full']), table=f(a['table']), recall=f(a['recall']), locked=a['full_whole_worlds'], locked_min=pct(a['full_whole_min']), full_new=pct(a['full_new']),
                gen=f(a['gen']), no_veto=f(a['no_veto']), no_spread=f(a['no_spread']), no_spread_new=pct(a['no_spread_new']), private=f(a['private']), private_new=pct(a['private_new']), valid=pct(a['full_valid']),
                eff_rows='\n'.join(rows), locked_eff=e['4096']['locked'], x4096=f"{e['4096']['table'] / e['4096']['practice0.8']:.1f}", x16384=f"{e['16384']['table'] / e['16384']['practice0.8']:.1f}", p4096=f(e['4096']['practice0.8']), t4096=f(e['4096']['table']), p8192=f(e['8192']['practice0.8']), t8192=f(e['8192']['table']),
                p2048=f(e['2048']['practice0.8']), t2048=f(e['2048']['table']), p32768=f(e['32768']['practice0.8']), t32768=f(e['32768']['table']), r4096=f(e['4096']['recall']), r8192=f(e['8192']['recall']),
                own=f(s['own']), twin=f(s['twin']), twin_kl=f(s['twin_kl']), table_end=f(s['table_end']), own_wins=s['own_wins'], cross=f"{s['cross']:,}",
                teacher_end=f(s['teacher']), table_wins=h['table_wins'], totals=len(h['total']), h13=f(best[i13]), ht13=f(h['table'][i13]), h22=f(best[i22]), ht22=f(h['table'][i22]), h22share=f"{h['best'][i22]:g}",
                gw=f"{f(min(gk('wake', 0, 'kl')))} to {f(max(gk('wake', 0, 'kl')))}", gp=f"{f(min(gk('practice', 0, 'kl')))} to {f(max(gk('practice', 0, 'kl')))}", gvw=f"{pct(min(gk('wake', 0, 'valid')))} to {pct(max(gk('wake', 0, 'valid')))}",
                gvp=f"{pct(min(gk('practice', 0, 'valid')))} to {pct(max(gk('practice', 0, 'valid')))}", glock=sum(g[k]['locked'] for k in g if '|0|' in k), gel_t=pct(g['random|1|practice']['el_teacher']), gel_o=pct(g['random|1|practice']['el_own']),
                gth_w=f(g['random|1|wake']['kl']), gth_p=f(g['random|1|practice']['kl']), d_flat=f(d['16384']['flat']), d_first=f(d['16384']['first']), d_between=f(d['16384']['between']), d_flat2=f(d['131072']['flat']),
                d_first2=f(d['131072']['first']), d_between2=f(d['131072']['between']), built=datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z'))
    o = json.loads(Path(sys.argv[1]).with_name('own_benefit_numbers.json').read_text()); ob, od = o['block'], o['depth']; mix = np.array(od['mix']); bm = mix.min(0); tn = np.array(sorted(int(n) for n in ob)); tl = np.array([ob[str(n)]['table'] for n in tn])
    cell = lambda x: f'<td class="num">{f(x) if x < 20 else "&gt; 20"}</td>'; i2 = lambda t: od['total'].index(2 ** t)
    nums |= dict(ob_rows='\n'.join(f'<tr><td class="num">{int(n):,}</td>' + ''.join(cell(ob[n][k]) for k in ('table', 'note', 'eta1', 'found', 'oracle')) + '</tr>' for n in ('1024', '2048', '4096', '8192', '16384', '32768')),
                 ob_drows='\n'.join(f'<tr><td class="num">\\(2^{{{int(np.log2(t))}}}\\)</td>{cell(od["flat"][i])}{cell(bm[i])}<td class="num">{od["best_own_share"][i]:g}</td>{cell(od["clustered"][i])}</tr>' for i, t in enumerate(od['total'])),
                 ob_ratio=f"{np.exp(np.interp(-np.log(ob['2048']['found']), -np.log(tl), np.log(tn))) / 2048:.0f}", ob_own_below=f"{o['speed']['below_table']:,}", ob_dworlds=od['worlds'], ob_dc15=f(od['clustered'][i2(15)]), ob_df15=f(od['flat'][i2(15)]),
                 **{f'ob_{k}{n}': f(ob[str(n)][key]) for k, key in (('t', 'table'), ('p', 'eta1'), ('f', 'found'), ('o', 'oracle')) for n in (1024, 2048, 4096)}, **{f'ob_d{k}{t}': f(v[i2(t)]) for k, v in (('m', bm), ('f', od['flat'])) for t in (19, 20, 21)})
    assert all(ob[n]['eta1'] < ob[n]['table'] for n in ('2048', '4096', '8192', '16384')) and ob['1024']['eta1'] > ob['1024']['table'] and all(w == od['worlds'] for w in od['mix_wins_worlds'][i2(18):]) and all(bm[:i2(18)] > np.array(od['flat'][:i2(18)]))
    assert all(np.array(od['clustered']) <= bm) and (mix[:5, i2(19):] < np.array(od['flat'][i2(19):])).all() and 0.25 <= min(od['best_own_share'][i2(18):]) and max(od['best_own_share'][i2(18):]) <= 0.9                                    # claims made in section 6
    assert nums['glock'] == 0 and h['table_wins'] == len(h['total']) and e['16384']['table'] > e['16384']['practice0.8'] and e['32768']['table'] < e['32768']['practice0.8']   # claims made in the text
    pr = json.loads(Path(sys.argv[1]).with_name('propagation_numbers.json').read_text()); A, Bs, gr = pr['stack']['4096'], pr['stack']['16384'], pr['gradient']; at = lambda r, n: r['own'].index(n)   # section 7
    gs = np.array(gr['samples']); gt, go = np.array(gr['t1_th0']), np.array(gr['o1_th0']); cross = int(gs[np.flatnonzero(~np.flip(np.cumprod(np.flip(gt >= go - 0.005)).astype(bool)))[-1] + 1]); imin = int(np.argmin(go))
    nums |= dict(pr_sworlds=A['worlds'], pr_gworlds=gr['worlds'], pr_ao2=pct(A['o2'][0]), pr_at2=pct(A['t2'][0]), pr_ao2m=pct(A['o2'][at(A, 2 ** 16)]), pr_at2m=pct(A['t2'][at(A, 2 ** 16)]), pr_ao2e=pct(A['o2'][-1]), pr_at2e=pct(A['t2'][-1]),
                 pr_bt2=pct(Bs['t2'][0]), pr_bo2=pct(Bs['o2'][0]), pr_at30=pct(A['t3'][0]), pr_at3e=pct(A['t3'][-1]), pr_bt30=pct(Bs['t3'][0]), pr_bt3e=pct(Bs['t3'][-1]), pr_go1=pct(go[0]), pr_gt1=pct(gt[0]), pr_go1min=pct(go[imin]),
                 pr_go1min_at=f'{int(gs[imin]):,}', pr_gcross=f'{cross:,}', pr_gt1e=pct(min(gt[-1], go[-1])), pr_gloss=f(gr['loss_th0'][-1]), pr_gt1max1=pct(max(gr['t1_th1'])), pr_gloss1=f(gr['loss_th1'][-1]))
    assert A['o2'][0] > A['t2'][0] and Bs['t2'][0] > Bs['o2'][0] and A['t3'][0] < 0.01 and Bs['t3'][0] < 0.05 and Bs['t3'][-1] > 0.95 and gt[0] < 0.1 < 0.7 < go[0] and max(gr['t1_th1']) < 0.15 and gt[-1] > 0.95 and go[-1] > 0.95   # claims made in section 7
    nw = json.loads(Path(sys.argv[1]).with_name('network_numbers.json').read_text()); nb, nc, bu = nw['baseline'], nw['curves'], nw['budget']; Pi = lambda n, P: nc[str(n)]['P'].index(P); c = lambda n, arm, P: nc[str(n)][arm][Pi(n, P)]; tb = lambda t: bu['total'].index(2 ** t)   # section 8
    bmix = np.array(bu['mix']).min(0); W8 = nc['4096']['worlds']
    nums |= dict(nw_brows='\n'.join(f'<tr><td class="num">{int(n):,}</td>' + ''.join(cell(nb[n][k]) for k in ('table', 'net16', 'hidden16', 'oracle')) + '</tr>' for n in ('1024', '2048', '4096', '8192')), nw_bworlds=nb['2048']['worlds'],
                 nw_bh2048=f(nb['2048']['hidden16']), nw_bh8192=f(nb['8192']['hidden16']), nw_bo2048=f(nb['2048']['oracle']), nw_bn2048=f(nb['2048']['net16']), nw_bratio=f"{nb['2048']['net16'] / nb['2048']['hidden16']:.0f}", nw_worlds=W8,
                 nw_crows='\n'.join(f'<tr><td class="num">{P:,}</td>' + ''.join(cell(c(4096, arm, P)) for arm in ('fresh', 'own', 'twin', 'draws')) + '</tr>' for P in (512, 1024, 2048, 4096, 8192, 16384, 32768)),
                 nw_o4096=f(c(4096, 'own', 4096)), nw_f4096=f(c(4096, 'fresh', 4096)), nw_o8192=f(c(8192, 'own', 8192)), nw_f8192=f(c(8192, 'fresh', 8192)), nw_o2048=f(c(2048, 'own', 8192)), nw_f2048=f(c(2048, 'fresh', 8192)),
                 nw_o1024=f(c(1024, 'own', 4096)), nw_f1024=f(c(1024, 'fresh', 4096)), nw_inv1024=f"{nc['1024']['invalid']:.0f}", nw_j2048=f(nc['2048']['judge']), nw_j4096=f(nc['4096']['judge']), nw_j8192=f(nc['8192']['judge']),
                 nw_t2048=f(c(2048, 'twin', 2048)), nw_ot2048=f(c(2048, 'own', 2048)), nw_s4096=f(c(4096, 'steps', 32768)), nw_w4096=f(c(4096, 'written', 32768)), nw_ri0=f"{nc['2048']['invalid_reader'][0]:.0f}", nw_ri1=f"{nc['2048']['invalid_reader'][-1]:.0f}",
                 nw_rj0=f(nc['2048']['judge_reader'][0]), nw_rj1=f(nc['2048']['judge_reader'][-1]), nw_or2048=f(c(2048, 'own_reader', 32768)), nw_oe2048=f(c(2048, 'own', 32768)), nw_bw=bu['worlds'], nw_m13=f(bmix[tb(13)]), nw_bf13=f(bu['fresh'][tb(13)]),
                 nw_m14=f(bmix[tb(14)]), nw_bf14=f(bu['fresh'][tb(14)]), nw_bt16=f(bu['table'][tb(16)]), nw_m16=f(bmix[tb(16)]), nw_s0=f(c(4096, 'own', 0)), nw_e4096=f(c(4096, 'own', 32768)))
    assert all(x == W8 for x in nc['4096']['own_wins'][Pi(4096, 512):Pi(4096, 8192) + 1]) and all(x == W8 for x in nc['8192']['own_wins'][1:]) and c(2048, 'own', 4096) > c(2048, 'fresh', 4096) and abs(c(2048, 'own', 2048) - c(2048, 'fresh', 2048)) < 0.01
    assert c(2048, 'twin', 2048) < c(2048, 'own', 2048) and all(c(n, 'draws', P) > 0.95 * c(n, 'fresh', P) for n in (2048, 4096, 8192) for P in nc['4096']['P'][1:]) and all(c(n, 'own', 32768) >= nc[str(n)]['judge'] for n in (2048, 4096, 8192))
    assert nb['8192']['hidden16'] < c(4096, 'own', 4096) and all(sh in (0.125, 0.25) for sh in bu['best_share'][tb(12):]) and all(x == bu['worlds'] for x in bu['wins'][tb(12):]) and bu['table'][tb(16)] < bmix[tb(16)] and nc['2048']['invalid_reader'][-1] == 0                       # claims made in section 8
    on, lo, lr = nw['online'], nw['local'], nw['localrule']; oi = lambda n, P: on[str(n)]['P'].index(P); oc = lambda d, n, arm, P: d[str(n)][arm][oi(n, P)]; W9 = on['8192']['worlds']                                              # section 9
    nums |= dict(w9_rows='\n'.join(f'<tr><td class="num">{P:,}</td>' + ''.join(cell(oc(on, 8192, arm, P)) for arm in ('fresh', 'table', 'online', 'online_local', 'online_twin', 'own')) + '</tr>' for P in (1024, 2048, 4096, 8192, 16384, 32768)), w9_worlds=W9,
                 w9_o=f(oc(on, 8192, 'online', 4096)), w9_f=f(oc(on, 8192, 'fresh', 4096)), w9_o2048=f(oc(on, 2048, 'online', 4096)), w9_f2048=f(oc(on, 2048, 'fresh', 4096)), w9_k1024=f(oc(on, 8192, 'own', 1024)), w9_l=f(oc(on, 8192, 'online_local', 4096)),
                 w9_top=f(oc(on, 8192, 'online_top', 4096)), w9_ap0=f"{on['4096']['online_apart'][0]:.2f}", w9_ap1=f"{on['4096']['online_apart'][-1]:.2f}", w9_apt=f"{on['4096']['online_top_apart'][-1]:.2f}",
                 w9_lrows='\n'.join(f'<tr><td class="num">{int(n):,}</td>' + ''.join(cell(lo[n][k]) for k in ('table', 'writes', 'hidden', 'judge')) + f'<td class="num">{lo[n]["invalid"]:.0f}</td></tr>' for n in ('1024', '2048', '4096', '8192', '16384')),
                 w9_lw4096=f(lo['4096']['writes']), w9_lw8192=f(lo['8192']['writes']), w9_lw16384=f(lo['16384']['writes']), w9_lh16384=f(lo['16384']['hidden']), w9_first=', '.join(str(x) for x in lo['first'][:-1]) + f" and {lo['first'][-1]}",
                 w9_rf=f(oc(lr, 8192, 'fresh', 4096)), w9_rl=f(oc(lr, 8192, 'online_local', 4096)), w9_rn=f(oc(lr, 8192, 'online', 4096)), w9_rt=f(oc(lr, 8192, 'table', 4096)), w9_rn2=f(oc(lr, 8192, 'online', 8192)), w9_rt2=f(oc(lr, 8192, 'table', 8192)))
    assert min(on['8192']['online_wins'][oi(8192, 1024):]) >= 15 and min(on['16384']['online_wins'][oi(16384, 256):]) >= 14 and oc(on, 2048, 'online', 4096) > oc(on, 2048, 'fresh', 4096) and abs(oc(on, 8192, 'online', 4096) - oc(on, 8192, 'own', 1024)) < 0.005
    assert all(lo[n]['right'] == 1 for n in ('2048', '4096', '8192', '16384')) and all(lo[n]['invalid'] == 0 for n in ('4096', '8192', '16384')) and all(lo[n]['table'] < lo[n]['writes'] for n in ('8192', '16384')) and lo['4096']['table'] < lo['4096']['writes'] + 0.03
    assert min(lr['8192']['online_local_wins'][1:]) == W9 and max(lr['8192']['online_local_wins_table']) <= 2 and min(lr['8192']['online_wins_table'][oi(8192, 8192):oi(8192, 16384) + 1]) == W9 and max(oc(lr, n, 'fresh', 32768) for n in (4096, 8192, 16384)) > 0.15   # claims made in section 9
    sr = json.loads(Path(sys.argv[1]).with_name('source_numbers.json').read_text()); sb, sl, sd, sw, sp = sr['bp'], sr['local'], sr['depth'], sr['watching'], sr['passes']; si = lambda d, n, P: d[str(n)]['P'].index(P); sc = lambda d, n, arm, P: d[str(n)][arm][si(d, n, P)]   # section 10
    row10 = lambda d, n, P, arms: f'<tr><td class="num">{P:,}</td>' + ''.join(cell(sc(d, n, a, P)) for a in arms) + '</tr>'; A10 = ('fresh', 'own', 'twin', 'judge', 'mix', 'mixtwin'); wi = lambda n: sw['ns'].index(n); W10, WD = sb['8192']['worlds'], sd['8192']['worlds']
    nums |= dict(s10_bp_rows='\n'.join(row10(sb, 8192, P, A10 + ('uniform',)) for P in (1024, 4096, 16384, 32768)), s10_lo_rows='\n'.join(row10(sl, 8192, P, A10 + ('uniform',)) for P in (1024, 4096, 16384, 32768)), s10_worlds=W10,
                 s10_bo=f(sc(sb, 8192, 'own', 4096)), s10_bt=f(sc(sb, 8192, 'twin', 4096)), s10_bw=sc(sb, 8192, 'twin_behind_own', 4096), s10_boe=f(sc(sb, 8192, 'own', 32768)), s10_bte=f(sc(sb, 8192, 'twin', 32768)), s10_bwe=sc(sb, 8192, 'twin_behind_own', 32768),
                 s10_lo=f(sc(sl, 8192, 'own', 4096)), s10_lt=f(sc(sl, 8192, 'twin', 4096)), s10_bj=f(sc(sb, 8192, 'judge', 4096)), s10_bje=f(sc(sb, 8192, 'judge', 32768)), s10_lj=f(sc(sl, 8192, 'judge', 4096)), s10_ltab=f(sc(sl, 8192, 'table', 4096)), s10_lm=f(sc(sl, 8192, 'mix', 4096)), s10_lmt=f(sc(sl, 8192, 'mixtwin', 4096)),
                 s10_bm=f(sc(sb, 8192, 'mix', 4096)), s10_bmt=f(sc(sb, 8192, 'mixtwin', 4096)), s10_bu=f(sc(sb, 8192, 'uniform', 4096)), s10_b0=f(sc(sb, 8192, 'own', 0)), s10_l0=f(sc(sl, 8192, 'own', 0)), s10_rv0=pct3(sc(sb, 8192, 'reward_valid', 0)), s10_rv1=pct3(sc(sb, 8192, 'reward_valid', 32768)),
                 s10_r1=f(sc(sb, 8192, 'reward', 32768)), s10_rt1=f(sc(sb, 8192, 'rewardtwin', 32768)), s10_bw0=f(sc(sb, 8192, 'own_wrong', 0)), s10_lw0=f(sc(sl, 8192, 'own_wrong', 0)), s10_dw0=f(sc(sd, 8192, 'own_wrong', 0)), s10_d0=f(sc(sd, 8192, 'own', 0)),
                 s10_cb=f"{sr['overlap']['bp']['8192'][1]:.2f}", s10_cl=f"{sr['overlap']['local']['8192'][1]:.2f}", s10_cd=f"{sd['8192']['corr']:.2f}", s10_ov=pct(sr['overlap']['local']['8192'][0]), s10_ltw=f(sc(sl, 8192, 'twin_wrong', 32768)), s10_low=f(sc(sl, 8192, 'own_wrong', 32768)),
                 s10_dtw=f(sc(sd, 8192, 'twin_wrong', 16384)), s10_dow=f(sc(sd, 8192, 'own_wrong', 16384)), s10_bow=f(sc(sb, 8192, 'own_wrong', 32768)), s10_btw=f(sc(sb, 8192, 'twin_wrong', 32768)), s10_djw=f(sc(sd, 8192, 'judge_wrong', 16384)),
                 s10_wrows='\n'.join(f'<tr><td class="num">{n:,}</td>' + ''.join(cell(sw[r][k][wi(n)]) for r, k in (('local', 'table'), ('local', 'writes'), ('local', 'hidden'), ('backprop', 'writes'), ('backprop', 'hidden'), ('local', 'oracle'))) + f'<td class="num">{pct(sw["local"]["right2"][wi(n)]) if sw["local"]["right2"][wi(n)] else "0%"}</td></tr>' for n in sw['ns']),
                 s10_wworlds=sw['worlds'], s10_uniform=f(float(np.log(64 ** 4 / 262144))), s10_h8192=f(sw['local']['hidden'][wi(8192)]), s10_r2l=pct(sw['local']['right2'][wi(4096)]), s10_r2b=pct(sw['backprop']['right2'][wi(4096)]), s10_t8192=f(sw['local']['table'][wi(8192)]), s10_t32768=f(sw['local']['table'][wi(32768)]),
                 s10_wb8192=f(sw['backprop']['writes'][wi(8192)]), s10_wb16384=f(sw['backprop']['writes'][wi(16384)]), s10_wl8192=f(sw['local']['writes'][wi(8192)]), s10_xb=f"{sw['backprop']['writes'][wi(8192)] / sw['backprop']['hidden'][wi(8192)]:.0f}", s10_pc0=sp['classes'][0], s10_ps0=sp['symbols'][0], s10_pc3=sp['classes'][3], s10_ps3=sp['symbols'][3],
                 s10_inv8192=f"{sw['local']['invalid'][wi(8192)]:,.0f}", s10_inv32768=f"{sw['local']['invalid'][wi(32768)]:,.0f}", s10_dn='8,192', s10_drows='\n'.join(row10(sd, 8192, P, A10 + ('shallow',)) for P in (1024, 2048, 4096, 8192, 16384)), s10_dworlds=WD,
                 s10_do=f(sc(sd, 8192, 'own', 16384)), s10_df=f(sc(sd, 8192, 'fresh', 16384)), s10_dwf=min(sd['8192']['own_below_fresh'][1:]), s10_do2=f(sc(sd, 16384, 'own', 16384)), s10_df2=f(sc(sd, 16384, 'fresh', 16384)), s10_dt=f(sc(sd, 8192, 'twin', 16384)), s10_dt2=f(sc(sd, 16384, 'twin', 16384)),
                 s10_dtk=f(sd['8192']['twin_after']), s10_dj=f(sc(sd, 8192, 'judge', 16384)), s10_dj2=f(sc(sd, 16384, 'judge', 16384)), s10_dm=f(sc(sd, 8192, 'mix', 16384)), s10_dmt=f(sc(sd, 8192, 'mixtwin', 16384)), s10_ds=f(sc(sd, 8192, 'shallow', 16384)), s10_ds2=f(sc(sd, 16384, 'shallow', 16384)))
    assert abs(sc(sb, 8192, 'own', 4096) - sc(sb, 8192, 'twin', 4096)) < 0.003 and sc(sb, 8192, 'twin_behind_own', 32768) == W10 and max(sl['8192']['twin_behind_own'][si(sl, 8192, 2048):]) == 0 and max(sl[n]['twin_behind_own'][si(sl, 8192, 2048)] for n in ('4096', '16384')) == 0
    assert all(sc(sl, 8192, 'judge', P) <= sc(sl, 8192, 'twin', P) + 0.001 < sc(sl, 8192, 'own', P) for P in (1024, 4096, 16384)) and sc(sl, 8192, 'judge', 4096) < sc(sl, 8192, 'table', 4096) < sc(sl, 8192, 'own', 4096) and sc(sb, 8192, 'judge', 4096) <= sc(sb, 8192, 'own', 4096) and sc(sb, 8192, 'judge', 32768) > sc(sb, 8192, 'own', 32768)
    assert sc(sb, 8192, 'uniform', 4096) > sc(sb, 8192, 'own', 0) and abs(sc(sb, 8192, 'uniform', 32768) - sc(sb, 8192, 'own', 32768)) < 0.002 and sc(sb, 8192, 'reward', 32768) > 2 * sc(sb, 8192, 'own', 0) and sc(sb, 8192, 'reward_valid', 32768) > sc(sb, 8192, 'reward_valid', 0) and sr['collapse']['loss'] > 10 and sr['collapse']['holes'] > 0.9
    assert sc(sl, 8192, 'mixtwin', 4096) < sc(sl, 8192, 'mix', 4096) and abs(sc(sb, 8192, 'mix', 4096) - sc(sb, 8192, 'mixtwin', 4096)) < 0.002 and all(2 * sc(d, 8192, 'own_wrong', 0) < sc(d, 8192, 'own', 0) for d in (sb, sl, sd)) and abs(sc(sb, 8192, 'judge_wrong', 32768) - sc(sb, 8192, 'own_wrong', 0)) < 0.001
    assert sc(sl, 8192, 'twin_wrong', 32768) < sc(sl, 8192, 'own_wrong', 32768) and sc(sd, 8192, 'twin_wrong', 16384) < sc(sd, 8192, 'own_wrong', 16384) and sc(sb, 8192, 'own_wrong', 32768) < sc(sb, 8192, 'twin_wrong', 32768) and sc(sd, 8192, 'judge_wrong', 16384) > sc(sd, 8192, 'own_wrong', 0)
    assert all(abs(sw[r]['hidden'][wi(n)] - sw[r]['oracle'][wi(n)]) < 0.0005 and sw[r]['right2'][wi(n)] == 1 for r in ('local', 'backprop') for n in (8192, 16384, 32768)) and all(sw[r]['right1'][wi(n)] > 0.98 for r in ('local', 'backprop') for n in sw['ns'][1:]) and min(sw['local']['table']) > 3.7
    assert all(sw['local']['refused'][wi(n)] == 0 for n in (8192, 16384, 32768)) and all(c < s_ for c, s_ in zip(sp['classes'], sp['symbols'])) and all(x == [1.0, 1.0] for x in sp['end'])
    assert min(sd['8192']['own_below_fresh'][1:]) >= 14 and min(sd['16384']['own_below_fresh'][1:]) == WD and max(sd['8192']['twin_behind_own'][2:]) <= 1 and max(sd['16384']['twin_behind_own'][2:]) == 0 and max(sd['8192']['judge_behind_own'] + sd['16384']['judge_behind_own']) == 0 and abs(sc(sd, 8192, 'mix', 16384) - sc(sd, 8192, 'mixtwin', 16384)) < 0.03
    assert all(min(sd[str(n)][a][-1] for a in ('own', 'twin', 'judge', 'mix', 'mixtwin')) > sw['backprop']['writes'][wi(n)] for n in (8192, 16384)) and sc(sd, 8192, 'shallow', 16384) > sc(sd, 8192, 'own', 0) and sc(sd, 16384, 'shallow', 16384) > 2 * sc(sd, 16384, 'own', 16384) and sc(sd, 8192, 'mix', 16384) < sc(sd, 8192, 'judge', 16384) and abs(sd['8192']['twin_after'] - sc(sd, 8192, 'own', 16384)) < 0.1   # claims made in section 10
    html = (ROOT / 'experiments/model5_page.html').read_text()
    for key in sorted(set(re.findall(r'\{\{(\w+)\}\}', html))): html = html.replace('{{' + key + '}}', str(nums[key]))
    head, rest = html.split('<main>', 1); body, tail = rest.split('</main>', 1)
    scripts = re.findall(r'<script>.*?</script>', body, re.S); body = re.sub(r'<script>.*?</script>', '<!--SCRIPT-->', body, flags=re.S); body = typeset(body)
    for sc in scripts: body = body.replace('<!--SCRIPT-->', sc, 1)
    html = head.replace('/*__KATEX_CSS__*/', stylesheet()) + '<main>' + body + '</main>' + tail
    assert '\\(' not in re.sub(r'<annotation.*?</annotation>', '', body, flags=re.S).replace(''.join(scripts), '')
    Path(sys.argv[2]).write_text(html); print(sys.argv[2]); print(f'{len(html) / 1e6:.2f} MB')


if __name__ == '__main__': main()
