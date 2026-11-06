"""For tex/proc_memo.html (via tex/figures/proc-layout.png; also used by the archived tex/archive/proc_brief.tex). Run: python proc_layout_brief_plot.py results/task_search/proc_layout/analysis.json tex/figures/proc-layout.png
Per-reader panel of reports/proc-layout.png (own-layout effect L and own-wording effect W, 95% CIs), redrawn from
results/task_search/proc_layout/analysis.json at a size that reads at text width, with readable reader names.
Panel A of the original (the four cell means) is given as a table in the brief instead."""
import json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import numpy as np

src, out = sys.argv[1], sys.argv[2]
res = json.load(open(src))
BLUE, ORANGE, INK = '#2a78d6', '#eb6834', '#52514e'
NAMES = {'olmo3_7b': 'OLMo-3-7B (en)', 'phi4_mini': 'Phi-4-mini (en)', 'gswallow_9b': 'Gemma2-Swallow-9B (ja)', 'abeja_qwen7b': 'ABEJA-Qwen2.5-7B (ja)', 'glm4_9b': 'GLM-4-9B (zh)', 'internlm3_8b': 'InternLM3-8B (zh)'}
fig, ax = plt.subplots(figsize=(5.4, 3.1), layout='constrained')
readers = list(res['per_reader']); labels = [NAMES[r] for r in readers] + ['all readers']
for j, (key, col, lab) in enumerate([('L_own_layout', BLUE, 'own layout (L)'), ('W_own_wording', ORANGE, 'own wording (W)')]):
    vals = [res['per_reader'][r][key] for r in readers] + [res[key]]
    y = np.arange(len(labels)) + (j - .5) * .3
    ax.errorbar([v['mean'] for v in vals], y, xerr=np.array([[v['mean'] - v['ci'][0], v['ci'][1] - v['mean']] for v in vals]).T, fmt='o', color=col, capsize=0, label=lab)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(np.arange(len(labels)), labels); ax.invert_yaxis(); ax.set_xlabel('effect on test accuracy (pp, 95% CI over episodes)')
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2); ax.grid(axis='x', alpha=.25, lw=.5)
fig.savefig(out, dpi=200); print(out)
