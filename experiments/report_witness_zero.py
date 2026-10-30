"""Report the separate zero-initial-example follow-up alongside its fixed controls."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def zero_section(root,out,summary,label,encoded):
    rows=sorted(summary['zero_screen'],key=lambda r:r['spec']);fresh=summary['zero_confirmation'];parts=['<h3>Closer to Self-ICL: the same task families with K = 0</h3><p>A separate plan was frozen before this follow-up and before inspecting the K = 4 confirmation. All twelve calibrated task settings were retained, with newly seeded episodes and no initial examples in either condition. Each source screen again used 4 × 8 tests and N = 0, 2, 8, 32. Generation still saw only its assigned question, without the held-out test input. This isolates an important difference from our four-example setup while retaining the strict source-only contrast.</p>']
    fig,ax=plt.subplots();y=np.arange(len(rows))
    for key,name in [('zero','Instruction only'),('teacher','Teacher N=32'),('own','Own N=32')]: ax.plot([r[key] for r in rows],y,'o',label=name)
    ax.set(yticks=y,yticklabels=[label(r['spec']) for r in rows],xlabel='Held-out accuracy',xlim=(0,1),title='No initial examples · exploratory screen');ax.invert_yaxis();ax.legend();fig.tight_layout();path=out/'task-search-witness-zero-screen.png';fig.savefig(path);plt.close(fig);parts.append('<figure>'+encoded(path)+'</figure>')
    parts.append('<details><summary>All K = 0 screen values and learning curves</summary><table><thead><tr><th>Task</th><th>Instruction only</th><th>Teacher N=32</th><th>Own N=32</th></tr></thead><tbody>')
    for r in rows: parts.append(f'<tr><td>{label(r["spec"])}</td><td>{r["zero"]:.1%}</td><td>{r["teacher"]:.1%}</td><td>{r["own"]:.1%}</td></tr>')
    parts.append('</tbody></table><div class="columns">');screen=json.loads((root/'witness_zero_screen/analysis.json').read_text())
    for row in rows:
        fig,ax=plt.subplots()
        for arm,name in [('teacher','Teacher'),('self_fixed','Own')]:
            values=sorted([r for r in screen['curves'] if r['spec']==row['spec'] and r['arm'] in {'zero',arm}],key=lambda r:r['n']);ax.plot([r['n'] for r in values],[r['accuracy'] for r in values],marker='o',label=name)
        ax.set(xlabel='Additional examples N',ylabel='Accuracy',ylim=(0,1),title=label(row['spec'])+' · K=0 screen');ax.legend();fig.tight_layout();p=root/'witness_analysis'/(row['spec']+'_zero_curve.png');fig.savefig(p);plt.close(fig);parts.append('<figure>'+encoded(p)+'</figure>')
    parts.append('</div></details>')
    if fresh:
        parts.append('<p>The strongest eligible K = 0 candidate received 16 fresh episodes × 16 tests, with N = 0, 1, 2, 4, 8, 16, 32 and fixed endpoint N = 32. The joint tests and Holm correction now include every K = 4 and K = 0 confirmation plus the nine earlier local primary attempts.</p><table><thead><tr><th>Task</th><th>Instruction / Teacher / Own</th><th>Own − teacher, points [95% CI]</th><th>Own − instruction, points [95% CI]</th><th>Joint Holm p</th></tr></thead><tbody>')
        for row in fresh:
            parts.append(f'<tr><td>{label(row["spec"])}</td><td>{row["zero"]:.1%} / {row["teacher"]:.1%} / {row["own"]:.1%}</td><td>{100*row["delta_teacher"]:+.1f} [{100*row["ci95_teacher"][0]:+.1f}, {100*row["ci95_teacher"][1]:+.1f}]</td><td>{100*row["delta_zero"]:+.1f} [{100*row["ci95_zero"][0]:+.1f}, {100*row["ci95_zero"][1]:+.1f}]</td><td>{row["p_joint_holm_all_primary"]:.4g}</td></tr>')
        parts.append('</tbody></table>')
        for row in fresh:
            fig,ax=plt.subplots()
            for arm,name in [('teacher','Teacher'),('self_fixed','Own')]:
                values=sorted([r for r in summary['zero_curves'] if r['spec']==row['spec'] and r['arm'] in {'zero',arm}],key=lambda r:r['n']);ax.plot([r['n'] for r in values],[r['accuracy'] for r in values],marker='o',label=name);ax.fill_between([r['n'] for r in values],[r['ci95'][0] for r in values],[r['ci95'][1] for r in values],alpha=.15)
            ax.set(xscale='symlog',xlabel='Additional examples N',ylabel='Accuracy',ylim=(0,1),title=label(row['spec'])+' · K=0 fresh confirmation');ax.legend();fig.tight_layout();path=out/('task-search-witness-zero-'+row['spec']+'.png');fig.savefig(path);plt.close(fig);parts.append('<figure>'+encoded(path)+'</figure>')
        names='; '.join(label(s) for s in summary['confirmed_zero_primary']) or 'none';parts.append('<p><strong>Confirmed K = 0 source advantages:</strong> '+names+'. This is a related control for the initial context, not an exact reproduction of Self-ICL’s test-conditioned input generation.</p>')
    else: parts.append('<p>No K = 0 screen candidate exceeded both controls at N = 32, so the fixed rule selected no fresh K = 0 confirmation. No additional source advantage was established.</p>')
    return '\n'.join(parts)
