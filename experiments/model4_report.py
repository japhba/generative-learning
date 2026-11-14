"""Build a self-contained, locally served scientific report from the Model 4 CPU studies."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

from model4_feasibility import OUT, X, lse, lsm, sigmoid, tables
from report_katex import stylesheet, typeset

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(Path.home()/".env")
CACHE=Path(os.environ["FAST_CACHE_DIR"])/"model4_report"
SERVED=Path("/ceph/scratch/jbauer/reports/generative-learning")
NAMES={"forward_explore":"Practice + exploration", "forward":"Pure ancestral practice", "watch":"Stored-data Watching", "fresh":"Fresh-data Watching", "ae":"End-to-end reconstruction", "unweighted":"Unweighted own wake", "forward_frozen":"Frozen recognition", "forward_no_evidence":"No evidence", "forward_explore_frozen":"Frozen recognition", "forward_explore_no_evidence":"No evidence", "symmetric":"Symmetric joint KL", "practice":"Reverse joint KL", "mle":"Exact-posterior likelihood", "symmetric_frozen":"Frozen recognition", "symmetric_no_evidence":"No evidence"}
COLORS={"forward_explore":"#196a85","forward":"#82a9b2","watch":"#bd782d","fresh":"#337b4d","ae":"#ac4355","unweighted":"#777777","forward_explore_frozen":"#76508f","forward_explore_no_evidence":"#a3a3a3","symmetric":"#78539b","practice":"#b35c20"}
STUDIES=["exploration_local","exploration_basis","final_exact","final_sampled","confirmation","sampled","local_ablations"]


def read(name): return json.loads((OUT/f"{name}.json").read_text())
def mean_se(v):
    v=np.array(v)
    return float(v.mean()),float(v.std(ddof=1)/np.sqrt(len(v)))


def paired(runs, baseline, arm="forward_explore"):
    delta=np.array([r["curves"][baseline][-1]["kl"]-r["curves"][arm][-1]["kl"] for r in runs])
    rng=np.random.default_rng(462)
    ci=np.quantile(delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1),[.025,.975])
    return {"mean":float(delta.mean()),"ci":ci.tolist(),"wins":int((delta>0).sum()),"n":len(delta)}


def packed(name):
    data=read(name)
    records=[]
    keys=["kl","validity","root_mi","branch_mi","forward_objective","objective","importance_ess","effective_modes"]
    for r in data["runs"]:
        curves={a:{"step":[v["step"] for v in rows],**{k:[round(v[k],7) for v in rows] for k in keys}} for a,rows in r["curves"].items()}
        record={"seed":r["seed"],"curves":curves,"critic_kl":r["critic_kl"],"watching":{k:[v[k] for v in r["watching"]] for k in ["step",*keys]}}
        if name in ["exploration_local","exploration_basis"]:
            record.update({k:r[k] for k in ["mask","valid_indices","observed_indices","selected","signs"]})
            record["distributions"]={a:np.exp(lse(tables([np.array(v) for v in p])[0],1)).tolist() for a,p in r["parameters"].items()}
            record["distributions"]["critic"]=np.exp(np.array(r["energy"])-lse(np.array(r["energy"]))).tolist()
            record["parameters"]={a:p for a,p in r["parameters"].items() if a in ["forward_explore","forward"]}
        records.append(record)
    return records


def summary_table(runs):
    rows=[]
    for a in runs[0]["curves"]:
        kl,se=mean_se([r["curves"][a][-1]["kl"] for r in runs])
        validity=np.mean([r["curves"][a][-1]["validity"] for r in runs])
        mi=np.mean([r["curves"][a][-1]["root_mi"] for r in runs])
        med=np.median([r["curves"][a][-1]["kl"] for r in runs])
        rows.append(f'<tr><td>{NAMES[a]}</td><td>{kl:.3f} ± {se:.3f}</td><td>{med:.3f}</td><td>{validity:.3f}</td><td>{mi:.3f}</td></tr>')
    return '<table><thead><tr><th>Continuation</th><th>Population KL, mean ± SE</th><th>Median KL</th><th>Validity</th><th>Root information</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def main():
    CACHE.mkdir(parents=True,exist_ok=True); SERVED.mkdir(parents=True,exist_ok=True)
    source=read("exploration_local")["runs"]
    payload={"studies":{name:packed(name) for name in STUDIES},"names":NAMES,"colors":COLORS}
    sensitivity=read("local_sensitivity")["runs"]
    payload["sensitivity"]=[{k:r[k] for k in ["seed","n","coupling","critic_kl","false_constraints"]} | {"end":{a:row[-1] for a,row in r["curves"].items()}} for r in sensitivity]
    allocations=read("allocation_exploration")["runs"]
    fractions=[0,.25,.5,.75,.9,1]
    payload["allocation"]=[{"seed":r["seed"],"total":128*(r["steps"]+r["watchsteps"]),"fraction":min(fractions,key=lambda f:abs(f-r["steps"]/(r["steps"]+r["watchsteps"]))),"end":r["curves"]["forward_explore"][-1]} for r in allocations]
    assert all(sum(r["total"]==n and r["fraction"]==f for r in payload["allocation"])==8 for n in [12800,32000,64000,128000,256000] for f in fractions)
    statistics={a:paired(source,a) for a in ["forward","watch","fresh","ae"]}
    (OUT/"summary.json").write_text(json.dumps(statistics,indent=2))
    table=summary_table(source)
    comparisons=''.join(f'<tr><td>{NAMES[a]}</td><td>{v["mean"]:.3f}</td><td>[{v["ci"][0]:.3f}, {v["ci"][1]:.3f}]</td><td>{v["wins"]}/{v["n"]}</td></tr>' for a,v in statistics.items())
    comparisons='<table><thead><tr><th>Comparator</th><th>KL advantage of practice</th><th>Paired bootstrap interval</th><th>Seeds won</th></tr></thead><tbody>'+comparisons+'</tbody></table>'
    initial=np.mean([r["curves"]["forward_explore"][0]["kl"] for r in source])
    end,se=mean_se([r["curves"]["forward_explore"][-1]["kl"] for r in source])
    slopes=[np.polyfit(np.log([v["step"] for v in r["curves"]["forward_explore"][-8:]]),np.log([v["forward_objective"] for v in r["curves"]["forward_explore"][-8:]]),1)[0] for r in source]
    qsource=Path(__file__).with_name("model4_feasibility.py")
    shutil.copy2(qsource,OUT/"model4_feasibility.py")
    shutil.copy2(Path(__file__),OUT/"model4_report.py")
    shutil.copy2(Path(__file__).with_name("MODEL4_FEASIBILITY_PROTOCOL.md"),OUT/"MODEL4_FEASIBILITY_PROTOCOL.md")
    shutil.copy2(Path(__file__).with_name("MODEL4_FEASIBILITY_RESULTS.md"),OUT/"MODEL4_FEASIBILITY_RESULTS.md")
    shutil.copy2(Path(__file__).with_name("model4_report_template.html"),OUT/"model4_report_template.html")
    shutil.copy2(Path(__file__).with_name("model4_report.js"),OUT/"model4_report.js")
    shutil.copy2(Path(__file__).with_name("report_katex.py"),OUT/"report_katex.py")

    # A standalone export uses Matplotlib defaults; interactive figures use Plotly defaults.
    fig,ax=plt.subplots(layout="constrained")
    for a in ["forward_explore","forward","watch","fresh","ae"]:
        y=np.array([[v["kl"] for v in r["curves"][a]] for r in source]); t=np.array([v["step"] for v in source[0]["curves"][a]])+1
        ax.plot(t,y.mean(0),label=NAMES[a],color=COLORS[a]); sem=y.std(0,ddof=1)/len(y)**.5
        ax.fill_between(t,y.mean(0)-sem,y.mean(0)+sem,color=COLORS[a],alpha=.15)
    ax.set(xscale="log",xlabel="Continuation updates + 1",ylabel="Population KL (nats)",title="Offline practice with learned local constraints")
    ax.legend(); fig.savefig(OUT/"learning.png",facecolor="white"); fig.savefig(OUT/"learning.svg",transparent=True); plt.close(fig)

    artifacts=[]
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name!="manifest.json":
            artifacts.append({"name":path.name,"path":str(path),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest={"created_at":datetime.now().astimezone().isoformat(),"data_directory":str(OUT),"artifacts":artifacts}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    links=''.join(f'<li><a href="model4-files/{a["name"]}">{a["name"]}</a></li>' for a in artifacts if not a['name'].endswith('.svg'))
    html=Path(__file__).with_name("model4_report_template.html").read_text()
    values={"TABLE":table,"COMPARISONS":comparisons,"INITIAL":f"{initial:.3f}","FINAL":f"{end:.3f}","FINAL_SE":f"{se:.3f}","SLOPE":f"{np.median(slopes):.2f}","ARTIFACTS":links,"DURABLE":str(OUT)}
    for key,value in values.items(): html=html.replace('@'+key+'@',value)
    html=typeset(html).replace('@KATEX_CSS@',stylesheet())
    html=html.replace('@PLOTLY@',(CACHE/"plotly.min.js").read_text()).replace('@DATA@',json.dumps(payload,separators=(',',':')))
    html=html.replace('@SCRIPT@',Path(__file__).with_name('model4_report.js').read_text())
    assert '@TABLE@' not in html and '@SCRIPT@' not in html
    report=SERVED/"model4_feasibility.html"; report.write_text(html)
    link=ROOT/"reports/model4_feasibility.html"
    if not link.exists(): link.symlink_to(report)
    datalink=ROOT/"reports/model4-files"
    if not datalink.exists(): datalink.symlink_to(OUT)
    print(report)
    print("Report bytes",report.stat().st_size,"persistent data",OUT)


if __name__=="__main__": main()
