"""Compile the research documents and record source-specific verification evidence."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from tqdm.auto import tqdm


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    fast = Path(os.environ['FAST_CACHE_DIR'])
    out = fast / 'generative_practice_loop_build'; out.mkdir(parents=True, exist_ok=True)
    python_checks = []
    for source in sorted(root.glob('experiments/practice_loop*.py')):
        py_compile.compile(str(source), cfile=str(out / (source.name + 'c')), doraise=True)
        python_checks.append({'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    documents = [p for p in sorted(root.glob('tex/**/practice_loop*.tex')) if '\\documentclass' in p.read_text()]
    documents += [root / 'tex/archive/practice_model.tex', root / 'tex/archive/practice_spectral.tex']
    builds = []
    for document in tqdm(documents, desc='Compile research documents'):
        for pass_number in [1, 2]:
            completed = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={out}', document.name], cwd=document.parent, capture_output=True, text=True, check=True)
            (out / f'{document.stem}-pass{pass_number}.stdout').write_text(completed.stdout)
        log = (out / f'{document.stem}.log').read_text()
        warnings = [line for line in log.splitlines() if 'Warning:' in line or 'Overfull' in line]
        assert 'undefined references' not in log and 'undefined citations' not in log
        builds.append({'source': str(document), 'sha256': hashlib.sha256(document.read_bytes()).hexdigest(), 'pages': int(re.search(r'Output written on .*?\((\d+) pages?', log, re.DOTALL).group(1)), 'warnings': warnings})
    evidence_paths = [fast / 'generative_practice_loop_theory/theory_checks.json', fast / 'generative_practice_loop_ring/theory_audit/summary.json', fast / 'generative_practice_loop_ring/separation_audit/summary.json', fast / 'generative_practice_loop_ring/posterior_separation_audit/summary.json', fast / 'generative_practice_loop_ring_audit/results.json', fast / 'generative_practice_loop_bayesian/audit/results.json']
    evidence_paths += [fast / 'generative_practice_loop_ring/orientation' / name for name in ['audit/results.json', 'audit/covariance_results.json', 'step_audit/summary.json', 'interval_audit/summary.json', 'noise_control/audit/results.json']]
    evidence = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'results': json.loads(path.read_text())} for path in evidence_paths]
    dependencies = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(root.glob('tex/**/practice*.tex')) + sorted(root.glob('tex/**/figures/practice-*.png'))]
    result = {'uk_verified': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'documents': builds, 'syntax_checked_sources': python_checks, 'document_sources_and_figures': dependencies, 'independent_evidence': evidence, 'scope': 'Compilation and source syntax checks do not rerun training. Independent analytical, replay, enumeration, Monte Carlo, and information-access audits are stored in the linked source-specific evidence. Source snapshots preserve original experimental versions. The dependency inventory includes all practice TeX fragments and PNG figures, including earlier context reports.'}
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'documents': builds, 'python_sources': len(python_checks), 'verification': str(out / 'verification.json')}, indent=2))


if __name__ == '__main__': main()
