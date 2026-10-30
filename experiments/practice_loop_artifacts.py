"""Preserve this follow-up research pass with a verified persistent hash manifest."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from tqdm.auto import tqdm


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    fast = Path(os.environ['FAST_CACHE_DIR'])
    archive = Path(os.environ['CACHE_DIR']) / 'generative-learning/practice_research/loop'
    pairs = []
    data_roots = [p for p in sorted(fast.glob('generative_practice_loop*')) if p.is_dir() and p.name != 'generative_practice_loop_archive']
    for data_root in data_roots:
        pairs.extend((p, archive / 'runs' / data_root.name / p.relative_to(data_root)) for p in sorted(data_root.rglob('*')) if p.is_file())
    for source in sorted(root.glob('experiments/practice_loop*.py')): pairs.append((source, archive / 'code' / source.name))
    for source in sorted(root.glob('tex/**/practice*.tex')): pairs.append((source, archive / 'documents' / source.name))
    for source in sorted(root.glob('tex/**/figures/practice-*.png')): pairs.append((source, archive / 'documents/figures' / source.name))
    for source in [root / 'tex/README.md', root / 'tex/notes/table.tex', root / 'tex/archive/practice_model.tex', root / 'tex/archive/practice_spectral.tex']:
        pairs.append((source, archive / 'context' / source.name))
    for source in sorted(fast.glob('practice_loop*.log')) + sorted(fast.glob('generative_practice_loop*.log')): pairs.append((source, archive / 'logs' / source.name))
    pairs.append((fast / 'generative_practice_loop_inventory.json', archive / 'context/agent_inventory.json'))
    entries = []
    for source, target in tqdm(pairs, desc='Archive and verify follow-up artifacts'):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        checksum = digest(source); assert checksum == digest(target)
        entries.append({'source': str(source), 'persistent_copy': str(target), 'sha256': checksum, 'bytes': source.stat().st_size})
    result = {'uk_archived': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'research_start_uk': '2026-10-18T23:00:32+01:00', 'scope': 'Follow-up investigation of the README original-reconstruction criterion. Preserves both supported and failed candidates and the explicitly corrected size-query-access run.', 'primary_document': str(root / 'tex/practice_loop_ring.tex'), 'synthesis_document': str(root / 'tex/archive/practice_loop.tex'), 'persistent_root': str(archive), 'file_count': len(entries), 'bytes': sum(x['bytes'] for x in entries), 'python': '/var/tmp/jbauer/venvs/generative-learning/bin/python', 'theory_python': '/nfs/nhome/live/jbauer/miniconda3/bin/python', 'reproduction': {'core': str(root / 'experiments/practice_loop_ring.py'), 'confirmation_arguments': '--seeds 32 --seed-start 86200 --steps 3000 --output NEW_EMPTY_OUTPUT_DIRECTORY', 'secondary_sources': 'Every secondary run saves its own protocol and source snapshot plus imported shared source where applicable. Secondary scripts may load earlier runs by their FAST_CACHE_DIR paths; preserve the saved directory layout. Some scripts use the scientific Python environment noted above for scipy.', 'exact_snapshot': 'Inspect the run directory and its source imports before reproducing. Some runs save run_source.py and shared_source.py: copy them into a new directory under the original executable filename and imported module name. Other runs preserve original filenames and several helper modules; copy all required saved helpers together. Additional dependencies are listed by source hashes or imports and are retained elsewhere in this archive. Preserve earlier-run input paths or redirect them explicitly. The canonical repository code may include later analysis or access corrections, so use the matching run snapshots for the original run and a separate output location. Run outputs are not deleted.', 'verification': str(fast / 'generative_practice_loop_build/verification.json'), 'query_access_correction': 'sizes/ is the original externally truth-filtered task catalogue. sizes_learned_queries/ is the corrected canonical finite-memory test. Both are retained; the report uses the latter.'}, 'files': entries}
    manifest = archive / 'manifest.json'
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    local = fast / 'generative_practice_loop_archive'; local.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, local / 'manifest.json')
    (local / 'file_paths.txt').write_text('\n'.join(x['source'] for x in entries) + '\n')
    print(json.dumps({'manifest': str(manifest), 'local_manifest': str(local / 'manifest.json'), 'files': len(entries), 'bytes': result['bytes']}, indent=2))


if __name__ == '__main__': main()
