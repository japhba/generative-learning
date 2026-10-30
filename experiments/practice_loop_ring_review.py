"""Check the final ring prose's identity, reflection, and local-file references."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from dotenv import load_dotenv
import numpy as np

import practice_loop_ring_audit as audit


def main():
    load_dotenv(Path.home() / '.env')
    workspace = Path(__file__).resolve().parents[1]
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring'
    out = base.parent / 'generative_practice_loop_ring_audit'
    reference = json.loads((base / 'analysis/summary.json').read_text())
    summary = json.loads((base / 'state_reconstruction/summary.json').read_text())
    data = np.load(base / 'state_reconstruction/trajectories.npz')
    identity = {}
    for name, expected in reference['identity_reconstruction_diagnostics'].items():
        phase = data[f'angles_{name}'][-1]
        embedding = np.stack([np.cos(phase), np.sin(phase)], axis=1)
        decoder, bias = data[f'decoder_{name}'], data[f'bias_{name}']
        logits = decoder @ embedding + bias[:, :, None]
        prediction = logits.argmax(axis=1)
        accuracy = np.mean(prediction == np.arange(12)[None], axis=1)
        maximum = logits.max(axis=1)
        lognormalizer = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
        nll = np.mean(lognormalizer - logits[:, np.arange(12), np.arange(12)])
        assert audit.error(accuracy, expected['identity_accuracy_by_seed']) == 0
        assert abs(nll - summary['final'][name][2]) < 1e-13
        identity[name] = {'accuracy': float(accuracy.mean()), 'all_worlds_perfect': bool(np.all(accuracy == 1)), 'nll': float(nll), 'decoder_norm': float(np.linalg.norm(decoder, axis=(1, 2)).mean()), 'future_success': summary['final'][name][0]}
    main = np.load(base / 'confirmation/trajectories.npz')
    phase = main['angles_practice'][-1]
    original = np.stack([np.cos(phase), np.sin(phase)], axis=1)
    reflected = np.stack([np.cos(-phase), np.sin(-phase)], axis=1)
    original_svd, reflected_svd = np.linalg.svd(original, full_matrices=False), np.linalg.svd(reflected, full_matrices=False)
    projector = original_svd[2].transpose(0, 2, 1) @ original_svd[2]
    reflected_projector = reflected_svd[2].transpose(0, 2, 1) @ reflected_svd[2]
    reflection = {'singular_value_max_error': audit.error(original_svd[1], reflected_svd[1]), 'right_subspace_projector_max_error': audit.error(projector, reflected_projector), 'gram_matrix_max_error': audit.error(original.transpose(0, 2, 1) @ original, reflected.transpose(0, 2, 1) @ reflected)}
    assert max(reflection.values()) < 1e-13
    delta = phase[:, None, :] - phase[:, :, None]
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    reflection['probability_complement_error'] = audit.error(sigmoid(-6 * np.sin(delta)), 1 - sigmoid(6 * np.sin(delta)))
    for name, reflected_phase in [('reflect_final', -phase), ('reflect_ideal_winding_one', -main['target'])]:
        success = audit.enumerate_routes(reflected_phase, main['truth'], main['queries'], 6, 6.)
        expected = reference['orientation_interventions'][name]
        assert audit.error(success, expected['by_seed']) < 2e-15
        reflection[name] = {'mean_success': float(success.mean()), 'independent_enumeration_max_error': audit.error(success, expected['by_seed'])}
    documents = [workspace / 'tex/practice_loop_ring.tex', workspace / 'tex/archive/practice_loop.tex']
    paths, hashes = [], {}
    for document in documents:
        text = document.read_text()
        hashes[str(document)] = hashlib.sha256(document.read_bytes()).hexdigest()
        for command, target in re.findall(r'\\(href|input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', text):
            if target.startswith(('https:', 'http:')): continue
            resolved = document.parent / ('figures' if command == 'includegraphics' else '') / target
            assert resolved.is_file(), (document, command, resolved)
            paths.append({'document': str(document), 'command': command, 'target': str(resolved)})
    result = {'identity_diagnostics': identity, 'reflection': reflection, 'verified_local_references': paths, 'document_sha256': hashes, 'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), Path(audit.__file__), base / 'analysis/run_source.py']}}
    (out / 'final_review.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(__file__, out / 'review_source.py')
    print(json.dumps({'identity': identity, 'reflection': reflection, 'local_references_checked': len(paths)}, indent=2))


if __name__ == '__main__': main()
