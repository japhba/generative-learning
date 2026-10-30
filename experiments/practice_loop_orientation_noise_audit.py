"""Coupled-RNG and path-enumeration checks for the reward-alignment control."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np

import practice_loop_bayesian as bayes
import practice_loop_orientation_audit as orientation
import practice_loop_orientation_noise as control


class ForcedSigns:
    def __init__(self, seed, sign): self.rng, self.sign = np.random.default_rng(seed), sign
    def random(self, shape): return self.rng.random(shape)
    def integers(self, high, size): return np.full(size, (1 + self.sign) // 2) if len(size) == 3 else self.rng.integers(high, size=size)


def main():
    load_dotenv(Path.home() / '.env')
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/orientation'
    out = base / 'noise_control/audit'; out.mkdir(parents=True, exist_ok=True)
    source = np.load(base / 'trajectories.npz')
    data = np.load(base / 'noise_control/trajectories.npz')
    worlds, posterior, queries, initial = [source[key] for key in ['worlds', 'posterior', 'queries', 'initial']]
    cumulative = posterior.cumsum(axis=1); cumulative[:, -1] = 1
    perturbation = initial + np.random.default_rng(1097321).normal(0, .3, initial.shape)
    reference = bayes.sampled(perturbation, worlds, cumulative, queries, np.random.default_rng(1097322), 2., 3, 64)
    errors = {}
    for sign in [1, -1]:
        candidate = control.scrambled(perturbation, worlds, cumulative, queries, ForcedSigns(1097322, sign), 2., 3, 64)
        errors[str(sign)] = float(abs(candidate - sign * reference).max())
        assert errors[str(sign)] == 0
    replay = {}
    for name, seed in [('practice', 1097223), ('sign_scrambled', 1097224)]:
        theta = initial.copy(); rng = np.random.default_rng(seed)
        sampler = bayes.sampled if name == 'practice' else control.scrambled
        for _ in range(100): theta += .1 * sampler(theta, worlds, cumulative, queries, rng, 2., 3, 64)
        replay[name] = float(abs(theta - data[f'angles_{name}'][1]).max())
        assert replay[name] == 0
    evaluation = {}
    for name in replay:
        exact, _ = orientation.posterior_success(data[f'angles_{name}'][-1], worlds, queries)
        evaluation[name] = float(abs(exact - data[f'curves_{name}'][-1, :, 0]).max())
        assert evaluation[name] < 1e-14
    result = {'uk_verified': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'forced_shared_sign_exact_update_errors': errors, 'first100_update_replay_errors': replay, 'independent_final_eight_path_enumeration_errors': evaluation, 'proof': 'For each paired update h, the control produces epsilon h with an independent fair Rademacher epsilon. E[epsilon h|h]=0 and (epsilon h)(epsilon h)^T=h h^T. This matches pair raw second moments at a fixed policy, not necessarily batch covariances away from zero mean or realized noise across diverging policies.', 'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), Path(control.__file__), Path(bayes.__file__), Path(orientation.__file__)]}}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(__file__, out / 'source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
