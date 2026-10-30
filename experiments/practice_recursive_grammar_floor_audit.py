"""KKT audit of the exact noise-floor-constrained categorical update."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np

import practice_recursive_grammar_screen as model


def main():
    load_dotenv(Path.home()/'.env')
    output = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_recursive_grammar/floor_audit'
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20102801)
    checks = {}
    for categories in [2, 4]:
        counts = rng.gamma(.2, size=(10000, categories))
        counts /= counts.sum(axis=-1, keepdims=True)
        floor = model.CONFIG['uniform_mass']/categories
        fitted = model.constrained_counts(counts, floor)
        fixed = fitted == floor
        free_count = np.where(fixed, 0., counts).sum(axis=-1)
        multiplier = free_count/(1-floor*fixed.sum(axis=-1))
        gradient = counts/fitted
        free_error = np.max(np.where(fixed, 0., np.abs(gradient-multiplier[:, None])))
        bound_error = np.max(np.where(fixed, gradient-multiplier[:, None], 0.))
        assert free_error < 1e-12
        assert bound_error < 1e-12
        assert fitted.min() >= floor
        assert np.max(np.abs(fitted.sum(axis=-1)-1)) < 1e-12
        initial = (1-model.CONFIG['uniform_mass'])*rng.dirichlet(np.ones(categories), len(counts))+floor
        gains = np.sum(counts*np.log(fitted/initial), axis=-1)
        assert gains.min() > -1e-12
        unconstrained = model.constrained_counts(counts, 0.)
        assert np.max(np.abs(unconstrained-counts)) < 1e-12
        checks[str(categories)] = dict(free_kkt_error=float(free_error), bound_kkt_violation=float(bound_error),
            minimum_gain_over_random_feasible=float(gains.min()), rows_with_active_bound=int(fixed.any(axis=-1).sum()))
    result = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), checks=checks, draws_per_dimension=10000)
    (output/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    shutil.copy2(__file__, output/'source.py'); shutil.copy2(model.__file__, output/'model_source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
