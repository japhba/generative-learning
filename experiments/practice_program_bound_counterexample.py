"""A perfect reconstruction cache can improve while data likelihood falls.

This is a two-token deterministic-program diagnostic, not a proposed model.
Both tokens are independent Bernoullis. Their XOR renders the observation.
The one-item cache stores 00 for the single observed output 0.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange


def sigmoid(x): return np.exp(-np.logaddexp(0, -x))


def main():
    load_dotenv(Path.home()/'.env')
    output = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_program_bound_counterexample'
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'scope': 'Analytic counterexample to deriving marginal-likelihood monotonicity from an improving finite program-cache bound. Fixed cache {00}, deterministic XOR renderer, independent Bernoulli token prior initialized at .9. Generator follows the exact cache-bound logit gradient; recognition learns the cached cause. Both cached-program reconstruction and eventual improvement are distinct from monotonic data likelihood. This does not refute all MWS variants or eventual learning.'}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, output/'source.py')
    generator = np.full(2, np.log(9.))
    recognizer = np.zeros(4)
    records = []
    rate = .01
    for iteration in trange(2001, desc='Program-cache likelihood diagnostic'):
        p = sigmoid(generator)
        r = np.exp(recognizer-recognizer.max()); r /= r.sum()
        likelihood = p.prod()+(1-p).prod()
        bound = np.log1p(-p).sum()
        records.append([iteration*rate, bound, np.log(likelihood), r[0], *p])
        if iteration == 2000: break
        def flow(u, v):
            rv = np.exp(v-v.max()); rv /= rv.sum()
            return -sigmoid(u), np.eye(4)[0]-rv
        a, b = flow(generator, recognizer)
        c, d = flow(generator+rate*a/2, recognizer+rate*b/2)
        e, f = flow(generator+rate*c/2, recognizer+rate*d/2)
        h, i = flow(generator+rate*e, recognizer+rate*f)
        generator += rate*(a+2*c+2*e+h)/6
        recognizer += rate*(b+2*d+2*f+i)/6
    records = np.array(records)
    assert np.diff(records[:, 1]).min() > 0
    assert np.diff(records[:, 3]).min() > 0
    assert records[1, 2] < records[0, 2]
    assert records[-1, 2] > records[0, 2]
    np.savez_compressed(output/'trajectory.npz', records=records)
    result = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'initial_bound_derivative': 1.62, 'initial_likelihood_derivative': -.1296, 'initial_log_likelihood_derivative': -.1296/.82, 'initial_likelihood': .82, 'minimum_likelihood': float(np.exp(records[:, 2]).min()), 'final_likelihood': float(np.exp(records[-1, 2])), 'cache_reconstruction_error_all_steps': 0}
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(records[:, 0], records[:, 1], label='cache bound')
    axes[0].plot(records[:, 0], records[:, 2], label='actual log likelihood')
    axes[0].set(ylabel='Log probability'); axes[0].legend()
    axes[1].plot(records[:, 0], np.exp(records[:, 2]), label='observed-output probability')
    axes[1].plot(records[:, 0], records[:, 3], label='recognition of cached cause')
    axes[1].set(ylabel='Probability', xlabel='Equal-rate gradient-flow time'); axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-program-bound-counterexample.png'); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
