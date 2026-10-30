"""Joint variational improvement need not improve marginal generation risk."""

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


def softmax(x, axis):
    values = np.exp(x-x.max(axis=axis, keepdims=True))
    return values/values.sum(axis=axis, keepdims=True)


def quantities(a, b, target):
    q, r = softmax(a, None), softmax(b, -1)
    marginal = q.sum(axis=1)
    log_ratio = np.log(q/(target[:, None]*r))
    joint_kl = np.sum(q*log_ratio)
    generation_risk = np.sum(target*np.log(target/marginal))
    da = q*(joint_kl-log_ratio)
    db = q-marginal[:, None]*r
    return np.array([joint_kl, generation_risk]), da, db, q, r


def main():
    load_dotenv(Path.home()/'.env')
    out = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_joint_bound_counterexample'
    out.mkdir(parents=True, exist_ok=False)
    # Three observed successes under a Beta(1,1) prior give predictive (4/5,1/5).
    target = np.array([.8, .2])
    q0 = np.array([[.4, .4], [.1, .1]])
    r0 = np.array([[.999, .001], [.5, .5]])
    a, b = np.log(q0), np.log(r0)
    dt, maximum = .05, 4000
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'prior': 'Beta(1,1)', 'original_data': [1, 1, 1], 'target': target.tolist(), 'q0': q0.tolist(), 'r0': r0.tolist(), 'dt': dt, 'maximum_steps': maximum, 'scope': 'Specific joint reverse-KL gradient flow, not a test of every GFlowNet update. Actual future log-loss excess is forward KL(target || generated marginal).'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, out/'source.py')
    initial, da, db, q, _ = quantities(a, b, target)
    dq = q*(da-np.sum(q*da))
    marginal_derivative = dq.sum(axis=1)
    risk_curvature = float(np.sum(marginal_derivative**2/target))
    joint_derivative = float(-np.sum(da**2)-np.sum(db**2))
    assert risk_curvature > 0 and joint_derivative < 0
    epsilon = 1e-5
    numeric = (quantities(a+epsilon*da,b+epsilon*db,target)[0][0]-quantities(a-epsilon*da,b-epsilon*db,target)[0][0])/(2*epsilon)
    assert abs(numeric-joint_derivative) < 1e-8
    records, saved_q, saved_r, steps = [], [], [], []
    for iteration in trange(maximum+1, desc='Joint-bound counterexample'):
        metrics, da, db, q, r = quantities(a, b, target)
        records.append(metrics); saved_q.append(q); saved_r.append(r); steps.append(iteration)
        if iteration == maximum: break
        if iteration > 100 and np.sum(da**2)+np.sum(db**2) < 1e-10: break
        if iteration >= 1000 and iteration % 20 == 0:
            recent = np.array(records[-400::20])
            times = np.array(steps[-400::20])*dt
            if np.all(recent > 0):
                slopes = np.polyfit(np.log(times), np.log(recent), 1)[0]
                if np.max(np.abs(slopes)) < .01: break
        # RK4 integrates both networks at the same rate.
        _, ca, cb, _, _ = quantities(a+dt*da/2,b+dt*db/2,target)
        _, ea, eb, _, _ = quantities(a+dt*ca/2,b+dt*cb/2,target)
        _, ga, gb, _, _ = quantities(a+dt*ea,b+dt*eb,target)
        a += dt*(da+2*ca+2*ea+ga)/6
        b += dt*(db+2*cb+2*eb+gb)/6
    records, steps = np.array(records), np.array(steps)
    assert np.max(np.diff(records[:, 0])) < 1e-10
    assert records[:, 1].max() > .001
    np.savez_compressed(out/'trajectory.npz', records=records, steps=steps, q=saved_q, r=saved_r, target=target)
    peak = int(records[:, 1].argmax())
    summary = protocol | {'initial': initial.tolist(), 'final': records[-1].tolist(), 'initial_marginal_derivative': marginal_derivative.tolist(), 'initial_generation_risk_second_derivative': risk_curvature, 'initial_joint_kl_derivative': joint_derivative, 'directional_derivative_error': float(abs(numeric-joint_derivative)), 'maximum_generation_risk': float(records[peak, 1]), 'time_of_maximum_generation_risk': float(steps[peak]*dt), 'generated_marginal_at_peak': np.array(saved_q)[peak].sum(axis=1).tolist(), 'last_step': int(steps[-1]), 'joint_kl_maximum_recorded_increase': float(np.diff(records[:, 0]).max())}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(steps*dt, records[:, 0]); axes[0].set(ylabel='Joint KL objective')
    axes[1].plot(steps*dt, records[:, 1]); axes[1].set(xlabel='Equal-rate gradient-flow time', ylabel='Generation log-loss excess')
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-joint-bound-counterexample.png'); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
