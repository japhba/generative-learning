"""Independent algebra, sampling, reconstruction, and saved-trajectory checks."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from practice_latent_channel import sampled_step


def main():
    load_dotenv(Path.home()/'.env')
    out = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_latent_channel'
    saved = np.load(out/'worlds.npz')
    mu, w0, k = saved['mean'], saved['w0'], saved['noise']
    u, y = saved['commands'], saved['observations']
    posterior_from_raw = np.linalg.solve(u.T@u+.1*np.eye(4), u.T@y.T).T
    assert np.allclose(mu, posterior_from_raw, atol=1e-13)
    ideal_w = mu/np.linalg.norm(mu, axis=-1, keepdims=True)
    initial_recon = np.mean((u[None]-np.einsum('nd,sd,se->sne', u, w0, w0))**2, axis=(1, 2))
    final_recon = np.mean((u[None]-np.einsum('nd,sd,se->sne', u, ideal_w, ideal_w))**2, axis=(1, 2))
    assert np.allclose(initial_recon, final_recon, atol=1e-14)
    a = np.load(out/'joint_mean_field.npz')
    signal = np.einsum('tsd,sd->ts', a['w'], mu)
    assert np.min(1-a['r']*signal) > 0
    assert np.max(np.diff(a['generation_mse'], axis=0)) < 1e-12
    assert np.max(np.diff(a['joint_mse'], axis=0)) < 1e-11
    assert np.max(np.abs(np.linalg.norm(a['w'], axis=-1)-1)) < 1e-12
    rng = np.random.default_rng(730219)
    gamma = .1; action_var = .03; env_var = .1
    mean = np.array([[1., -.6, .2, .8]])
    w = np.array([[.3, -.4, .5, .7]]); w /= np.linalg.norm(w)
    r = np.array([.2]); s = float(w[0]@mean[0])
    noise = env_var+gamma+action_var*(np.sum(mean**2)+4*gamma)
    expected_w = r[0]*(1-r[0]*s)*(mean[0]-s*w[0])
    expected_r = (1-r[0]*s)*s-noise*r[0]
    samples = []
    for _ in tqdm(range(80), desc='Independent Monte Carlo gradient audit'):
        wn, rn = sampled_step(w, r, mean, gamma, action_var, env_var, 4096, 1e-6, rng, False)
        samples.append(np.r_[((wn-w)/1e-6)[0], (rn-r)/1e-6])
    samples = np.array(samples)
    expected = np.r_[expected_w, expected_r]
    zscores = (samples.mean(axis=0)-expected)/(samples.std(axis=0, ddof=1)/np.sqrt(len(samples)))
    assert np.max(np.abs(zscores)) < 5, zscores
    def f(v, gain):
        signal = float(v@mean[0])
        return .5*((1-gain*signal)**2+noise*gain**2)
    h = 1e-6
    plus = w[0]+h*expected_w; plus /= np.linalg.norm(plus)
    minus = w[0]-h*expected_w; minus /= np.linalg.norm(minus)
    slope = (f(plus,r[0]+h*expected_r)-f(minus,r[0]-h*expected_r))/(2*h)
    expected_slope = -np.sum(expected_w**2)-expected_r**2
    assert abs(slope-expected_slope) < 1e-9
    frozen = np.load(out/'frozen_recognizer.npz')
    s0 = np.abs(np.sum(w0*mu, axis=-1)); strength = np.linalg.norm(mu, axis=-1)
    fixed_gain = s0/(s0*s0+k)
    limiting_signal = np.minimum(strength, 1/fixed_gain)
    predicted_frozen = k/(k+limiting_signal**2)
    optimum = k/(k+strength**2)
    matched_time = float(frozen['step'][-1]*.01)
    joint = np.load(out/'joint_sampled.npz')
    matched_index = np.flatnonzero(joint['step'] == frozen['step'][-1])[0]
    paired = frozen['generation_mse'][-1]-joint['generation_mse'][matched_index]
    draws = rng.integers(0, len(paired), size=(20000,len(paired)))
    ci = np.quantile(paired[draws].mean(axis=1), [.025,.975])
    summary = {'sampled_gradient_max_abs_zscore': float(np.max(np.abs(zscores))), 'sampled_gradient_expected': expected.tolist(), 'sampled_gradient_estimated': samples.mean(axis=0).tolist(), 'joint_gradient_directional_derivative_error': float(abs(slope-expected_slope)), 'mean_field_generation_max_increase': float(np.diff(a['generation_mse'], axis=0).max()), 'mean_field_joint_max_increase': float(np.diff(a['joint_mse'], axis=0).max()), 'posterior_raw_reconstruction_bypass_error': float(np.abs(mu-posterior_from_raw).max()), 'command_autoencoder_reconstruction_mse_before': float(initial_recon.mean()), 'command_autoencoder_reconstruction_mse_after': float(final_recon.mean()), 'frozen_recognizer_analytic_limit_mse': float(predicted_frozen.mean()), 'frozen_recognizer_observed_mse': float(frozen['generation_mse'][-1].mean()), 'frozen_recognizer_suboptimal_worlds_predicted': int(np.sum(predicted_frozen>optimum+1e-8)), 'matched_time': matched_time, 'frozen_minus_joint_generation_mse': float(paired.mean()), 'paired_world_bootstrap_95ci': ci.tolist(), 'scope': 'The command autoencoder is exactly at its global optimum throughout. However, the original input/output ridge fit followed by normalization is a successful direct-data reconstruction bypass. This candidate does not prove universal necessity of practice.'}
    (out/'audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
