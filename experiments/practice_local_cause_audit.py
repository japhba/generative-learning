"""Independent checks of the gradual drawing generator and saved trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m
import practice_drawing_sampled as s
import practice_local_cause as local
from practice_drawing_fixed_posterior import complete_q


def independent_gradient(logits, counts):
    k = logits.shape[-1]
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    categorical = shifted / shifted.sum(axis=-1, keepdims=True)
    coefficients = .98 * np.eye(k) + .02 / k
    numerator = coefficients * shifted[..., None, :]
    numerator /= numerator.sum(axis=-1, keepdims=True)
    return np.sum(counts[..., :, None] * (numerator - categorical[..., None, :]), axis=-2)


def check_step(a, b, acount, bcount, rate):
    au, bu = local.logits_from_probability(a), local.logits_from_probability(b)
    ag, bg = local.local_gradient(au, acount), local.local_gradient(bu, bcount)
    independent_error = max(np.max(abs(ag - independent_gradient(au, acount))),
                            np.max(abs(bg - independent_gradient(bu, bcount))))
    assert independent_error < 1e-12
    assert np.max(abs(acount.sum(axis=-1) - 1)) < 1e-10
    assert np.max(bcount.sum(axis=-1)) <= 3 + 1e-10
    next_a, next_b = local.probabilities(au + rate * ag), local.probabilities(bu + rate * bg)
    gain = complete_q(next_a, next_b, acount, bcount) - complete_q(a, b, acount, bcount)
    square_norm = np.sum(ag ** 2) + np.sum(bg ** 2)
    guaranteed_gain = rate * (1 - .75 * rate) * square_norm
    assert gain >= guaranteed_gain - 1e-11
    assert np.max(abs(next_a.sum(axis=-1) - 1)) < 1e-12
    assert np.max(abs(next_b.sum(axis=-1) - 1)) < 1e-12
    return dict(independent_gradient_error=float(independent_error), conditional_gain=float(gain),
                theoretical_minimum_gain=float(guaranteed_gain), gradient_squared_norm=float(square_norm))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.runs / "protocol.json").read_text())
    rate = protocol["arguments"]["rate"]
    rows = []
    for index in tqdm(range(args.worlds), desc="Independent local-cause audit"):
        data = np.load(args.source / f"run_{index}/world.npz")
        initial = np.load(args.source / f"run_{index}/initial.npz")
        initial_caches = list(np.split(initial["caches"], np.cumsum(initial["cache_sizes"])[:-1]))
        codes, weights, initial_bound = s.posterior(initial_caches, initial["mass"], data["initial_a"], data["initial_b"])
        fixed_acount, fixed_bcount = s.expected_counts(codes, weights, data["initial_a"], data["initial_b"])
        row = dict(world=index, initial=check_step(data["initial_a"], data["initial_b"], fixed_acount, fixed_bcount, rate),
                   initial_inverse_maximum=float(np.max(abs(initial["recognition"]))), conditions={})
        assert row["initial_inverse_maximum"] == 0
        for condition in protocol["arguments"]["conditions"]:
            run = np.load(args.runs / f"run_{index}/{condition}.npz")
            records = run["records"]
            gains = run["conditional_gains"]
            caches = list(np.split(run["caches"], np.cumsum(run["cache_sizes"])[:-1]))
            assert np.all(m.render(s.decode(run["caches"])) == np.repeat(initial["targets"], run["cache_sizes"]))
            assert gains.min() >= -1e-10
            assert np.max(np.diff(records[:, 3])) < 1e-9
            assert np.max(records[:, 1] - records[:, 3]) < 1e-9
            assert np.max(abs(records[:, 4] + records[:, 3] + initial["upper_ll"])) < 1e-9
            assert records[0, 3] == -initial_bound
            if condition == "sleep": assert records[-1, 4] > 0
            assert np.max(abs(run["weights"][0] - np.r_[data["initial_a"].ravel(), data["initial_b"].ravel()])) < 1e-12
            if condition == "fixed_posterior":
                acount, bcount = fixed_acount, fixed_bcount
                assert all(np.array_equal(c, d) for c, d in zip(caches, initial_caches))
                assert np.max(abs(run["recognition"])) == 0
            else:
                codes, weights, final_bound = s.posterior(caches, initial["mass"], run["a"], run["b"])
                acount, bcount = s.expected_counts(codes, weights, run["a"], run["b"])
                assert abs(final_bound + records[-1, 3]) < 1e-9
                if condition == "fixed_cache":
                    assert all(np.array_equal(c, d) for c, d in zip(caches, initial_caches))
                    assert np.max(abs(run["recognition"])) == 0
            slopes = [np.polyfit(np.log(records[-10:, 0]), np.log(records[-10:, column]), 1)[0] for column in [3, 7]]
            condition_row = dict(final_step=int(records[-1, 0]), status=str(run["status"]),
                                 final=check_step(run["a"], run["b"], acount, bcount, rate),
                                 minimum_stored_conditional_gain=float(gains.min()),
                                 maximum_bound_increase_as_loss=float(np.diff(records[:, 3]).max()),
                                 largest_observed_nll_increase=float(np.diff(records[:, 1]).max()),
                                 final_likelihood_improvement_certificate=float(records[-1, 4]),
                                 final_population_kl=float(records[-1, 2]), stopping_slopes=slopes)
            if str(run["status"]) == "stationary variational bound and dream recognition":
                assert max(abs(value) for value in slopes) < s.CONFIG["slope"]
            row["conditions"][condition] = condition_row
        rows.append(row)
    report = dict(rate=rate, guaranteed_coefficient=rate*(1-.75*rate), worlds=rows,
                  stopping_scope="Observed-data variational loss and fresh generated-program inverse cross entropy; population risk is monitored but does not select stopping.",
                  limitation="Saved conditional gains cover every update; the analytic gradient-norm lower bound is independently checked at every world's initial and final parameters.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
