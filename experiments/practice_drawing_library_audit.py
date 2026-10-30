"""Independent renderer, latent marginalization, local-score and monotonicity checks."""

import itertools
import json
from pathlib import Path

import numpy as np

import practice_drawing_library as m


def scalar_render(actions):
    x, y, direction, ink = 3, 3, 0, {(3, 3)}
    for action in actions:
        if action in (0, 3):
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
            x, y = (x+dx) % 8, (y+dy) % 8
            if action == 0: ink.add((x, y))
        elif action == 1: direction = (direction+1) % 4
        elif action == 2: direction = (direction-1) % 4
    return sum(1 << (8*y+x) for x, y in ink)


def main():
    cat, rng = m.Catalogue(), np.random.default_rng(20103002)
    world = m.world(cat, 20103000)
    a, b = world['initial_a'], world['initial_b']
    code, image = cat.probabilities(a, b)
    selected = rng.integers(len(cat.actions), size=512)
    assert all(scalar_render(cat.actions[i]) == int(cat.canvas[i]) for i in selected)
    assert np.all(cat.canvas[world['canonical']] == cat.images[np.unique(world['observed'])])
    errors = []
    count_a, count_b = np.zeros_like(a), np.zeros_like(b)
    for i in selected:
        marginal, terms = 0., []
        for labels in itertools.product(range(3), repeat=3):
            p = np.prod([a[j, labels[j]]*np.prod(b[labels[j], np.arange(3), cat.actions[i, 3*j:3*j+3]]) for j in range(3)])
            terms.append((labels, p)); marginal += p
        errors.append(abs(marginal-code[i]))
        for labels, p in terms:
            for j, label in enumerate(labels):
                count_a[j, label] += p/marginal
                for k, symbol in enumerate(cat.actions[i, 3*j:3*j+3]): count_b[label, k, symbol] += p/marginal
    fitted_a, fitted_b = m.fit_counts(cat, a, b, selected, np.ones(len(selected)))
    count_error = max(np.max(abs(fitted_a-m.row_normalize(count_a))), np.max(abs(fitted_b-m.row_normalize(count_b))))
    recognition = rng.normal(0, .03, (9, 4, 129))
    _, valid, _, gradient = m.recognize(cat, recognition, cat.canvas[selected], rng, cat.actions[selected], update=True)
    assert valid.all()
    derivative_error = []
    for index in [(0, 0, 0), (2, 2, 55), (5, 1, 91), (8, 3, 128)]:
        high, low = recognition.copy(), recognition.copy()
        high[index] += 1e-5; low[index] -= 1e-5
        hp = m.recognize(cat, high, cat.canvas[selected], rng, cat.actions[selected])[2].mean()
        lp = m.recognize(cat, low, cat.canvas[selected], rng, cat.actions[selected])[2].mean()
        derivative_error.append(abs((hp-lp)/2e-5-gradient[index]))
    draws = m.draw(a, b, rng, 100000)
    frequency = np.bincount(cat.ids[draws], minlength=len(image))/len(draws)
    frequent = image > .001
    sampling_z = np.max(abs(frequency[frequent]-image[frequent])/np.sqrt(image[frequent]*(1-image[frequent])/len(draws)))
    result = dict(renderer_programs=512, program_normalization_error=float(abs(code.sum()-1)),
        image_normalization_error=float(abs(image.sum()-1)), enumerated_motif_error=float(max(errors)),
        coactivity_counts_error=float(count_error), recognition_gradient_error=float(max(derivative_error)),
        ancestral_sampling_max_z=float(sampling_z))
    assert max(errors) < 1e-14 and count_error < 1e-12 and max(derivative_error) < 1e-8 and sampling_z < 6
    path = m.CACHE/'audit'
    path.mkdir(parents=True, exist_ok=False)
    (path/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
