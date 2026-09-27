"""Matched prefixes with unknowable future visual reidentification outcomes."""

import json
from pathlib import Path

import torch

from fly_connectome.sensor import EventCamera
from streaming_visual_state import LearnedVisualCandidate


def episode(model, *, continued, start, speed, learn):
    model.reset_state()
    camera = EventCamera(1, 32, 64)
    forecast = None
    for t in range(12):
        image = torch.zeros(32, 64, dtype=torch.bool)
        if continued or t < 8:
            image[15:18, start+speed*t:start+speed*t+3] = True
        sparse = camera.observe(image[None])
        event = torch.zeros(2, 32*64)
        event[sparse.on.long(), sparse.pixels] = 1.
        available = not (8 <= t <= 10)
        result = model.step(event.reshape(2, 32, 64) if available else torch.zeros(2, 32, 64),
                            image if available else None, learn=learn)
        if t == 7:
            forecast = next(f for f in result['forecasts'] if f['horizon_samples'] == 4)
    seen = any(e['id'] == forecast['id'] and e['observed'] for e in result['entities'])
    result_row = dict(predicted=forecast['reidentification_probability'], reidentified=seen,
                position=forecast['position'].tolist(),
                counterfactual_support=float(result['support_field'][16, start+speed*11+1]))
    if 'mixture_centers' in forecast:
        result_row.update(mixture_centers=forecast['mixture_centers'].tolist(),
                          mixture_weights=forecast['mixture_weights'].tolist())
    return result_row


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    model = LearnedVisualCandidate()
    for _ in range(16):
        for continued in (True, False):
            episode(model, continued=continued, start=5, speed=1, learn=True)
    pairs = []
    for start in (3, 9, 15, 21):
        for speed in (1, 2):
            pairs.append([episode(model, continued=c, start=start, speed=speed, learn=False)
                          for c in (True, False)])
    rows = [row for pair in pairs for row in pair]
    result = dict(target='same identity visually reidentified at endpoint, not hidden existence',
        heldout_pairs=len(pairs), brier=sum((r['predicted']-r['reidentified'])**2 for r in rows)/len(rows),
        observed_rate=sum(r['reidentified'] for r in rows)/len(rows),
        mean_probability=sum(r['predicted'] for r in rows)/len(rows),
        max_prefix_probability_difference=max(abs(a['predicted']-b['predicted']) for a, b in pairs),
        max_prefix_position_difference=max(float((torch.tensor(a['position'])-torch.tensor(b['position'])).abs().max())
                                           for a, b in pairs), pairs=pairs)
    Path('docs/experiments/2026-09-26-streaming-uncertainty-results.json').write_text(
        json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
