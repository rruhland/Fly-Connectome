"""Bounded streaming throughput, grayscale sensor transfer and uncertainty audit."""

import json
import time
from pathlib import Path

import torch

from camera_state_audit import crossing_scene
from streaming_visual_state import StreamingVisualState


@torch.no_grad()
def main(*, state_type=StreamingVisualState,
         out=Path('docs/experiments/2026-09-26-streaming-camera-transfer-results.json')):
    torch.set_num_threads(1)
    results = {}
    for condition in ('binary', 'low_contrast', 'illumination', 'outage'):
        retained, frames, elapsed, max_slots, max_pending = 0, 0, 0., 0, 0
        residuals, radii, outcomes, probabilities = [], [], [], []
        model = state_type(height=64, width=64)
        for repeat in range(2):
            for axis in ('horizontal', 'vertical', 'diagonal'):
                images, events, truth = crossing_scene(axis, 2*repeat, bool(repeat))
                model.reset_state()
                ids, pending = [], {}
                for t, (image, event) in enumerate(zip(images, events)):
                    if condition == 'low_contrast':
                        image = .3+.3*image
                    elif condition == 'illumination':
                        image = (.2+.1*torch.sin(torch.tensor(t/8.)))+.5*image
                    elif condition == 'outage' and 16 <= t < 21:
                        image, event = None, torch.zeros_like(event)
                    start = time.perf_counter()
                    state = model.step(event, image, learn=True)
                    elapsed += time.perf_counter()-start
                    frames += 1
                    max_slots = max(max_slots, len(model.state.tracker.slots))
                    max_pending = max(max_pending, sum(map(len, model.pending.values())))
                    entities = {e['id']: e for e in state['entities']}
                    if t == 6:
                        claimed = set()
                        for target in truth[t]:
                            candidates = [(float((e['position']-target).norm()), identity)
                                          for identity, e in entities.items() if identity not in claimed]
                            distance, identity = min(candidates) if candidates else (64., None)
                            ids.append(identity if distance <= 2 else None)
                            claimed.add(identity)
                    if image is not None:
                        for forecast in pending.pop(t, []):
                            entity = entities.get(forecast['id'])
                            seen = entity is not None and entity['observed']
                            outcomes.append(float(seen))
                            probabilities.append(forecast['reidentification_probability'])
                            if seen and forecast['conditional_radius_90'] is not None:
                                residuals.append(float((entity['position']-forecast['position']).norm()))
                                radii.append(forecast['conditional_radius_90'])
                    for forecast in state['forecasts']:
                        pending.setdefault(t+forecast['horizon_samples'], []).append(forecast)
                retained += sum(identity in entities and
                                float((entities[identity]['position']-target).norm()) <= 4
                                for identity, target in zip(ids, truth[-1]))
        results[condition] = dict(identities=12, retained=retained, frames=frames,
            seconds=elapsed, frames_per_second=frames/elapsed,
            max_slots=max_slots, max_pending=max_pending,
            reidentification_brier=(sum((a-b)**2 for a, b in zip(outcomes, probabilities))/len(outcomes)
                                    if outcomes else None),
            conditional_radius_coverage=(sum(a <= b+1e-6 for a, b in zip(residuals, radii))/len(radii)
                                         if radii else None))
        print(condition, results[condition], flush=True)
    out.write_text(
        json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
