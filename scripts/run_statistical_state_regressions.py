"""Noise and binding gates for the statistical observation stage."""

import json
import time
from pathlib import Path

import torch

from camera_state_audit import crossing_scene, score_scene
from statistical_visual_state import StatisticalVisualCandidate, StatisticalFrameState
from streaming_visual_state import StreamingVisualState


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    source = StreamingVisualState.load('checkpoints/m1a5/streaming-candidate.pt')
    model = StatisticalVisualCandidate(height=64, width=64)
    generator = torch.Generator().manual_seed(151001)
    counts, peak = [], 0
    start = time.perf_counter()
    for _ in range(600):
        image = .4+.02*torch.randn(64, 64, generator=generator)
        event = (torch.rand(2, 64, 64, generator=generator) < .0005).float()
        result = model.step(event, image, learn=True)
        counts.append(sum(e['observed'] for e in result['entities']))
        peak = max(peak, len(model.state.tracker.slots))
    results = dict(blank=dict(frames=600, max_confirmed=max(counts),
        mean_confirmed=sum(counts)/len(counts), max_slots=peak,
        fps=600/(time.perf_counter()-start)), binding={})
    for condition in ('clean', 'image_noise', 'outage'):
        rows = [score_scene(source.observer, source.state.memory,
                    crossing_scene(axis, offset, background), condition, 160000+index,
                    state_type=StatisticalFrameState)
                for index, (axis, offset, background) in enumerate(
                    (a, o, b) for a in ('horizontal', 'vertical', 'diagonal')
                    for o in (-4, 0, 4) for b in (False, True))]
        results['binding'][condition] = dict(
            retained=sum(sum(r['identity_retained']) for r in rows), total=36,
            mean_error=sum(sum(r['center_errors']) for r in rows)/36,
            mean_extra=sum(r['extra_moving_hypotheses'] for r in rows)/len(rows),
            outage_hits=sum(sum(r['outage_hits']) for r in rows),
            outage_targets=sum(len(r['outage_hits']) for r in rows))
    Path('docs/experiments/2026-09-26-statistical-state-regressions-results.json').write_text(
        json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
