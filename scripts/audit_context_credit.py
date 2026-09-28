"""Trace the unchanged credit rule on the fixed integration training stream."""
import json
import math
from pathlib import Path

import torch

from fly_connectome.vision import load_legacy_default
from run_local_motion_dynamics import KINDS, visual_episode
from run_zero_shot_visual_transfer import corrupt_events


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    model = load_legacy_default()
    model.dynamics, model.calibration = {}, {}
    generator = torch.Generator().manual_seed(291027)
    updates = []
    observe = model.state.memory.observe
    anchors = {}
    before = []

    def record(key, value):
        identity = next(i for i, k in model.state.keys.items() if k is key)
        position, velocity, seen, confirmed, hits = before[identity]
        anchor = anchors.get(identity)
        updates.append(dict(scene=index, kind=KINDS[index % 6], sample=t,
            previous_winner_seen=seen, last_actual_seen=None if anchor is None else anchor[2],
            previously_confirmed=confirmed, previous_hits=hits,
            previous_position=position.tolist(), previous_velocity=velocity.tolist(),
            correction=value.tolist(), current_position=model.state.tracker.slots[identity]['position'].tolist()))
        observe(key, value)

    model.state.memory.observe = record
    for index in range(192):
        angle = float(torch.rand((), generator=generator))*2*math.pi
        scale = .6+float(torch.rand((), generator=generator))
        phase = int(torch.randint(8, (), generator=generator))
        case, _ = visual_episode(KINDS[index % 6], phase,
            'dot' if (index//6) % 2 else 'square', angle=angle, scale=scale,
            background=bool((index//6) % 2))
        events = corrupt_events(case['events'], seed=292000+index, dropout=.15, false_rate=.0005)
        model.reset_state()
        anchors = {}
        for t, (event, binary) in enumerate(zip(events, case['visible'])):
            image = (.25+.05*math.sin(t/9)+.35*binary.float()+
                     .02*torch.randn(binary.shape, generator=generator)).clamp(0, 1)
            before = [(s['position'].clone(), s['velocity'].clone(), s['last_seen'],
                       s['confirmed'], s['hypotheses'][0]['hits'])
                      for s in model.state.tracker.slots]
            model.step(event, image, learn=True)
            for i, slot in enumerate(model.state.tracker.slots):
                if slot['last_seen'] == t:
                    anchors[i] = (slot['position'].clone(), slot['velocity'].clone(), t)
    result = dict(updates=len(updates), winner_anchor_mismatches=sum(
        r['previous_winner_seen'] != r['last_actual_seen'] for r in updates), rows=updates)
    Path('docs/experiments/2026-09-27-context-credit-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
