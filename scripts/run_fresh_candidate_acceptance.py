"""Fresh integrated-checkpoint transfer, with all labels confined to scoring."""

import json
import math
from collections import deque
from pathlib import Path

import torch

from run_local_motion_dynamics import KINDS, visual_episode, fixed_displacement
from run_zero_shot_visual_transfer import corrupt_events
from streaming_visual_state import StreamingVisualState
from visual_history_benchmark import SHAPES


@torch.no_grad()
def main(*, checkpoint='checkpoints/m1a5/streaming-candidate.pt',
         out=Path('docs/experiments/2026-09-26-fresh-candidate-acceptance-results.json'),
         phases=(7, 9), angles=(math.pi/6, math.pi/3, 2*math.pi/3), scales=(.75, 1.5),
         candidate=None, entity_dynamics=None):
    torch.set_num_threads(1)
    model = candidate if candidate is not None else StreamingVisualState.load(checkpoint)
    SHAPES.update(L=((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
                  T=((-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)),
                  line=tuple((0, x) for x in range(-2, 3)))
    errors = {kind: {h: {name: [] for name in ('learned', 'velocity', 'mean_velocity', 'lag_two', 'persistence', 'acceleration')}
                    for h in (1, 4, 8)} for kind in KINDS}
    missing = samples = scenes = 0
    for kind in KINDS:
        for shape in ('L', 'T', 'line'):
            for phase in phases:
                for angle in angles:
                    for scale in scales:
                        case, truth = visual_episode(kind, phase, shape, angle=angle, scale=scale,
                                                     background=bool(scenes % 2))
                        events = corrupt_events(case['events'], seed=120000+scenes,
                                                dropout=.15, false_rate=.0005)
                        generator = torch.Generator().manual_seed(130000+scenes)
                        model.reset_state()
                        history = deque(maxlen=5)
                        local_models = {}
                        for t, (event, binary) in enumerate(zip(events, case['visible'])):
                            image = (.25+.05*math.sin(t/9)+.35*binary.float()+
                                     .02*torch.randn(binary.shape, generator=generator)).clamp(0., 1.)
                            state = model.step(event, image)
                            if entity_dynamics is not None:
                                for item in state['entities']:
                                    local = local_models.setdefault(item['id'], entity_dynamics())
                                    local.observe(item['position'] if item['observed'] else None)
                            candidates = [e for e in state['entities'] if e['observed']]
                            entity = min(candidates, key=lambda e: e['id']) if candidates else None
                            history.append(None if entity is None else (entity['id'], entity['position']))
                            eligible = (len(history) == 5 and all(p is not None for p in history)
                                        and len({p[0] for p in history}) == 1)
                            if t < 6:
                                continue
                            if eligible:
                                positions = torch.stack([p[1] for p in history])
                                differences = positions[1:]-positions[:-1]
                            for h, arms in errors[kind].items():
                                if t+h >= len(truth):
                                    continue
                                samples += 1
                                matched = [f for f in state['forecasts'] if eligible and
                                    f['id'] == entity['id'] and f['horizon_samples'] == h and f['evidence_age'] == 0]
                                if entity_dynamics is not None and eligible:
                                    position = local_models[entity['id']].predict(h)
                                    matched = [] if position is None else [dict(position=position)]
                                if not eligible or not matched:
                                    missing += 1
                                    for values in arms.values():
                                        values.append(64.)
                                    continue
                                for name, values in arms.items():
                                    prediction = (matched[0]['position'] if name == 'learned' else
                                        positions[-1]+fixed_displacement(differences, h, name))
                                    values.append(float((prediction-truth[t+h]).norm()))
                        scenes += 1
    mean = lambda values: sum(values)/len(values)
    families = {k: {h: {n: mean(v) for n, v in arm.items()} for h, arm in row.items()}
                for k, row in errors.items()}
    totals = {h: {n: mean([v for kind in KINDS for v in errors[kind][h][n]])
                  for n in errors[KINDS[0]][h]} for h in (1, 4, 8)}
    aggregate_pass = all(totals[h]['learned'] <= .8*min(v for n, v in totals[h].items() if n != 'learned')
                         for h in (4, 8))
    family_pass = all(row[8]['learned'] <= 1+min(v for n, v in row[8].items() if n != 'learned')
                      for row in families.values())
    result = dict(scenes=scenes, samples=samples, missing=missing, means=totals, families=families,
                  aggregate_gate=aggregate_pass, family_gate=family_pass,
                  coverage_gate=1-missing/samples >= .95)
    out.write_text(
        json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
