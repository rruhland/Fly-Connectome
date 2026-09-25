"""Frozen generic-calibrated event budget for four-frame visual forecasts."""

import json
import time
from collections import deque
from pathlib import Path

import torch

from event_budget_probe import select_by_mass
from gap_timing_transfer import MODEL_OUT
from generic_native_cadence import scene_events
from run_event_centric_competition import train_heads, train_state
from run_observation_horizon_forecast import (empty_score, finish,
                                              score_event, sensory_sequence)
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-event-budget-results.json')
ARMS = ('competitive', 'independent', 'shuffled_credit')


@torch.no_grad()
def probe(model, heads, cases, *, gains=None):
    result = {name: dict(mass=0., target_events=0, frames=0,
                         active_mass=0., active_frames=0,
                         quiet_mass=0., quiet_frames=0,
                         score=empty_score()) for name in ARMS}
    for events in cases:
        model.reset_state()
        for head in heads.values():
            head.reset_state()
        pending = deque()
        for t, (event, code, surface) in enumerate(sensory_sequence(events)):
            if len(pending) == 4:
                origin, predictions, previous = pending.popleft()
                if origin >= 4:
                    for name, prediction in predictions.items():
                        row = result[name]
                        mass = float(prediction.sum())
                        row['mass'] += mass
                        row['target_events'] += int(event.sum())
                        row['frames'] += 1
                        if event.any():
                            row['active_mass'] += mass
                            row['active_frames'] += 1
                        else:
                            row['quiet_mass'] += mass
                            row['quiet_frames'] += 1
                        if gains is not None:
                            selected = select_by_mass(
                                prediction, gain=gains[name])
                            score_event(row['score'], selected, event,
                                        previous, lead=4)
            model.step((code, surface))
            predictions = {name: head.step(model.state, event)
                           for name, head in heads.items()}
            pending.append((t, predictions, event))
    for row in result.values():
        row['mean_mass'] = row['mass']/max(row['frames'], 1)
        row['mean_target_events'] = row['target_events']/max(row['frames'], 1)
        row['mean_active_mass'] = row['active_mass']/max(row['active_frames'], 1)
        row['mean_quiet_mass'] = row['quiet_mass']/max(row['quiet_frames'], 1)
        row['score'] = finish(row['score'])
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    training = [scene_events(seed) for seed in range(64)]
    model = train_state(saved['models']['learned_split'].code.motion,
                        training)
    all_heads = train_heads(model, training)
    heads = all_heads['four_frame']
    calibration = probe(model, heads, training)
    gains = {name: row['target_events']/max(row['mass'], 1e-6)
             for name, row in calibration.items()}
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)],
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    scores = {name: probe(model, heads, cases, gains=gains)
              for name, cases in groups.items()}
    result = dict(calibration=calibration, gains=gains,
                  scores=scores,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(gains={name: round(gain, 3)
                                  for name, gain in gains.items()},
        scores={group: {name: dict(
            f1=round(row['score']['f1'], 3),
            quiet_alarms=round(row['score']['quiet_alarms_per_frame'], 3),
            active_mass=round(row['mean_active_mass'], 3),
            quiet_mass=round(row['mean_quiet_mass'], 3))
            for name, row in arms.items()} for group, arms in scores.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
