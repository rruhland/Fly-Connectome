"""Frozen causal memory ceiling for local four-frame event support."""

import json
import time
from collections import deque
from pathlib import Path

import torch

from forecast_reachability import reached_targets
from gap_timing_transfer import MODEL_OUT
from generic_native_cadence import scene_events
from run_event_centric_competition import train_state
from run_observation_horizon_forecast import sensory_sequence
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-memory-reachability-results.json')


def empty():
    return dict(targets=0, quiet_targets=0,
                current=0, held_4=0, held_8=0,
                quiet_current=0, quiet_held_4=0, quiet_held_8=0,
                source_free=0, held_4_source_free=0,
                held_8_source_free=0)


@torch.no_grad()
def audit(model, cases):
    row = empty()
    for events in cases:
        model.reset_state()
        history = deque(maxlen=8)
        pending = deque()
        for t, (event, code, surface) in enumerate(sensory_sequence(events)):
            if len(pending) == 4:
                origin, snapshots, old_event = pending.popleft()
                if origin >= 4:
                    target_count = int(event.sum())
                    row['targets'] += target_count
                    for name, state in snapshots.items():
                        hits, _ = reached_targets(state, event)
                        row[name] += hits
                        if not (state.max(0).values >= .5).any():
                            row[name.replace('current', 'source_free').replace(
                                'held_4', 'held_4_source_free').replace(
                                'held_8', 'held_8_source_free')] += target_count
                        if not old_event.any():
                            row['quiet_'+name] += hits
                    if not old_event.any():
                        row['quiet_targets'] += target_count
            model.step((code, surface))
            history.append(model.state.clone())
            recent = list(history)
            snapshots = dict(current=model.state.clone(),
                             held_4=torch.stack(recent[-4:]).amax(0),
                             held_8=torch.stack(recent).amax(0))
            pending.append((t, snapshots, event))
    return {**row, **{name+'_ceiling': row[name]/max(row['targets'], 1)
                    for name in ('current', 'held_4', 'held_8')},
            **{'quiet_'+name+'_ceiling': row['quiet_'+name]/max(
                row['quiet_targets'], 1)
               for name in ('current', 'held_4', 'held_8')}}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    model = train_state(saved['models']['learned_split'].code.motion,
                        [scene_events(seed) for seed in range(64)])
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)],
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    result = {name: audit(model, cases) for name, cases in groups.items()}
    result['native_by_seed'] = {str(seed): audit(model, [events])
                                for seed, events in zip(SEEDS,
                                                        groups['native'])}
    result['elapsed_seconds'] = time.perf_counter()-start
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {key: round(row[key], 3)
                             for key in ('current_ceiling', 'held_4_ceiling',
                                         'held_8_ceiling',
                                         'quiet_current_ceiling',
                                         'quiet_held_4_ceiling',
                                         'quiet_held_8_ceiling')}
                      for name, row in result.items()
                      if name in ('generic_heldout', 'native')}), flush=True)


if __name__ == '__main__':
    main()
