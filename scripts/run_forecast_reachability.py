"""Audit whether causal local sources can reach four-frame event targets."""

import json
import time
from collections import deque
from pathlib import Path

import torch

from correlation_latent_robustness import make_robust_cases
from forecast_reachability import reached_targets
from gap_timing_transfer import MODEL_OUT
from generic_native_cadence import scene_events
from run_event_centric_competition import train_state
from run_observation_horizon_forecast import sensory_sequence
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-forecast-reachability-results.json')


def empty():
    return dict(frames=0, future_events=0, quiet_origin_events=0,
                state_reached=0, observed_reached=0, camera_reached=0,
                quiet_state_reached=0, quiet_observed_reached=0,
                quiet_camera_reached=0, source_sites=0,
                observed_sites=0, camera_sites=0,
                source_free_frames=0, source_free_events=0)


@torch.no_grad()
def audit(model, cases, *, radius=2):
    row = empty()
    for events in cases:
        model.reset_state()
        pending = deque()
        for t, (event, code, surface) in enumerate(sensory_sequence(events)):
            if len(pending) == 4:
                origin, state, observed, old_event = pending.popleft()
                if origin >= 4:
                    state_hits, targets = reached_targets(state, event,
                                                          radius=radius)
                    observed_hits, _ = reached_targets(observed, event,
                                                        radius=radius)
                    camera_hits, _ = reached_targets(old_event, event,
                                                      radius=radius)
                    row['frames'] += 1
                    row['future_events'] += targets
                    row['state_reached'] += state_hits
                    row['observed_reached'] += observed_hits
                    row['camera_reached'] += camera_hits
                    count = int((state.max(0).values >= .5).sum())
                    row['source_sites'] += count
                    if count == 0:
                        row['source_free_frames'] += 1
                        row['source_free_events'] += targets
                    row['observed_sites'] += int((observed.max(0).values >= .5).sum())
                    row['camera_sites'] += int(old_event.any(0).sum())
                    if not old_event.any():
                        row['quiet_origin_events'] += targets
                        row['quiet_state_reached'] += state_hits
                        row['quiet_observed_reached'] += observed_hits
                        row['quiet_camera_reached'] += camera_hits
            model.step((code, surface))
            pending.append((t, model.state.clone(), model.observed.clone(),
                            event))
    return {**row,
            'state_recall_ceiling': row['state_reached']/max(
                row['future_events'], 1),
            'observed_recall_ceiling': row['observed_reached']/max(
                row['future_events'], 1),
            'camera_recall_ceiling': row['camera_reached']/max(
                row['future_events'], 1),
            'quiet_state_ceiling': row['quiet_state_reached']/max(
                row['quiet_origin_events'], 1),
            'mean_source_sites': row['source_sites']/max(row['frames'], 1)}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    training = [scene_events(seed) for seed in range(64)]
    model = train_state(saved['models']['learned_split'].code.motion,
                        training)
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)],
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    for family, _, events in make_robust_cases():
        if family in ('independent', 'crossing'):
            groups.setdefault(family, []).append(events)
    result = {name: audit(model, cases) for name, cases in groups.items()}
    result['radius_sweep'] = {
        str(radius): {name: audit(model, groups[name], radius=radius)
                      for name in ('generic_heldout', 'native')}
        for radius in (4, 6, 8, 16)}
    result['native_by_seed'] = {str(seed): audit(model, [events])
                                for seed, events in zip(SEEDS,
                                                        groups['native'])}
    result['elapsed_seconds'] = time.perf_counter()-started
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: dict(
        state=round(row['state_recall_ceiling'], 3),
        observed=round(row['observed_recall_ceiling'], 3),
        camera=round(row['camera_recall_ceiling'], 3),
        quiet_state=round(row['quiet_state_ceiling'], 3),
        targets=row['future_events'])
        for name, row in result.items()
        if name in ('generic_heldout', 'native', 'independent', 'crossing')}),
        flush=True)


if __name__ == '__main__':
    main()
