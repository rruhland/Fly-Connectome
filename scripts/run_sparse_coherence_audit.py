"""Frozen support/activity test for causal local continuity and inhibition."""

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
from sparse_coherent_state import SparseCoherentState


OUT = Path('docs/experiments/2026-09-25-sparse-coherence-audit-results.json')
ARMS = dict(competition=SparseCoherentState(decay=0., radius=1),
            continuity=SparseCoherentState(decay=.9, radius=0),
            coherent=SparseCoherentState(decay=.9, radius=1))


def empty():
    return dict(targets=0, quiet_targets=0, frames=0, quiet_frames=0,
                arms={name: dict(reached=0, quiet_reached=0,
                                source_sites=0, quiet_source_sites=0)
                      for name in ('current', *ARMS)})


@torch.no_grad()
def audit(model, cases):
    row = empty()
    for events in cases:
        model.reset_state()
        for probe in ARMS.values():
            probe.reset_state()
        pending = deque()
        for t, (event, code, surface) in enumerate(sensory_sequence(events)):
            if len(pending) == 4:
                origin, states, origin_event = pending.popleft()
                if origin >= 4:
                    quiet = not origin_event.any()
                    row['frames'] += 1
                    row['targets'] += int(event.sum())
                    if quiet:
                        row['quiet_frames'] += 1
                        row['quiet_targets'] += int(event.sum())
                    for name, state in states.items():
                        arm = row['arms'][name]
                        reached, _ = reached_targets(state, event)
                        sites = int((state.max(0).values >= .5).sum())
                        arm['reached'] += reached
                        arm['source_sites'] += sites
                        if quiet:
                            arm['quiet_reached'] += reached
                            arm['quiet_source_sites'] += sites
            model.step((code, surface))
            states = {'current': model.state.clone()}
            states.update({name: probe.step(model.state).clone()
                           for name, probe in ARMS.items()})
            pending.append((t, states, event))
    for arm in row['arms'].values():
        arm['support'] = arm['reached']/max(row['targets'], 1)
        arm['quiet_support'] = arm['quiet_reached']/max(
            row['quiet_targets'], 1)
        arm['sites_per_frame'] = arm['source_sites']/max(row['frames'], 1)
        arm['quiet_sites_per_frame'] = arm['quiet_source_sites']/max(
            row['quiet_frames'], 1)
    return row


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
    results = {name: audit(model, cases) for name, cases in groups.items()}
    results['elapsed_seconds'] = time.perf_counter()-started
    OUT.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {arm: dict(
        support=round(metrics['support'], 3),
        quiet_support=round(metrics['quiet_support'], 3),
        sites=round(metrics['sites_per_frame'], 2))
        for arm, metrics in row['arms'].items()}
        for name, row in results.items() if isinstance(row, dict)}), flush=True)


if __name__ == '__main__':
    main()
