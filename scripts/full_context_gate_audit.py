"""Audit issue-time gate and raw local eligibility at real Pong event targets."""
import argparse
import json
from pathlib import Path

import torch

from full_context_pong import OpenLoopContextRun
from fly_connectome.data import checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = Path('checkpoints/event-v1-combined-rate-initial.pt')
    source_sha = checksum(source)
    payload = torch.load(source, weights_only=True)
    training = (torch.load('runs/full-context-m1a-v1/pilot-train-500.pt', weights_only=True)
                if args.trained else None)
    components = training['network']['components'] if training is not None else None
    run = OpenLoopContextRun(payload, 1102, 0., components=components)
    n, rule = run.net, run.rule
    counts = {name:dict(events=0, gate_open=0, raw_eligibility=0, both=0,
                        raw_correct_sign=0, gated_correct_sign=0,
                        gated_abs_at_least_point_one=0,
                        raw_abs_at_least_point_one=0,
                        raw_anticipation_sum=0., gated_anticipation_sum=0.)
              for name in ('on', 'off')}
    quiet = dict(samples=0, raw_false_alarms=0, gated_false_alarms=0)
    previous_raw_eligibility = None
    original_observe = rule.observe

    def observed_with_audit(activity, reward):
        nonlocal previous_raw_eligibility
        boundary = rule.tick % 8 == 0
        if boundary and rule.last_issue is not None:
            target = rule.config.observation(activity, n.config.threshold,
                rule.sensory_mask, rule.sensory_gain)[0, n.targets]
            issued = rule.last_issue
            quiet_mask = target == 0
            quiet['samples'] += int(quiet_mask.sum())
            quiet['raw_false_alarms'] += int(
                ((issued['raw_prediction'].abs() >= .1) & quiet_mask).sum())
            quiet['gated_false_alarms'] += int(
                ((issued['prediction'].abs() >= .1) & quiet_mask).sum())
            for name, event_mask in (('on', target < 0), ('off', target > 0)):
                stats = counts[name]
                gate = issued['gate'] & event_mask
                eligible = (previous_raw_eligibility > 0) & event_mask
                raw = issued['raw_prediction']*target
                gated = issued['prediction']*target
                stats['events'] += int(event_mask.sum())
                stats['gate_open'] += int(gate.sum())
                stats['raw_eligibility'] += int(eligible.sum())
                stats['both'] += int((gate & eligible).sum())
                stats['raw_correct_sign'] += int(((raw > 0) & event_mask).sum())
                stats['gated_correct_sign'] += int(((gated > 0) & event_mask).sum())
                stats['gated_abs_at_least_point_one'] += int(
                    ((issued['prediction'].abs() >= .1) & event_mask).sum())
                stats['raw_abs_at_least_point_one'] += int(
                    ((issued['raw_prediction'].abs() >= .1) & event_mask).sum())
                stats['raw_anticipation_sum'] += float(raw[event_mask].sum())
                stats['gated_anticipation_sum'] += float(gated[event_mask].sum())
        original_observe(activity, reward)
        if boundary:
            edges = rule.keys.remainder(n.e)
            positions = n.target_lookup[n.post[edges]]
            active = rule.values.abs() > rule.config.prune_epsilon
            previous_raw_eligibility = torch.bincount(positions[active],
                                                     minlength=len(n.targets))

    rule.observe = observed_with_audit
    run.run(500)
    label = 'trained' if args.trained else 'initial'
    original = torch.load(f'runs/full-context-m1a-v1/pilot-{label}-500.pt', weights_only=True)
    if run.events != original['events']:
        raise AssertionError('audit did not replay the original camera event stream')
    if checksum(source) != source_sha:
        raise AssertionError('source checkpoint changed')
    result = dict(source_sha256=source_sha, seed=1102, frames=500, weights=label,
        training=False, eligible_edges=len(n.incoming), targets=len(n.targets),
        counts=counts, quiet=quiet, camera_events_identical_to_frozen_pilot=True,
        script_sha256=checksum(Path(__file__)))
    path = Path(f'runs/full-context-m1a-v1/gate-audit-v2-{label}.json')
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
