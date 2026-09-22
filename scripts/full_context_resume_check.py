"""Exact midstream resume check for the opt-in full-graph Pong M1A runner."""
import json
from pathlib import Path

import torch

from full_context_pong import (BODY_STATE, NETWORK_STATE, PONG_STATE, RULE_STATE,
                               OpenLoopContextRun)
from fly_connectome.data import checksum


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            equal_nested(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(
            equal_nested(x, y) for x, y in zip(a, b))
    return a == b


def main():
    torch.set_num_threads(4)
    source = Path('checkpoints/event-v1-combined-rate-initial.pt')
    source_sha = checksum(source)
    payload = torch.load(source, weights_only=True)
    continuous = OpenLoopContextRun(payload, 1101, 1.)
    continuous.run(19)
    interrupted = OpenLoopContextRun(payload, 1101, 1.)
    interrupted.run(9)
    checkpoint = Path('runs/full-context-m1a-v1/resume-fixture-v2.pt')
    interrupted.save(checkpoint, source_sha)
    resumed = OpenLoopContextRun.load(checkpoint, payload, source_sha)
    resumed.run(10)
    for left, right, names in ((continuous.net, resumed.net, NETWORK_STATE),
                               (continuous.rule, resumed.rule, RULE_STATE),
                               (continuous.pong, resumed.pong, PONG_STATE),
                               (continuous.pong.body, resumed.pong.body, BODY_STATE)):
        for name in names:
            if not equal_nested(getattr(left, name), getattr(right, name)):
                raise AssertionError(f'exact resume mismatch: {name}')
    for name in ('forecast', 'last_issue', 'last_confirmation'):
        if not equal_nested(getattr(continuous.rule, name), getattr(resumed.rule, name)):
            raise AssertionError(f'exact resume mismatch: {name}')
    for name in ('previous_prediction', 'previous_target', 'metrics', 'trace',
                 'events', 'spike_counts', 'peak_spikes_per_tick', 'max_keys',
                 'open_gate_issues', 'max_update_error'):
        if not equal_nested(getattr(continuous, name), getattr(resumed, name)):
            raise AssertionError(f'exact resume mismatch: {name}')
    if not torch.equal(continuous.camera.previous, resumed.camera.previous):
        raise AssertionError('exact resume mismatch: camera reference')
    if continuous.net.step_index != resumed.net.step_index or continuous.rule.tick != resumed.rule.tick:
        raise AssertionError('exact resume mismatch: neural/learning tick')
    if checksum(source) != source_sha:
        raise AssertionError('source checkpoint changed')
    report = dict(source_sha256=source_sha, checkpoint_sha256=checksum(checkpoint),
        seed=1101, eta=1., uninterrupted_frames=19, checkpoint_frame=9,
        exact_fields=list(NETWORK_STATE)+list(RULE_STATE)+list(PONG_STATE)+list(BODY_STATE)
            + ['forecast', 'last_issue', 'last_confirmation', 'camera.previous',
               'previous_prediction', 'previous_target', 'metrics', 'trace', 'events',
               'spike_counts', 'peak_spikes_per_tick', 'max_keys', 'open_gate_issues',
               'max_update_error', 'step counters'],
        source_unchanged=True, script_sha256=checksum(Path(__file__)))
    output = Path('runs/full-context-m1a-v1/resume-check-v2.json')
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
