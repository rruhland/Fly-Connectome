"""Check batched timing against the scalar observer on fixed native trajectories."""

import argparse
from collections import deque
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_timing_controller import TimedMotorController, TimingMotorBelief
from pong_timing_fast import BatchedTimedMotorController, observe_shadow_bank
from pong_visual_button_effects import VisualButtonEffects
from public_entity_pong_native import snapshot
from run_pong_inverse_cycle import save, sha
from run_pong_surprise_recovery import POLICY


NORMAL = (0, 1, 2)
CYCLE = (1, 2, 0)
POINTS = (
    (240, 12, NORMAL, True),
    (120, 17, CYCLE, True),
    (360, 21, NORMAL, True),
    (0, 0, NORMAL, False),
)


def check_bank():
    rng = np.random.default_rng(721893)
    reference = [TimingMotorBelief(NORMAL) for _ in range(9)]
    batched = [TimingMotorBelief(NORMAL) for _ in range(9)]
    for belief in reference + batched:
        belief.initialize(.383)
    updates = 2000
    for index in range(updates):
        requests = rng.integers(0, 3, size=9)
        elapsed = rng.choice([2 / 120, 3 / 120])
        position = float(np.clip(rng.normal(.35, .06), -.397, .397)) if index % 7 else None
        expected = [belief.observe(request, elapsed, position)
                    for belief, request in zip(reference, requests, strict=True)]
        actual = observe_shadow_bank(batched, requests, elapsed, position)
        np.testing.assert_array_equal(actual, expected)
        for scalar, batch in zip(reference, batched, strict=True):
            check_belief(scalar, batch)
    return updates


def check_belief(reference, batched):
    assert reference.confirmed == batched.confirmed
    for name in ('probabilities', 'states', 'covariances'):
        np.testing.assert_array_equal(getattr(reference, name), getattr(batched, name))


def check_native(seed, checkpoint):
    torch.set_num_threads(1)
    learner = VisualButtonEffects.from_state(json.loads(checkpoint.read_text()))
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'], POLICY['coefficients'],
        POLICY['scales']), POLICY['require_first_hit'])
    camera = PongCamera(seed, max_episode_ticks=1200, opponent_speed=.65)
    vision, encoder = load_default(), VisualStateEncoder(64, 64, .02)

    def observe(frame, previous):
        changed = torch.stack((previous & ~frame, ~previous & frame)).float()
        return snapshot(encoder.encode_entities(*vision.track_entities(changed, frame.float())))

    carried, results = None, []
    for index, (switch, delay, mapping, enabled) in enumerate(POINTS):
        reference = TimedMotorController(policy, learner, timing_state=carried, timing_updates=enabled)
        restored = json.loads(json.dumps(carried)) if carried is not None else None
        batched = BatchedTimedMotorController(policy, learner, timing_state=restored, timing_updates=enabled)
        initial_state = reference.state()
        assert initial_state == batched.state()
        frame = camera.reset() if index else camera.game.render(64, 64)[0].clone()
        vision.reset_state()
        encoder.reset_state()
        public = observe(frame, torch.zeros_like(frame))
        requests = deque(maxlen=delay + 1)
        times = dict(reference=0., batched=0., sensor=0., camera=0.)
        transitions = hits = 0
        while True:
            action, detail = reference.choose(public, 1)
            assert batched.choose(public, 1) == (action, detail)
            requests.append(action)
            active_delay = delay if camera.episode_ticks >= switch else 0
            delivered = requests[-1 - active_delay] if len(requests) > active_delay else 1
            if camera.episode_ticks >= switch:
                delivered = mapping[delivered]
            started = perf_counter()
            result = camera.step(delivered)
            times['camera'] += perf_counter() - started
            started = perf_counter()
            public = observe(result.frame, frame)
            times['sensor'] += perf_counter() - started
            frame = result.frame
            for name, controller in [('reference', reference), ('batched', batched)]:
                started = perf_counter()
                controller.observe(own_hits=result.hits, reward=result.reward,
                    terminated=result.terminated, truncated=result.truncated,
                    elapsed_seconds=result.elapsed_seconds, public_entities=public)
                times[name] += perf_counter() - started
            assert reference.state() == batched.state()
            assert reference.candidate == batched.candidate
            assert reference.candidate_dwell == batched.candidate_dwell
            assert reference.delay_changes == batched.delay_changes
            np.testing.assert_array_equal(reference.evidence, batched.evidence)
            for lag in reference.lags:
                check_belief(reference.shadows[lag], batched.shadows[lag])
            transitions += 1
            hits += result.hits
            if result.terminated or result.truncated:
                break
        carried = reference.state()
        results.append(dict(point=index + 1, switch=switch, delay=delay, mapping=mapping,
            timing_updates=enabled, transitions=transitions, hits=hits, times=times,
            reference_state_before=initial_state, reference_state_after=carried,
            terminated=result.terminated, truncated=result.truncated))
    return results


def run(directory, checkpoint):
    directory.mkdir(exist_ok=False, parents=True)
    synthetic_transitions = check_bank()
    seeds = (987186821, 1981831, 273681705)
    native = {str(seed): check_native(seed, checkpoint) for seed in seeds}
    active = [row for points in native.values() for row in points if row['timing_updates']]
    count = sum(row['transitions'] for row in active)
    timings = {name: sum(row['times'][name] for row in active) for name in active[0]['times']}
    sources = [TimedMotorController, BatchedTimedMotorController, observe_shadow_bank]
    files = [Path(__file__).resolve(), *(Path(__import__('inspect').getfile(item)).resolve() for item in sources)]
    summary = dict(seeds=seeds, synthetic_transitions=synthetic_transitions,
                   native=native, active_transitions=count,
                   all_native_transitions=sum(row['transitions'] for points in native.values() for row in points),
                   active_timing_microseconds={name: seconds * 1e6 / count for name, seconds in timings.items()},
                   active_observer_speedup=timings['reference'] / timings['batched'],
                   sources={str(path): sha(path) for path in files})
    save(directory / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    result = run(args.directory, args.checkpoint)
    print(json.dumps({key: value for key, value in result.items() if key != 'native'}, sort_keys=True))
