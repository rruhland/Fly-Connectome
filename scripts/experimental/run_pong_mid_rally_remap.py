"""Measure an unannounced command cycle during a continuing native Pong rally."""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import MODEL_HASH, model_bytes, source_signature
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_visual_button_effects import VisualButtonEffects, VisualIntentController
from public_entity_pong_native import snapshot
from return_metrics import ReturnCounter
from run_pong_inverse_cycle import save, sha
from run_pong_surprise_recovery import POLICY, exclusions, reached_deadline, state_hash

NORMAL = (0, 1, 2)
CYCLED = (1, 2, 0)
SWITCH_TICK = 4 * 120
END_TICK = 20 * 120
PAIRS = 12
START_SEED = 128947331
WALL_BUDGET = 300
MODULUS = 2147483647
FILES = ('run_pong_mid_rally_remap.py', 'pong_visual_button_effects.py',
         'pong_relational_feedback.py', 'pong_relational_persistent_binding.py',
         'public_entity_pong_native.py', 'return_metrics.py', 'run_pong_inverse_cycle.py',
         'run_pong_surprise_recovery.py')


def prepare(directory, checkpoint):
    parent = checkpoint.parent
    acquisition = json.loads((parent / 'live/game-05.json').read_text())
    acquired = json.loads(checkpoint.read_text())
    assert acquired == acquisition['state'] and state_hash(acquired) == acquisition['after']
    learner = VisualButtonEffects.from_state(acquired)
    assert learner.updates_enabled and not learner.probe_queue and learner.effects is not None
    assert [learner.request(intent) for intent in NORMAL] == list(NORMAL)
    used, metadata = exclusions(directory.parent)
    used.add(START_SEED)
    rng = np.random.default_rng(START_SEED)
    starts = []
    while len(starts) < PAIRS:
        seed = int(rng.integers(1, MODULUS))
        if seed not in used and ((seed * 48271 + 1) % MODULUS) % 2 == len(starts) % 2:
            used.add(seed)
            starts.append(seed)
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'schedules.json', starts)
    save(directory / 'checkpoint-acquired.json', acquired)
    spec = dict(contract='paired-native-mid-rally-visual-command-reidentification-v1',
        starts=PAIRS, max_games=2 * PAIRS, max_transition_cameras=2 * PAIRS * 1000,
        max_initial_cameras=2 * PAIRS, wall_budget=WALL_BUDGET, hidden_switch_tick=SWITCH_TICK,
        max_episode_ticks=END_TICK, normal=NORMAL, changed=CYCLED, opponent_speed=.65,
        acquisition_checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        acquisition_receipt_sha256=sha(parent / 'live/game-05.json'),
        schedule_sha256=sha(directory / 'schedules.json'), frozen_vision_sha256=MODEL_HASH,
        sources=source_signature(), prior_seed_metadata=metadata, policy=POLICY,
        files={name: sha(Path(__file__).with_name(name)) for name in FILES},
        learner_inputs='public visual entities, own-hit/terminal feedback for the inherited controller only, elapsed camera time',
        prefix_parity='hash exact initial visual entities, requests, observed entities and outcomes through tick 480',
        criterion='first fully learned correct ordering checked after each update before terminal; must precede first changed-control own opportunity',
        scope='prepared paired test, no restarts or tuning')
    save(directory / 'spec.json', spec)
    return spec, sha(directory / 'spec.json'), starts, acquired


def add_digest(digest, value):
    digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    digest.update(b'\n')


def play(seed, learner):
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'],
        POLICY['coefficients'], POLICY['scales']), POLICY['require_first_hit'])
    controller = VisualIntentController(policy, learner)
    camera = PongCamera(seed, max_episode_ticks=END_TICK, opponent_speed=.65)
    vision, encoder = load_default(), VisualStateEncoder(64, 64, .02)
    count, events = ReturnCounter(), []
    game = camera.game
    step = game.step

    def instrument(drive, **kwargs):
        before = count.opportunities
        before_returns = count.returns
        tick = int(game.rally_steps[0]) + 1
        outcome = count.step(game, step, drive, **kwargs)
        if count.opportunities > before:
            events.append(dict(tick=tick, seconds_since_change=(tick - SWITCH_TICK) / 120,
                               hit=count.returns > before_returns))
        return outcome
    game.step = instrument
    frame = game.render(64, 64)[0].clone()

    def observed(frame, previous):
        changes = torch.stack((previous & ~frame, ~previous & frame)).float()
        return snapshot(encoder.encode_entities(*vision.track_entities(changes, frame.float())))

    public = observed(frame, torch.zeros_like(frame))
    prefix = hashlib.sha256()
    add_digest(prefix, public)
    expected_requests = [CYCLED.index(intent) for intent in NORMAL]
    switch_entered = False
    initial_calibrations = learner.completed_calibrations
    calibration_events = []
    requests = 0
    prefix_requests = 0
    correct_tick = None
    first_contradiction_tick = None
    while True:
        tick = camera.total_ticks
        changed = tick >= SWITCH_TICK
        if changed and not switch_entered:
            assert tick == SWITCH_TICK
            switch_entered = True
        action, diagnostic = controller.choose(public, 1)
        result = camera.step((CYCLED if changed else NORMAL)[action])
        public = observed(result.frame, frame)
        frame = result.frame
        requests += 1
        previous_calibrations = learner.completed_calibrations
        previous_contradictions = learner.contradictions
        controller.observe(public_entities=public, own_hits=result.hits, reward=result.reward,
            terminated=result.terminated, truncated=result.truncated, elapsed_seconds=result.elapsed_seconds)
        terminal = result.terminated or result.truncated
        if not changed:
            assert camera.total_ticks <= SWITCH_TICK
            prefix_requests += 1
            add_digest(prefix, (action, public, result.hits, result.reward, result.elapsed_seconds,
                               result.terminated, result.truncated))
        if changed and first_contradiction_tick is None and learner.contradictions > previous_contradictions:
            first_contradiction_tick = camera.total_ticks
        if learner.completed_calibrations > previous_calibrations:
            calibration_events.append(dict(tick=camera.total_ticks, effects=learner.effects.copy(),
                decoded_requests=diagnostic_order(learner), before_terminal=not terminal))
        if (changed and not terminal and correct_tick is None and learner.completed_calibrations > initial_calibrations
                and diagnostic_order(learner) == expected_requests):
            correct_tick = camera.total_ticks
        if terminal:
            break
    assert requests <= 1000 and len(events) == count.opportunities
    assert sum(e['hit'] for e in events) == count.returns
    changed_events = [event for event in events if event['tick'] > SWITCH_TICK]
    first_changed = changed_events[0] if changed_events else None
    return dict(entered_switch=switch_entered, prefix_requests=prefix_requests, prefix_sha256=prefix.hexdigest(),
        post_change_events=changed_events, events=events, total_ticks=camera.total_ticks,
        terminated=result.terminated, truncated=result.truncated, transitions=requests, initial=1,
        first_correct_order_tick=correct_tick,
        first_correct_order_seconds=(correct_tick - SWITCH_TICK) / 120 if correct_tick is not None else None,
        order_ready_before_first_post_change_opportunity=(
            correct_tick < first_changed['tick'] if correct_tick is not None and first_changed else False),
        first_contradiction_seconds=(first_contradiction_tick - SWITCH_TICK) / 120
            if first_contradiction_tick is not None else None,
        calibrations=calibration_events)


def diagnostic_order(learner):
    return [learner.request(intent) for intent in NORMAL]


def metrics(rows):
    entries = [row for row in rows if row['entered_switch']]
    first_exposure = sum(bool(row['post_change_events']) for row in entries)
    first_hits = sum(bool(row['post_change_events'] and row['post_change_events'][0]['hit']) for row in entries)
    later_events = [event for row in entries for event in row['post_change_events'][1:]]
    correct = [row['first_correct_order_seconds'] for row in entries if row['first_correct_order_tick'] is not None]
    return dict(games=len(rows), entered_switch=len(entries), first_returns=first_hits, first_opportunities=first_exposure,
        later_returns=sum(event['hit'] for event in later_events), later_opportunities=len(later_events),
        order_identified_before_terminal=len(correct), order_ready_before_first=sum(
            row['order_ready_before_first_post_change_opportunity'] for row in entries),
        order_identified_seconds_after_change=correct,
        probe_frames=sum(row['probe_frames'] for row in rows))


def run(directory, checkpoint):
    torch.set_num_threads(1)
    model_bytes()
    signal.signal(signal.SIGALRM, reached_deadline)
    spec, digest, starts, acquired = prepare(directory, checkpoint)
    began = time.monotonic()
    deadline = began + WALL_BUDGET
    save(directory / 'work-clock.json', dict(start=began, deadline=deadline, spec_sha256=digest))
    results = {}
    timed_out = False
    try:
        for index, seed in enumerate(starts):
            results[index] = {}
            arms = ('live', 'frozen') if index % 2 == 0 else ('frozen', 'live')
            for arm in arms:
                learner = VisualButtonEffects.from_state(acquired)
                if arm == 'frozen':
                    learner = learner.frozen_branch()
                before = learner.state()
                save(directory / 'active.json', dict(seed=seed, index=index, arm=arm, camera_grant=1000))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('wall budget reached at game dispatch')
                signal.setitimer(signal.ITIMER_REAL, remaining)
                try:
                    row = play(seed, learner)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                if arm == 'frozen':
                    assert learner.state() == before
                row.update(seed=seed, index=index, arm=arm, before=state_hash(before),
                           after=state_hash(learner.state()), probe_frames=learner.dither_count - before['dither_count'])
                save(directory / f'pair-{index:02d}-{arm}.json', row)
                (directory / 'active.json').unlink()
                results[index][arm] = row
            live, frozen = (results[index][arm] for arm in ('live', 'frozen'))
            print(f"pair {index + 1} switch {live['entered_switch']} "
                  f"order time {live['first_correct_order_seconds']} "
                  f"prefix parity {live['prefix_sha256'] == frozen['prefix_sha256']}", flush=True)
    except TimeoutError:
        timed_out = True
    model_bytes()
    pinned = (sha(directory / 'spec.json') == digest and source_signature() == spec['sources']
        and sha(checkpoint) == spec['checkpoint_sha256']
        and sha(directory / 'schedules.json') == spec['schedule_sha256']
        and all(sha(Path(__file__).with_name(name)) == expected for name, expected in spec['files'].items()))
    rows = [row for pair in results.values() for row in pair.values()]
    complete_pairs = [pair for pair in results.values() if len(pair) == 2]
    parity = [pair['live']['prefix_sha256'] == pair['frozen']['prefix_sha256'] for pair in complete_pairs]
    transitions = sum(row['transitions'] for row in rows)
    pending_grant = 1000 if (directory / 'active.json').exists() else 0
    assert transitions + pending_grant <= spec['max_transition_cameras']
    summary = dict(complete=len(complete_pairs) == PAIRS and not timed_out, timed_out=timed_out,
        complete_pairs=len(complete_pairs), all_normal_prefixes_identical=all(parity),
        identical_normal_prefix_pairs=sum(parity), sources_assets_unchanged=pinned, spec_sha256=digest,
        transitions=transitions, initial=sum(row['initial'] for row in rows), pending_camera_grant=pending_grant,
        wall_seconds=time.monotonic() - began,
        arms={arm: metrics([row for row in rows if row['arm'] == arm]) for arm in ('live', 'frozen')})
    save(directory / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory.resolve(), args.checkpoint.resolve()), sort_keys=True))
