"""Test native hidden request delays between the controller's candidate timings."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import model_bytes, source_signature
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_timing_controller import DELAYS, TimedMotorController
from pong_visual_button_effects import VisualButtonEffects
from public_entity_pong_native import snapshot
from return_metrics import ReturnCounter
from run_pong_inverse_cycle import save, sha
from run_pong_mid_rally_remap import FILES, MODULUS, add_digest
from run_pong_surprise_recovery import POLICY, exclusions, reached_deadline

PHYSICAL_DELAYS = (10, 17)
SWITCHES = (360, 600)
END_TICK = 1800
WALL_SECONDS = 180
MAX_CAMERAS = 750
SEED = 1679382879
ARMS = ('live', 'frozen')
PHASES = ('onset', 'held')


def read(path):
    return json.loads(path.read_text())


def prepare(directory, acquisition, promotion):
    model_bytes()
    timing_state = read(promotion)
    learner = VisualButtonEffects.from_state(read(acquisition))
    if timing_state['delay_commands'] != 0 or timing_state['motor_order'] != [0, 1, 2]:
        raise ValueError('a restored Stage 1 checkpoint is required')
    if not learner.updates_enabled or not set(DELAYS).isdisjoint(PHYSICAL_DELAYS):
        raise ValueError('both arms must keep the Stage 1 action model live')
    used, prior = exclusions(directory.parent)
    for path in Path('/workspace/scratch').glob('pong-*qualification-*/*-schedules.json'):
        rows = read(path)
        used.update(row['seed'] for row in rows if isinstance(row, dict) and 'seed' in row)
        prior[str(path)] = sha(path)
    rng = np.random.default_rng(SEED)
    sequences = []
    for delay in PHYSICAL_DELAYS:
        for switch in SWITCHES:
            for direction in (0, 1):
                seeds = []
                for _ in PHASES:
                    while True:
                        seed = int(rng.integers(1, MODULUS))
                        if seed not in used and ((seed * 48271 + 1) % MODULUS) % 2 == direction:
                            break
                    used.add(seed)
                    seeds.append(seed)
                sequences.append(dict(index=len(sequences), delay=delay, switch=switch, seeds=seeds))
    directory.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).with_name(name) for name in
             (*FILES, 'pong_timing_controller.py', 'pong_rapid_motor_recovery.py', Path(__file__).name)]
    lineage = promotion.with_name('lineage.json')
    paths.extend([acquisition, promotion, lineage])
    stage1_receipt = Path(read(lineage)['checkpoints']['stage-1-restored-promotion']['receipt'])
    paths.append(stage1_receipt)
    save(directory / 'schedules.json', sequences)
    paths.append(directory / 'schedules.json')
    spec = dict(contract='native-public-pong-off-grid-timing-v1', delays=PHYSICAL_DELAYS,
                switches=SWITCHES, inherited_policy=POLICY, timing_candidates=DELAYS,
                trials=len(sequences), arms=ARMS, phases=PHASES, max_cameras_per_game=MAX_CAMERAS,
                max_wall_seconds=WALL_SECONDS, endpoint_tick=END_TICK, opponent_speed=.65,
                source=source_signature(), inputs={str(p): sha(p) for p in paths},
                excluded_seed_receipts=prior, seed_generator=SEED,
                actuator_mapping=[0, 1, 2], no_backprop=True,
                initial_timing_state=timing_state, acquired_buttons=learner.state())
    save(directory / 'spec.json', spec)
    return tuple(sequences), spec


def play(seed, switch, delay, policy, learner, timing_state, live):
    controller = TimedMotorController(policy, learner, timing_state=timing_state, timing_updates=live)
    camera = PongCamera(seed, max_episode_ticks=END_TICK, opponent_speed=.65)
    vision, encoder = load_default(), VisualStateEncoder(64, 64, .02)
    game, counter = camera.game, ReturnCounter()
    original = game.step
    events = []

    def counted(drive, **kwargs):
        before, returned = counter.opportunities, counter.returns
        tick = int(game.rally_steps[0]) + 1
        outcome = counter.step(game, original, drive, **kwargs)
        if counter.opportunities > before:
            events.append(dict(tick=tick, hit=counter.returns > returned,
                               estimated_delay=controller.delay_commands,
                               order=controller.effective_requests))
        return outcome

    def observe(frame, previous):
        changes = torch.stack((previous & ~frame, ~previous & frame)).float()
        return snapshot(encoder.encode_entities(*vision.track_entities(changes, frame.float())))

    game.step = counted
    frame = game.render(64, 64)[0].clone()
    public = observe(frame, torch.zeros_like(frame))
    prefix = hashlib.sha256()
    add_digest(prefix, public)
    requests = deque(maxlen=delay + 1)
    changes, first = [], None
    entered = False
    while True:
        tick = camera.total_ticks
        entered |= tick >= switch
        action, detail = controller.choose(public, 1)
        requests.append(action)
        offset = delay if tick >= switch else 0
        executed = requests[-offset - 1] if len(requests) > offset else 1
        result = camera.step(executed)
        public = observe(result.frame, frame)
        frame = result.frame
        feedback = dict(own_hits=result.hits, reward=result.reward, terminated=result.terminated,
                        truncated=result.truncated, elapsed_seconds=result.elapsed_seconds)
        controller.observe(public_entities=public, **feedback)
        learned = controller.delay_commands
        if learned != detail['delay_commands']:
            changes.append(dict(tick=camera.total_ticks, lag=learned))
        if first is None and tick >= switch and abs(learned - delay) <= 2:
            first = camera.total_ticks
        if tick < switch:
            add_digest(prefix, (action, public, *feedback.values()))
        if result.terminated or result.truncated:
            break
    assert len(events) == counter.opportunities and sum(event['hit'] for event in events) == counter.returns
    assert camera.decisions <= MAX_CAMERAS
    assert live or controller.delay_commands == timing_state['delay_commands']
    return dict(seed=seed, switch=switch, entered_switch=entered, events=events, changes=changes,
                first_near_lag_tick=first, start=timing_state, final=controller.state(),
                prefix=prefix.hexdigest(), transitions=camera.decisions,
                reward=result.reward, terminated=result.terminated, truncated=result.truncated)


def sequence(directory, case, spec, deadline):
    torch.set_num_threads(1)
    signal.signal(signal.SIGALRM, reached_deadline)
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'], POLICY['coefficients'],
        POLICY['scales']), POLICY['require_first_hit'])
    rows = []
    for arm in (ARMS if case['index'] % 2 == 0 else ARMS[::-1]):
        state = spec['initial_timing_state']
        for game, (phase, seed) in enumerate(zip(PHASES, case['seeds'], strict=True)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('off-grid run exceeded its camera work allocation')
            learner = VisualButtonEffects.from_state(spec['acquired_buttons'])
            signal.setitimer(signal.ITIMER_REAL, remaining)
            try:
                row = play(seed, case['switch'] if game == 0 else 0, case['delay'], policy,
                           learner, state, arm == 'live')
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            state = row['final']
            row.update(phase=phase, arm=arm, delay=case['delay'], index=case['index'])
            save(directory / f"{case['index']:02d}-{phase}-{arm}.json", row)
            rows.append(row)
    return rows


def summarize(rows):
    opportunities = [r for r in rows if r['entered_switch']]
    events = [[e for e in r['events'] if e['tick'] > r['switch']] for r in opportunities]
    return dict(games=len(rows), entered=len(opportunities), first_opportunities=sum(map(bool, events)),
                first_returns=sum(bool(x and x[0]['hit']) for x in events),
                returns=sum(e['hit'] for x in events for e in x), opportunities=sum(map(len, events)),
                first_near_lag_seconds=[(r['first_near_lag_tick'] - r['switch']) / 120 for r in rows
                                        if r['first_near_lag_tick'] is not None],
                final_estimates=[r['final']['delay_commands'] for r in rows])


def run(directory, acquisition, promotion):
    cases, spec = prepare(directory, acquisition, promotion)
    start = time.monotonic()
    save(directory / 'dispatch.json', dict(start=start, deadline=start + WALL_SECONDS))
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(sequence, directory, case, spec, start + WALL_SECONDS) for case in cases]
        rows = [row for job in jobs for row in job.result()]
    results = {}
    for delay in PHYSICAL_DELAYS:
        results[delay] = {phase: {arm: summarize([r for r in rows if
            (r['delay'], r['phase'], r['arm']) == (delay, phase, arm)]) for arm in ARMS} for phase in PHASES}
    matched = sum(len({r['prefix'] for r in rows
                       if r['index'] == case['index'] and r['phase'] == 'onset'}) == 1 for case in cases)
    summary = dict(complete=len(rows) == len(cases) * len(PHASES) * len(ARMS),
        results=results, onset_prefixes_matched=matched, pairs=len(cases),
        cameras=sum(r['transitions'] for r in rows), initial_cameras=len(rows),
        unchanged_sources=source_signature() == spec['source']
            and all(sha(Path(p)) == digest for p, digest in spec['inputs'].items()),
        wall_seconds=time.monotonic() - start)
    save(directory / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('acquisition', type=Path)
    parser.add_argument('promotion', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory.resolve(), args.acquisition.resolve(), args.promotion.resolve()),
                     sort_keys=True))
