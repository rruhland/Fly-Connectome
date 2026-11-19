"""Prospective paired qualification of useful control after a hidden request delay."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import secrets
import signal
import time

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import model_bytes, source_signature
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_visual_button_effects import VisualButtonEffects
from public_entity_pong_native import snapshot
from return_metrics import ReturnCounter
from run_pong_inverse_cycle import save, sha
from run_pong_mid_rally_remap import END_TICK, MODULUS, NORMAL, add_digest
from run_pong_rapid_recovery_qualification import Candidate, rate, read, worker_init
from run_pong_surprise_recovery import POLICY, exclusions, runtime_dependencies, state_hash

ARMS = ('live', 'timing_frozen', 'stage1_undelayed')
PHASES = ('mid_rally_delay', 'carried_delay', 'zero_delay_remap')
DELAYS = (12, 21)
SWITCH_TICKS = (360, 600)
RESTORE_REMAP_TICK = 480
RESTORE_MAPPINGS = ((1, 2, 0), (2, 1, 0))
WORKERS = 4
WALL_BUDGET = 360
GATES = dict(first_planned_floor=.65, paired_first_gain_floor=.20, paired_stream_p_max=.05,
             later_rate_floor=.70, later_opportunity_minimum=24, two_return_start_floor=.50,
             max_extra_pre_switch_misses=0, max_extra_pre_switch_endings=0,
             restoration_opening_floor=.75, restoration_later_floor=.80,
             restoration_remap_first_floor=.70, restoration_remap_later_floor=.80,
             max_first_restoration_loss_vs_stage1=.20)
SOURCES = ('run_pong_timing_qualification.py', 'run_pong_rapid_recovery_qualification.py',
           'run_pong_mid_rally_remap.py', 'public_entity_pong_native.py', 'return_metrics.py',
           'pong_relational_feedback.py', 'pong_relational_persistent_binding.py',
           'pong_visual_button_effects.py', 'run_pong_inverse_cycle.py', 'run_pong_surprise_recovery.py')


def prepare(directory, acquisition, prior, motor_carry_receipt):
    model_bytes()
    acquired = read(acquisition)
    learner = VisualButtonEffects.from_state(acquired)
    if learner.probe_queue or tuple(learner.request(i) for i in NORMAL) != NORMAL:
        raise ValueError('acquisition checkpoint must have settled original controls')
    carried = read(motor_carry_receipt)['points'][-1]['final']
    if tuple(carried['order']) != NORMAL or not carried['updates_enabled']:
        raise ValueError('expected live motor memory after restoring the original controls')
    initial_timing = dict(schema='public-visual-pong-timing-v1', delay_commands=0,
                          motor_order=carried['order'], motor_probabilities=carried['probabilities'])
    used, excluded = exclusions(prior)
    entropy = secrets.randbits(128)
    rng = np.random.default_rng(entropy)

    def draw(direction):
        while True:
            seed = int(rng.integers(1, MODULUS))
            if seed not in used and ((seed * 48271 + 1) % MODULUS) % 2 == direction:
                used.add(seed)
                return seed

    cases = [dict(delay=delay, switch_tick=tick, direction=direction, restore_mapping=mapping,
                  seeds=[draw((direction + phase) % 2) for phase in range(len(PHASES))])
             for delay in DELAYS for tick in SWITCH_TICKS for direction in range(2)
             for mapping in RESTORE_MAPPINGS]
    rng.shuffle(cases)
    for index, case in enumerate(cases):
        case['index'] = index
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'schedules.json', cases)
    save(directory / 'checkpoint-acquired.json', acquired)
    save(directory / 'timing-initial.json', initial_timing)
    spec = dict(contract='paired-native-timing-delay-useful-control-v1', source=source_signature(),
                dependencies=runtime_dependencies(),
                experimental_sources={name: sha(Path(__file__).with_name(name)) for name in SOURCES},
                prepared_files={name: sha(directory / name) for name in
                                ('schedules.json', 'checkpoint-acquired.json', 'timing-initial.json')},
                motor_carry_receipt_sha256=sha(motor_carry_receipt),
                excluded_metadata=excluded, entropy=hex(entropy), acquisition_state_sha256=state_hash(acquired),
                arms=ARMS, phases=PHASES, cases=len(cases), delays=DELAYS, hidden_switches=SWITCH_TICKS,
                restore_remap_tick=RESTORE_REMAP_TICK, gates=GATES, workers=WORKERS, wall_budget=WALL_BUDGET,
                max_transitions=len(cases)*len(ARMS)*len(PHASES)*1000,
                max_initial_cameras=len(cases)*len(ARMS)*len(PHASES),
                timing_input='public entities, own requests, elapsed time and inherited hit/terminal feedback',
                continuation='actual state exported after the previous episode in that arm; fresh native command queue per episode',
                analysis_unit='independent paired stream; first opportunities in both delay phases count per planned episode',
                timing_frozen='pins the initial zero delay only; motor-order learning remains live')
    save(directory / 'spec.json', spec)
    return sha(directory / 'spec.json')


def verify(directory, spec_digest):
    path = directory / 'spec.json'
    if sha(path) != spec_digest:
        raise ValueError('registration SHA-256 mismatch')
    spec = read(path)
    model_bytes()
    if spec['source'] != source_signature() or spec['dependencies'] != runtime_dependencies():
        raise ValueError('package source or dependencies changed')
    for files, root in ((spec['experimental_sources'], Path(__file__).parent), (spec['prepared_files'], directory)):
        if any(sha(root / name) != expected for name, expected in files.items()):
            raise ValueError('preregistered source or schedule changed')
    return spec


def episode(case, arm, phase, controller):
    phase_index = PHASES.index(phase)
    switch_tick = (case['switch_tick'], 0, RESTORE_REMAP_TICK)[phase_index]
    camera = PongCamera(case['seeds'][phase_index], max_episode_ticks=END_TICK, opponent_speed=.65)
    game, vision, encoder = camera.game, load_default(), VisualStateEncoder(64, 64, .02)
    counter, events = ReturnCounter(), []
    step = game.step

    def instrument(drive, **kwargs):
        before = counter.opportunities, counter.returns
        tick = int(game.rally_steps[0]) + 1
        outcome = counter.step(game, step, drive, **kwargs)
        if counter.opportunities > before[0]:
            events.append(dict(tick=tick, hit=counter.returns > before[1]))
        return outcome

    def entities(frame, previous):
        changes = torch.stack((previous & ~frame, frame & ~previous)).float()
        return snapshot(encoder.encode_entities(*vision.track_entities(changes, frame.float())))

    game.step = instrument
    prefix, public_trace = hashlib.sha256(), hashlib.sha256()
    frame = game.render(64, 64)[0].clone()
    public = entities(frame, torch.zeros_like(frame))
    for digest in (prefix, public_trace):
        add_digest(digest, public)
    history, timing_changes = [], []
    prior_delay = None
    entered = False
    state_at_switch = None
    while True:
        tick = camera.total_ticks
        after = tick >= switch_tick
        if after and not entered:
            if tick != switch_tick:
                raise ValueError('intervention must coincide with a camera boundary')
            entered = True
            if arm != 'stage1_undelayed':
                state_at_switch = controller.state()
        lag = case['delay'] if phase != 'zero_delay_remap' and after and arm != 'stage1_undelayed' else 0
        mapping = case['restore_mapping'] if phase == 'zero_delay_remap' and after else NORMAL
        request, diagnostic = controller.choose(public, 1)
        if type(request) is not int or request not in NORMAL:
            raise ValueError('expected legal camera command')
        delay_belief = diagnostic.get('delay_commands', 0)
        if delay_belief != prior_delay:
            timing_changes.append(dict(tick=tick, estimated_delay=delay_belief))
            prior_delay = delay_belief
        history.append(request)
        applied = history[-lag-1] if len(history) > lag else 1
        result = camera.step(mapping[applied])
        public = entities(result.frame, frame)
        frame = result.frame
        feedback = dict(own_hits=result.hits, reward=result.reward, terminated=result.terminated,
                        truncated=result.truncated, elapsed_seconds=result.elapsed_seconds)
        controller.observe(public_entities=public, **feedback)
        row = request, public, *feedback.values()
        if not after:
            add_digest(prefix, row)
        add_digest(public_trace, row)
        if result.terminated or result.truncated:
            break
    if len(history) > 1000 or counter.opportunities != len(events) or counter.returns != sum(e['hit'] for e in events):
        raise ValueError('native budget or return accounting mismatch')
    return dict(phase=phase, switch_tick=switch_tick, entered=entered, events=events,
                prefix_sha256=prefix.hexdigest(), trace_sha256=public_trace.hexdigest(),
                timing_changes=timing_changes, state_at_switch=state_at_switch,
                total_ticks=camera.total_ticks, transitions=len(history),
                reward=result.reward, terminated=result.terminated, truncated=result.truncated)


def stream(output, case, arm, acquired, initial_timing, timing, rapid, deadline):
    from pong_rapid_motor_recovery import ORDERS, RapidMotorController
    if sha(Path(rapid)) != timing['rapid_sha256']:
        raise ValueError('inherited rapid motor source changed')
    timed_controller = timing['source'].controller()
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'], POLICY['coefficients'], POLICY['scales']),
                                       POLICY['require_first_hit'])
    learner = VisualButtonEffects.from_state(acquired)
    state = initial_timing
    receipts = []
    for phase in PHASES:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            return False
        if arm == 'stage1_undelayed':
            controller = RapidMotorController(policy, learner)
            controller.belief.confirmed = ORDERS.index(tuple(state['motor_order']))
            controller.belief.probabilities = np.asarray(state['motor_probabilities'], dtype=float)
        else:
            controller = timed_controller(policy, learner, timing_state=state, timing_updates=arm == 'live')
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            row = episode(case, arm, phase, controller)
        except TimeoutError:
            return False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if arm != 'stage1_undelayed':
            state = controller.state()
            if arm == 'timing_frozen' and state['delay_commands'] != 0:
                raise ValueError('timing-frozen arm changed its delay')
        else:
            state = timed_controller.carried_motor(controller)
        row['carried_state'] = state
        receipts.append(row)
    save(output / arm / f"stream-{case['index']:02d}.json", dict(case=case, episodes=receipts))
    return True


def affected(row):
    return [e for e in row['events'] if e['tick'] > row['switch_tick']]


def first(row):
    events = affected(row)
    return bool(events and events[0]['hit'])


def metrics(episodes, annotated_delay=None):
    available = [affected(row) for row in episodes]
    later = [event for events in available for event in events[1:]]
    opening = [row['events'][0] for row in episodes if row['events']]
    all_later = [event for row in episodes for event in row['events'][1:]]
    pre = [e for row in episodes for e in row['events'] if e['tick'] <= row['switch_tick']]
    hits = sum(first(row) for row in episodes)
    times_to_exact = [next((max(0, change['tick']-row['switch_tick']) / 120
                           for change in row['timing_changes'] if change['estimated_delay'] == annotated_delay), None)
                      for row in episodes] if annotated_delay is not None else None
    return dict(games=len(episodes), entered=sum(row['entered'] for row in episodes),
                first_returns=hits, first_opportunities=sum(bool(e) for e in available),
                planned_first_fraction=rate(hits, len(episodes)), later_returns=sum(e['hit'] for e in later),
                later_opportunities=len(later), later_fraction=rate(sum(e['hit'] for e in later), len(later)),
                two_return_starts=sum(sum(e['hit'] for e in events) >= 2 for events in available),
                full_game_first_returns=sum(e['hit'] for e in opening), full_game_first_opportunities=len(opening),
                full_game_later_returns=sum(e['hit'] for e in all_later), full_game_later_opportunities=len(all_later),
                pre_switch_misses=sum(not e['hit'] for e in pre), pre_switch_opportunities=len(pre),
                timing_exact=sum(value is not None for value in times_to_exact) if times_to_exact is not None else None,
                time_to_exact_seconds=times_to_exact,
                transitions=sum(row['transitions'] for row in episodes), losses=sum(row['reward'] < 0 for row in episodes),
                wins=sum(row['reward'] > 0 for row in episodes))


def comparisons(receipts, delays):
    live, frozen, reference = (receipts[arm] for arm in ARMS)
    shared = sorted(live.keys() & frozen.keys() & reference.keys())
    differences = []
    gains = {str(delay): 0 for delay in delays}
    matched_prefixes = extra_misses = extra_endings = 0
    for index in shared:
        active, control, normal = [arm[index] for arm in (live, frozen, reference)]
        if active['case'] != control['case'] or active['case'] != normal['case']:
            raise ValueError('stream cases do not match')
        a, b, c = [arm['episodes'] for arm in (active, control, normal)]
        matched_prefixes += a[0]['prefix_sha256'] == b[0]['prefix_sha256'] == c[0]['prefix_sha256']
        counts = [sum(not e['hit'] and e['tick'] <= rows[0]['switch_tick'] for e in rows[0]['events'])
                  for rows in (a, b, c)]
        extra_misses += max(0, counts[0]-min(counts[1:]))
        extra_endings += bool(not a[0]['entered'] and (b[0]['entered'] or c[0]['entered']))
        gain = sum(first(row) for row in a[:2])-sum(first(row) for row in b[:2])
        differences.append(gain)
        gains[str(active['case']['delay'])] += gain
    positives = sum(n > 0 for n in differences)
    discordant = sum(n != 0 for n in differences)
    p = sum(math.comb(discordant, k) for k in range(positives, discordant+1)) / 2 ** discordant
    return dict(paired_streams=len(shared), shared_prefixes=matched_prefixes, exact_one_sided_stream_sign_p=p,
                positive_streams=positives, negative_streams=discordant-positives,
                first_gain=rate(sum(differences), 2*len(shared)), first_count_gain_by_delay=gains,
                extra_pre_switch_misses=extra_misses, extra_pre_switch_endings=extra_endings)


def assessment(arms, paired, complete):
    active = arms['live']['delayed']
    restore = arms['live']['zero_delay_remap']
    reference = arms['stage1_undelayed']['zero_delay_remap']
    g = GATES
    timing = dict(complete=complete, first=active['planned_first_fraction'] >= g['first_planned_floor'],
                  paired_first=paired['first_gain'] >= g['paired_first_gain_floor']
                      and paired['exact_one_sided_stream_sign_p'] <= g['paired_stream_p_max'],
                  improvement_at_both_delays=all(gain > 0 for gain in paired['first_count_gain_by_delay'].values()),
                  later=active['later_fraction'] >= g['later_rate_floor']
                      and active['later_opportunities'] >= g['later_opportunity_minimum'],
                  sustained=rate(active['two_return_starts'], active['games']) >= g['two_return_start_floor'],
                  prefix=paired['extra_pre_switch_misses'] <= g['max_extra_pre_switch_misses']
                      and paired['extra_pre_switch_endings'] <= g['max_extra_pre_switch_endings'])
    retained_first = rate(restore['full_game_first_returns'], restore['games'])
    restoration = dict(complete=complete, opening=retained_first >= g['restoration_opening_floor'],
        later=rate(restore['full_game_later_returns'], restore['full_game_later_opportunities']) >= g['restoration_later_floor'],
        remap_first=restore['planned_first_fraction'] >= g['restoration_remap_first_floor']
            and restore['planned_first_fraction'] >= reference['planned_first_fraction']-g['max_first_restoration_loss_vs_stage1'],
        remap_later=restore['later_fraction'] >= g['restoration_remap_later_floor'])
    return dict(timing=timing, restored_motor_retention=restoration,
                timing_qualification=all(timing.values()), restoration_qualification=all(restoration.values()),
                full_qualification=all(timing.values()) and all(restoration.values()))


def run(directory, spec_digest, timing_source):
    from pong_rapid_motor_recovery import __file__ as rapid_source
    spec = verify(directory, spec_digest)
    cases = read(directory / 'schedules.json')
    acquired = read(directory / 'checkpoint-acquired.json')
    initial_timing = read(directory / 'timing-initial.json')
    candidate = Candidate(str(timing_source), 'TimedMotorController', sha(timing_source))
    timing = dict(source=candidate, rapid_sha256=sha(Path(rapid_source)))
    output = directory / 'run'
    output.mkdir(exist_ok=False)
    for arm in ARMS:
        (output / arm).mkdir()
    started = time.monotonic()
    deadline = started + WALL_BUDGET
    save(output / 'dispatch.json', dict(candidate=vars(candidate), rapid_sha256=timing['rapid_sha256'],
         spec_sha256=spec_digest, deadline=deadline))
    errors = []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('spawn'), initializer=worker_init) as pool:
        work = {pool.submit(stream, output, case, arm, acquired, initial_timing, timing, rapid_source, deadline): (arm, case['index'])
                for case in cases for arm in ARMS}
        for future in as_completed(work):
            try:
                if not future.result():
                    errors.append((work[future], 'deadline'))
            except Exception as error:
                errors.append((work[future], repr(error)))
    verify(directory, spec_digest)
    if sha(timing_source) != candidate.sha256 or sha(Path(rapid_source)) != timing['rapid_sha256']:
        raise ValueError('frozen controller changed')
    receipts = {arm: {row['case']['index']: row for row in
                      (read(path) for path in sorted((output / arm).glob('stream-*.json')))} for arm in ARMS}
    complete = not errors and all(len(items) == spec['cases'] for items in receipts.values())
    stats = {}
    for arm, rows in receipts.items():
        summaries = {phase: metrics([row['episodes'][index] for row in rows.values()])
                     for index, phase in enumerate(PHASES)}
        summaries['delayed'] = metrics([episode for row in rows.values() for episode in row['episodes'][:2]])
        summaries['by_delay'] = {str(delay): metrics([episode for row in rows.values()
             if row['case']['delay'] == delay for episode in row['episodes'][:2]], delay) for delay in DELAYS}
        stats[arm] = summaries
    evidence = comparisons(receipts, DELAYS)
    report = dict(complete=complete, errors=errors, spec_sha256=spec_digest, elapsed_wall_seconds=time.monotonic()-started,
                  arms=stats, paired=evidence, qualification=assessment(stats, evidence, complete),
                  initial_cameras=sum(len(item['episodes']) for arm in receipts.values() for item in arm.values()),
                  transitions=sum(item['transitions'] for arm in receipts.values() for row in arm.values()
                                  for item in row['episodes']))
    save(output / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    planning = commands.add_parser('prepare')
    planning.add_argument('directory', type=Path)
    planning.add_argument('acquisition', type=Path)
    planning.add_argument('prior_experiments', type=Path)
    planning.add_argument('motor_carry_receipt', type=Path)
    running = commands.add_parser('run')
    running.add_argument('directory', type=Path)
    running.add_argument('timing_source', type=Path)
    running.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.directory.resolve(), args.acquisition.resolve(), args.prior_experiments.resolve(),
                      args.motor_carry_receipt.resolve()))
    else:
        result = run(args.directory.resolve(), args.spec_sha256, args.timing_source.resolve())
        print(json.dumps({key: result[key] for key in ('complete', 'errors', 'qualification')}, sort_keys=True))
