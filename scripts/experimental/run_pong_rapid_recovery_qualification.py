"""Prospective paired native Pong test of rapid recovery from unseen actuator relabeling."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib.util
from itertools import permutations
import json
import math
import multiprocessing as mp
from pathlib import Path
import secrets
import signal
import sys
import time

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import model_bytes, source_signature
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_visual_button_effects import VisualButtonEffects, VisualIntentController
from public_entity_pong_native import snapshot
from return_metrics import ReturnCounter
from run_pong_fast_mid_rally_scout import EarlyVisualIntentController
from run_pong_inverse_cycle import save, sha
from run_pong_mid_rally_remap import END_TICK, FILES, MODULUS, NORMAL, add_digest
from run_pong_surprise_recovery import POLICY, exclusions, reached_deadline, runtime_dependencies, state_hash

BASELINE_ARMS = ('unchanged', 'shorter', 'frozen', 'calibrated_at_switch')
CANDIDATE_ARMS = ('candidate_unchanged', 'candidate_changed')
SWITCH_TICKS = (240, 480, 840)
MAPPINGS = tuple(order for order in permutations(NORMAL) if order != NORMAL)
CONFIRM_CAMERAS = 10
GATES = dict(first_planned_fraction=.80, first_gain_over_shorter=.20,
             first_gain_p_one_sided=.05, per_mapping_first_fraction=.50,
             later_fraction=.90, later_min_opportunities=30, multi_return_fraction=.50,
             max_extra_pre_switch_misses=0, max_extra_pre_switch_endings=0,
             unchanged_first_fraction=.90, unchanged_later_fraction=.95,
             max_unchanged_fraction_loss=.05, correct_within_ticks=90,
             correct_within_fraction=.60, correct_consecutive_cameras=CONFIRM_CAMERAS)
WORKERS = 4
WALL_SECONDS = {'baseline': 300, 'candidate': 240}
MAX_TRANSITIONS = 1000
SOURCE_NAMES = (*FILES, 'run_pong_fast_mid_rally_scout.py',
                'run_pong_rapid_recovery_qualification.py')


def read(path):
    return json.loads(path.read_text())


def prepare(directory, checkpoint, prior):
    model_bytes()
    acquired = read(checkpoint)
    receipt = checkpoint.parent / 'live/game-05.json'
    event = read(receipt)
    if acquired != event['state'] or state_hash(acquired) != event['after']:
        raise ValueError('checkpoint does not match the acquired normal-play receipt')
    learner = VisualButtonEffects.from_state(acquired)
    if (not learner.updates_enabled or learner.probe_queue or learner.effects is None
            or tuple(learner.request(intent) for intent in NORMAL) != NORMAL):
        raise ValueError('expected a settled, live checkpoint for original controls')
    used, metadata = exclusions(prior)
    seed_entropy = secrets.randbits(128)
    rng = np.random.default_rng(seed_entropy)
    directory.mkdir(parents=True, exist_ok=False)
    for batch in ('primary', 'reserve'):
        rows = []
        for switch in SWITCH_TICKS:
            for mapping in MAPPINGS:
                for direction in range(2):
                    while True:
                        seed = int(rng.integers(1, MODULUS))
                        if seed not in used and ((seed * 48271 + 1) % MODULUS) % 2 == direction:
                            used.add(seed)
                            break
                    rows.append(dict(seed=seed, switch_tick=switch, mapping=mapping, direction=direction))
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            row['index'] = index
        save(directory / f'{batch}-schedules.json', rows)
    save(directory / 'checkpoint-acquired.json', acquired)
    file_names = ('primary-schedules.json', 'reserve-schedules.json', 'checkpoint-acquired.json')
    spec = dict(contract='paired-rapid-native-public-visual-control-v1',
                frozen_sources=source_signature(), dependencies=runtime_dependencies(),
                experimental_sources={name: sha(Path(__file__).with_name(name)) for name in SOURCE_NAMES},
                prepared_files={name: sha(directory / name) for name in file_names},
                acquisition_sources={str(path): sha(path) for path in (checkpoint, receipt)},
                prior_seed_metadata=metadata, excluded_seed_count=len(used) - 2 * len(rows),
                seed_entropy=hex(seed_entropy), cases_per_batch=len(rows),
                baseline_arms=BASELINE_ARMS, candidate_arms=CANDIDATE_ARMS,
                opponent_speed=.65, max_episode_ticks=END_TICK, switches=SWITCH_TICKS,
                changed_mappings=MAPPINGS, original_mapping=NORMAL, gates=GATES,
                budgets=dict(workers=WORKERS, wall_seconds=WALL_SECONDS,
                             max_transitions_per_game=MAX_TRANSITIONS, initial_cameras_per_game=1),
                stable_order='ten consecutive correct effective request orders, credited at the tenth decision',
                population='all designated independent starts; pre-switch endings or absent first opportunities count as failures for first/later reach and speed gates',
                candidate_input='only the public entities, fallback request, own-hit/score/terminal feedback and elapsed seconds',
                reserve_use='independent confirmation after one frozen candidate change; no trial extension')
    save(directory / 'spec.json', spec)
    return sha(directory / 'spec.json')


def verify(directory, digest):
    path = directory / 'spec.json'
    if sha(path) != digest:
        raise ValueError('prepared spec hash mismatch')
    spec = read(path)
    model_bytes()
    if spec['frozen_sources'] != source_signature() or spec['dependencies'] != runtime_dependencies():
        raise ValueError('package sources or dependencies differ from prepared spec')
    for inputs, root in ((spec['experimental_sources'], Path(__file__).parent),
                         (spec['prepared_files'], directory), (spec['acquisition_sources'], Path('/'))):
        for name, expected in inputs.items():
            if sha(root / name) != expected:
                raise ValueError(f'prepared input changed: {name}')
    return spec


@dataclass(frozen=True)
class Candidate:
    path: str
    class_name: str
    sha256: str

    def controller(self):
        if sha(Path(self.path)) != self.sha256:
            raise ValueError('candidate source changed')
        spec = importlib.util.spec_from_file_location('qualification_candidate', self.path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return getattr(module, self.class_name)


def play(case, arm, acquired, candidate=None):
    switch = case['switch_tick']
    unchanged = arm in ('unchanged', 'candidate_unchanged')
    mapping = tuple(case['mapping']) if not unchanged else NORMAL
    expected = tuple(mapping.index(intent) for intent in NORMAL)
    learner = VisualButtonEffects.from_state(acquired)
    controller_type = VisualIntentController
    if arm == 'shorter':
        controller_type = EarlyVisualIntentController
    elif arm in CANDIDATE_ARMS:
        controller_type = candidate.controller()
    else:
        learner = learner.frozen_branch()
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'], POLICY['coefficients'],
                                                         POLICY['scales']), POLICY['require_first_hit'])
    controller = controller_type(policy, learner)
    camera = PongCamera(case['seed'], max_episode_ticks=END_TICK, opponent_speed=.65)
    vision, encoder = load_default(), VisualStateEncoder(64, 64, .02)
    game, counter, events = camera.game, ReturnCounter(), []
    step = game.step

    def instrument(drive, **kwargs):
        prior = counter.opportunities, counter.returns
        tick = int(game.rally_steps[0]) + 1
        outcome = counter.step(game, step, drive, **kwargs)
        if counter.opportunities != prior[0]:
            events.append(dict(tick=tick, hit=counter.returns > prior[1]))
        return outcome

    def observe(frame, previous):
        changes = torch.stack((previous & ~frame, ~previous & frame)).float()
        return snapshot(encoder.encode_entities(*vision.track_entities(changes, frame.float())))

    game.step = instrument
    frame = game.render(64, 64)[0].clone()
    public = observe(frame, torch.zeros_like(frame))
    prefix, physical = hashlib.sha256(), hashlib.sha256()
    for digest in (prefix, physical):
        add_digest(digest, public)
    entered = False
    consecutive = prefix_transitions = probe_cameras = 0
    first_correct = confirmed = None
    prior_order, order_changes = None, []
    while True:
        tick = camera.total_ticks
        after_switch = tick >= switch
        if after_switch and not entered:
            if tick != switch:
                raise ValueError('switch tick falls between camera decisions')
            entered = True
            if arm == 'calibrated_at_switch':
                learner.effects = [acquired['effects'][physical_action] for physical_action in mapping]
        action, diagnostic = controller.choose(public, 1)
        if type(action) is not int or action not in NORMAL:
            raise ValueError('controller must issue a legal integer request')
        active_mapping = mapping if after_switch else NORMAL
        if after_switch:
            order = diagnostic.get('effective_requests')
            order = tuple(order) if order is not None else None
            if order is not None and (len(order) != 3 or set(order) != set(NORMAL)):
                raise ValueError('declared effective requests must be a permutation or null')
            if order != prior_order or tick == switch:
                order_changes.append(dict(tick=tick, order=order))
                prior_order = order
            if order == expected:
                consecutive += 1
                if first_correct is None:
                    first_correct = tick
                if consecutive == CONFIRM_CAMERAS and confirmed is None:
                    confirmed = tick
            else:
                consecutive = 0
            probe_cameras += bool(diagnostic.get('probing'))
        executed = active_mapping[action]
        result = camera.step(executed)
        public = observe(result.frame, frame)
        frame = result.frame
        feedback = dict(own_hits=result.hits, reward=result.reward, terminated=result.terminated,
                        truncated=result.truncated, elapsed_seconds=result.elapsed_seconds)
        controller.observe(public_entities=public, **feedback)
        if not after_switch:
            prefix_transitions += 1
            add_digest(prefix, (action, public, *feedback.values()))
        add_digest(physical, (executed, public, *feedback.values()))
        if result.terminated or result.truncated:
            break
    if not camera.decisions <= MAX_TRANSITIONS or len(events) != counter.opportunities:
        raise ValueError('native camera budget or event accounting mismatch')
    if sum(event['hit'] for event in events) != counter.returns:
        raise ValueError('native hit accounting mismatch')
    return dict(case=case, arm=arm, entered_switch=entered, total_ticks=camera.total_ticks,
                prefix_transitions=prefix_transitions, prefix_sha256=prefix.hexdigest(),
                physical_sha256=physical.hexdigest(), first_correct_tick=first_correct,
                confirmed_correct_tick=confirmed, effective_order_changes=order_changes,
                probe_cameras=probe_cameras, events=events, transitions=camera.decisions,
                reward=result.reward, terminated=result.terminated, truncated=result.truncated)


def worker_init():
    torch.set_num_threads(1)
    signal.signal(signal.SIGALRM, reached_deadline)


def save_game(output, case, arm, acquired, candidate, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        row = play(case, arm, acquired, candidate)
    except TimeoutError:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    save(output / arm / f"game-{case['index']:02d}.json", row)
    return True


def changed_events(row):
    return [event for event in row['events'] if event['tick'] > row['case']['switch_tick']]


def first_hit(row):
    events = changed_events(row)
    return bool(events and events[0]['hit'])


def rate(numerator, denominator):
    return numerator / denominator if denominator else 0.


def metrics(rows):
    exposed = [changed_events(row) for row in rows]
    earlier = [event for row in rows for event in row['events']
               if event['tick'] <= row['case']['switch_tick']]
    later = [event for events in exposed for event in events[1:]]
    all_events = [event for row in rows for event in row['events']]
    all_first = [row['events'][0] for row in rows if row['events']]
    all_later = [event for row in rows for event in row['events'][1:]]
    stable = [(row['confirmed_correct_tick'] - row['case']['switch_tick']) / 120
              for row in rows if row['confirmed_correct_tick'] is not None]
    confirmed_before_contact = sum(bool(events and row['confirmed_correct_tick'] is not None
                                        and row['confirmed_correct_tick'] < events[0]['tick'])
                                   for row, events in zip(rows, exposed, strict=True))
    first = sum(first_hit(row) for row in rows)
    return dict(games=len(rows), entered_switch=sum(row['entered_switch'] for row in rows),
                ended_before_switch=sum(not row['entered_switch'] for row in rows),
                pre_switch_opportunities=len(earlier), pre_switch_misses=sum(not e['hit'] for e in earlier),
                post_switch_without_opportunity=sum(row['entered_switch'] and not events
                                                    for row, events in zip(rows, exposed, strict=True)),
                first_returns=first, first_opportunities=sum(bool(events) for events in exposed),
                first_fraction_planned=rate(first, len(rows)), later_returns=sum(e['hit'] for e in later),
                later_opportunities=len(later), later_fraction=rate(sum(e['hit'] for e in later), len(later)),
                multi_return_starts=sum(sum(e['hit'] for e in events) >= 2 for events in exposed),
                complete_game_first_returns=sum(e['hit'] for e in all_first),
                complete_game_first_opportunities=len(all_first),
                complete_game_later_returns=sum(e['hit'] for e in all_later),
                complete_game_later_opportunities=len(all_later), complete_game_opportunities=len(all_events),
                complete_game_returns=sum(e['hit'] for e in all_events),
                wins=sum(row['reward'] > 0 for row in rows), losses=sum(row['reward'] < 0 for row in rows),
                confirmed_correct=len(stable), confirmed_before_first_contact=confirmed_before_contact,
                confirmed_correct_in_speed_window=sum(t * 120 <= GATES['correct_within_ticks'] for t in stable),
                confirmed_correct_seconds=stable, probe_cameras=sum(row['probe_cameras'] for row in rows),
                transitions=sum(row['transitions'] for row in rows), initial_cameras=len(rows))


def splits(rows):
    return dict(overall=metrics(rows), mapped={str(mapping): metrics([
                row for row in rows if tuple(row['case']['mapping']) == mapping]) for mapping in MAPPINGS},
                switch_ticks={str(tick): metrics([
                    row for row in rows if row['case']['switch_tick'] == tick]) for tick in SWITCH_TICKS})


def read_arm(output, arm):
    return {row['case']['index']: row for row in
            (read(path) for path in sorted((output / arm).glob('game-*.json')))}


def paired_comparison(reference, trial):
    shared = reference.keys() & trial.keys()
    first_reference_only = first_trial_only = first_clock_mismatch = 0
    extra_pre_switch_misses = extra_pre_switch_endings = 0
    matched_prefixes = matched_physical = first_saved_without_confirmed_order = 0
    pre_switch_delay_buckets = Counter()
    for index in sorted(shared):
        a, b = reference[index], trial[index]
        if a['case'] != b['case']:
            raise ValueError('paired receipts do not share the planned case')
        matched_prefixes += a['prefix_sha256'] == b['prefix_sha256']
        matched_physical += a['physical_sha256'] == b['physical_sha256']
        x, y = changed_events(a), changed_events(b)
        x_tick = x[0]['tick'] if x else None
        y_tick = y[0]['tick'] if y else None
        first_clock_mismatch += x_tick != y_tick
        a_first, b_first = first_hit(a), first_hit(b)
        first_reference_only += a_first and not b_first
        first_trial_only += b_first and not a_first
        if b_first and not a_first:
            first_saved_without_confirmed_order += (b['confirmed_correct_tick'] is None
                                                     or b['confirmed_correct_tick'] >= y_tick)
        if x_tick is not None:
            delay = x_tick - a['case']['switch_tick']
            bucket = str(math.ceil(delay / 30) * 30)
            pre_switch_delay_buckets[bucket] += 1
        pre_misses = [sum(not event['hit'] and event['tick'] <= row['case']['switch_tick']
                          for event in row['events']) for row in (a, b)]
        extra_pre_switch_misses += max(0, pre_misses[1] - pre_misses[0])
        extra_pre_switch_endings += a['entered_switch'] and not b['entered_switch']
    discordant = first_reference_only + first_trial_only
    probability = sum(math.comb(discordant, k) for k in range(first_trial_only, discordant + 1)) / 2 ** discordant
    return dict(pairs=len(shared), matched_prefixes=matched_prefixes, matched_physical=matched_physical,
                first_clock_mismatches=first_clock_mismatch, first_reference_only=first_reference_only,
                first_trial_only=first_trial_only, first_gain=rate(first_trial_only - first_reference_only, len(shared)),
                first_gain_p_one_sided=probability, extra_pre_switch_misses=extra_pre_switch_misses,
                extra_pre_switch_endings=extra_pre_switch_endings,
                first_saved_without_confirmed_order=first_saved_without_confirmed_order,
                reference_first_contact_delay_tick_upper_bins=dict(sorted(pre_switch_delay_buckets.items())))


def assess(stats, comparisons, complete):
    gates = GATES
    changed = stats['candidate_changed']['overall']
    unchanged = stats['candidate_unchanged']['overall']
    original = stats['unchanged']['overall']
    shorter = comparisons['shorter_to_candidate_changed']
    pre_change = comparisons['unchanged_to_candidate_changed']
    normal = comparisons['unchanged_to_candidate_unchanged']
    original_first = rate(original['complete_game_first_returns'], original['complete_game_first_opportunities'])
    original_later = rate(original['complete_game_later_returns'], original['complete_game_later_opportunities'])
    unchanged_first = rate(unchanged['complete_game_first_returns'], unchanged['complete_game_first_opportunities'])
    unchanged_later = rate(unchanged['complete_game_later_returns'], unchanged['complete_game_later_opportunities'])
    behavior = dict(complete=complete,
        first=changed['first_fraction_planned'] >= gates['first_planned_fraction'],
        first_gain=shorter['first_gain'] >= gates['first_gain_over_shorter']
                   and shorter['first_gain_p_one_sided'] <= gates['first_gain_p_one_sided'],
        first_mapping_breadth=all(item['first_fraction_planned'] >= gates['per_mapping_first_fraction']
                                  for item in stats['candidate_changed']['mapped'].values()),
        later=changed['later_fraction'] >= gates['later_fraction']
              and changed['later_opportunities'] >= gates['later_min_opportunities'],
        sustained_starts=rate(changed['multi_return_starts'], changed['games']) >= gates['multi_return_fraction'],
        pre_switch=all(pair['extra_pre_switch_misses'] <= gates['max_extra_pre_switch_misses']
                       and pair['extra_pre_switch_endings'] <= gates['max_extra_pre_switch_endings']
                       for pair in (pre_change, normal)),
        unchanged=unchanged_first >= max(gates['unchanged_first_fraction'], original_first-gates['max_unchanged_fraction_loss'])
                  and unchanged_later >= max(gates['unchanged_later_fraction'], original_later-gates['max_unchanged_fraction_loss']))
    speed = rate(changed['confirmed_correct_in_speed_window'], changed['games']) >= gates['correct_within_fraction']
    return dict(behavior_gates=behavior, behavioral_qualification=all(behavior.values()),
                order_speed_gate=speed, full_qualification=all(behavior.values()) and speed)


def run(directory, digest, batch, phase, candidate_path=None, candidate_class=None):
    spec = verify(directory, digest)
    batch_cases = read(directory / f'{batch}-schedules.json')
    acquired = read(directory / 'checkpoint-acquired.json')
    batch_dir = directory / batch
    batch_dir.mkdir(exist_ok=True)
    candidate = None
    arms = BASELINE_ARMS
    if phase == 'candidate':
        if candidate_path is None or candidate_class is None:
            raise ValueError('candidate run requires a frozen source path and its controller class')
        if not read(batch_dir / 'baseline/summary.json')['complete']:
            raise ValueError('complete baseline receipts are required before candidate play')
        arms = CANDIDATE_ARMS
        candidate = Candidate(str(candidate_path), candidate_class, sha(candidate_path))
    output = batch_dir / phase
    output.mkdir(exist_ok=False)
    for arm in arms:
        (output / arm).mkdir()
    start = time.monotonic()
    deadline = start + WALL_SECONDS[phase]
    save(output / 'dispatch.json', dict(spec_sha256=digest, batch=batch, phase=phase, arms=arms,
         candidate=vars(candidate) if candidate else None, deadline=deadline, planned_games=len(arms)*len(batch_cases)))
    errors = []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('spawn'),
                             initializer=worker_init) as pool:
        active = {pool.submit(save_game, output, case, arm, acquired, candidate, deadline): (arm, case['index'])
                  for case in batch_cases for arm in arms}
        for future in as_completed(active):
            arm, index = active[future]
            try:
                if not future.result():
                    errors.append(dict(arm=arm, index=index, reason='timeout'))
            except Exception as error:
                errors.append(dict(arm=arm, index=index, reason=repr(error)))
    verify(directory, digest)
    if candidate and sha(candidate_path) != candidate.sha256:
        raise ValueError('candidate source changed during play')
    baseline = {arm: read_arm(batch_dir / 'baseline', arm) for arm in BASELINE_ARMS}
    rows = baseline | ({arm: read_arm(output, arm) for arm in arms} if candidate else {})
    complete = not errors and all(len(items) == spec['cases_per_batch'] for items in rows.values())
    stats = {arm: splits(list(items.values())) for arm, items in rows.items()}
    comparisons = {f'unchanged_to_{arm}': paired_comparison(rows['unchanged'], items)
                   for arm, items in rows.items() if arm != 'unchanged'}
    oracle = comparisons['unchanged_to_calibrated_at_switch']
    if complete and oracle['matched_physical'] != len(batch_cases):
        raise ValueError('calibrated-at-switch feasibility counterfactual broke physical parity')
    if candidate:
        comparisons.update(shorter_to_candidate_changed=paired_comparison(rows['shorter'], rows['candidate_changed']),
                           frozen_to_candidate_changed=paired_comparison(rows['frozen'], rows['candidate_changed']))
    result = dict(complete=complete, errors=errors, spec_sha256=digest, batch=batch, phase=phase,
                  elapsed_wall_seconds=time.monotonic()-start, arms=stats, comparisons=comparisons,
                  qualification=assess(stats, comparisons, complete) if candidate else None)
    save(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest='command', required=True)
    preparing = phases.add_parser('prepare')
    preparing.add_argument('directory', type=Path)
    preparing.add_argument('checkpoint', type=Path)
    preparing.add_argument('prior_experiments', type=Path)
    running = phases.add_parser('run')
    running.add_argument('directory', type=Path)
    running.add_argument('--spec-sha256', required=True)
    running.add_argument('--batch', choices=('primary', 'reserve'), required=True)
    running.add_argument('--phase', choices=('baseline', 'candidate'), required=True)
    running.add_argument('--candidate', type=Path)
    running.add_argument('--candidate-class')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.directory.resolve(), args.checkpoint.resolve(), args.prior_experiments.resolve()))
    else:
        result = run(args.directory.resolve(), args.spec_sha256, args.batch, args.phase,
                     args.candidate.resolve() if args.candidate else None, args.candidate_class)
        print(json.dumps(dict(complete=result['complete'], errors=result['errors'],
                              qualification=result['qualification']), sort_keys=True))
