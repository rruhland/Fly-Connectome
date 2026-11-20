"""Test RGB reward transfer and delayed visual motor adaptation on a simulated arm."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from body_reaching_arena import DEFAULT_MAX_STEPS, GOAL_TOLERANCE, SUCCESS_TICKS, TICK_SECONDS
from body_strategy_transfer import CueLayout, RELATIONS, RewardAssociation, sample_layout
from camera_motor_learner import CALIBRATION_TICKS, CameraMotorLearner
from camera_reaching_arena import CameraArena, CameraView, Drive, decode_camera
from planar_position_arm import RESET_WRIST_LIMIT


TRANSFER = ('carried', 'reset', 'source_reward_masked', 'opposite')
MOTOR = ('adaptive', 'fixed', 'fresh', 'fresh_at_boundary')
COURSES = ('both', 'camera', 'drive', 'unchanged')
SOURCE_REACHES = 4
WARMUP_REACHES = 4
MOTOR_TEST_REACHES = 3
VISUAL_DWELL = 5
VISUAL_RADIUS = 2.6


def _sample_layout(rng):
    while True:
        layout = sample_layout(rng)
        spacing = min(np.linalg.norm(np.subtract(marker, layout.reference)) for marker in layout.markers)
        if spacing >= .105:
            return layout


def _visible(frame):
    try:
        return decode_camera(frame)
    except ValueError:
        return None


def _ranked_goals(observation):
    distances = np.linalg.norm(observation.markers - observation.reference, axis=1)
    return observation.markers[np.argsort(distances)]


def _physical_holds(arena):
    """Independently reconstruct holds from the recorded physical path."""
    path = np.asarray(arena.trace[1:])
    verified = []
    for relation, goal in enumerate(arena.layout.by_distance()):
        inside = np.linalg.norm(path - goal, axis=1) <= GOAL_TOLERANCE
        if len(inside) < SUCCESS_TICKS:
            continue
        windows = np.convolve(inside, np.ones(SUCCESS_TICKS, dtype=int), mode='valid') == SUCCESS_TICKS
        entrances = np.flatnonzero(windows & ~np.r_[False, windows[:-1]]) + SUCCESS_TICKS
        verified.extend({'tick': int(tick), 'relation': relation, 'reward': int(relation == arena.paying)}
                        for tick in entrances)
    verified.sort(key=lambda event: event['tick'])
    if verified != arena.contacts or arena.success != any(event['reward'] for event in verified):
        raise AssertionError('physical trajectory and simulator contact scorer disagree')
    return verified


def run_trial(arena, motor, memory, guess, seed, *, replay=False):
    """The trial controller receives issued-command history, pixels, and scalar score."""
    motor.begin_trial(seed)
    frame = arena.frame()
    observation = _visible(frame)
    selected = guess if memory.preferred is None else memory.preferred
    first_choice = selected
    visual_hold = 0
    events = []
    misses = int(observation is None)
    while not arena.done:
        error = None if observation is None else _ranked_goals(observation)[selected] - observation.tip
        command = motor.act(error)
        frame, reward = arena.step(command)
        new_observation = _visible(frame)
        if new_observation is None or observation is None:
            displacement = None
        else:
            # Subtract a fixed painted reference so camera pan does not explain arm motion.
            displacement = ((new_observation.tip - new_observation.reference)
                            - (observation.tip - observation.reference))
        motor.observe(displacement)
        observation = new_observation
        if observation is None:
            misses += 1
            visual_hold = 0
            continue
        distances = np.linalg.norm(_ranked_goals(observation) - observation.tip, axis=1)
        visual_hold = visual_hold + 1 if distances[selected] <= VISUAL_RADIUS else 0
        if reward or visual_hold >= VISUAL_DWELL:
            relation = int(distances.argmin()) if reward else selected
            memory.observe(relation, reward)
            events.append({'tick': arena.steps, 'relation': relation, 'reward': reward})
            if not reward:
                selected = 1 - relation
                motor.begin_goal()
                visual_hold = 0
    contacts = _physical_holds(arena)
    rejected = []
    approach_start = 0
    for event in events:
        if event['reward'] == 0:
            verified = any(approach_start < contact['tick'] <= event['tick']
                           and contact['relation'] == event['relation'] and not contact['reward']
                           for contact in contacts)
            rejected.append(dict(event, physically_verified=verified,
                                 rejected_paying_goal=event['relation'] == arena.paying))
            approach_start = event['tick']
    result = {
        'commands': arena.steps, 'success': arena.success, 'first_choice': RELATIONS[first_choice],
        'paying': RELATIONS[arena.paying], 'contacts': contacts, 'reward_events': events,
        'rejections': rejected, 'lost_frames': misses, 'probes': len(motor.probe_ticks),
        'calibrations': motor.calibrations, 'adaptations': motor.adaptations,
        'inferred_delay': None if motor.model is None else motor.model.delay,
        'actual_delay': 0 if arena.drive is None else arena.drive.delay,
        'actions_sha256': sha256(np.asarray(motor.commands, dtype='<f8').tobytes()).hexdigest(),
    }
    if replay:
        result['replay'] = {'layout': asdict(arena.layout), 'camera': asdict(arena.camera),
                            'drive': None if arena.drive is None else asdict(arena.drive),
                            'wrist': arena.wrist, 'actions': np.asarray(motor.commands).tolist(),
                            'probe_ticks': motor.probe_ticks}
    return result


def _source(seed, max_steps):
    rng = np.random.default_rng(seed)
    camera = CameraView.sample(rng)
    rule = int(rng.integers(2))
    memory = RewardAssociation()
    motor = CameraMotorLearner(2)
    history = [run_trial(CameraArena(_sample_layout(rng), rule, camera, max_steps=max_steps),
                         motor, memory, int(rng.integers(2)), int(rng.integers(2**32)))
               for _ in range(SOURCE_REACHES)]
    return rule, memory, history


def _layout_spec(rng):
    return (_sample_layout(rng), float(rng.uniform(-RESET_WRIST_LIMIT, RESET_WRIST_LIMIT)),
            int(rng.integers(2)), int(rng.integers(2**32)))


def _arm(spec, rule, camera, drive, max_steps, motor, memory, replay=False):
    layout, wrist, guess, seed = spec
    arena = CameraArena(layout, rule, camera, max_steps=max_steps, drive=drive, wrist=wrist)
    return run_trial(arena, motor, memory, guess, seed, replay=replay)


def _transfer(seed, rule, learned, source, max_steps, replay):
    rng = np.random.default_rng(seed)
    spec = _layout_spec(rng)
    camera, drive = CameraView.sample(rng, changed=True), Drive.sample(rng, changed=True)
    masked = RewardAssociation()
    for trial in source:
        for event in trial['reward_events']:
            masked.observe(event['relation'], None)
    opposite = None if learned.preferred is None else 1 - learned.preferred
    priors = (learned.export_prior(), RewardAssociation().export_prior(), masked.export_prior(),
              RewardAssociation(opposite).export_prior())
    result = {}
    for condition, prior in zip(TRANSFER, priors, strict=True):
        result[condition] = _arm(spec, rule, camera, drive, max_steps, CameraMotorLearner(3),
                                 RewardAssociation.from_prior(prior), replay=replay and condition in ('carried', 'reset'))
    if result['reset']['actions_sha256'] != result['source_reward_masked']['actions_sha256']:
        raise AssertionError('reward-masked control differs from reset')
    return {'reward_priors': dict(zip(TRANSFER, priors, strict=True)), 'target': result}


def _motor(seed, rule, reward_prior, max_steps, courses, replay):
    rng = np.random.default_rng(seed)
    nominal_camera = CameraView.sample(rng)
    nominal_drive = Drive.sample(rng)
    new_camera = CameraView.sample(rng, changed=True)
    new_drive = Drive.sample(rng, changed=True)
    warmups = [_layout_spec(rng) for _ in range(WARMUP_REACHES)]
    tests = [warmups[-1], *(_layout_spec(rng) for _ in range(MOTOR_TEST_REACHES - 1))]
    memory = RewardAssociation.from_prior(reward_prior)
    motor = CameraMotorLearner(3)
    history = [_arm(spec, rule, nominal_camera, nominal_drive, max_steps, motor, memory) for spec in warmups]
    # A rewarded hold can end a trial before a replacement calibration finishes.
    # Both carried conditions start from the same most recently completed fit.
    fitted_model = motor.last_model
    if fitted_model is None:
        raise AssertionError('warmup did not obtain a calibration for the fixed policy')
    windows = {}
    conditions = {'both': (new_camera, new_drive), 'camera': (new_camera, nominal_drive),
                  'drive': (nominal_camera, new_drive), 'unchanged': (nominal_camera, nominal_drive)}
    for course in courses:
        camera, drive = conditions[course]
        by_condition = {}
        for condition in MOTOR:
            goal_memory = RewardAssociation.from_prior(memory.export_prior())
            model = deepcopy(fitted_model) if condition in ('adaptive', 'fixed') else None
            controller = CameraMotorLearner(3, model=model, frozen=condition == 'fixed')
            by_condition[condition] = [
                _arm(spec, rule, camera, drive, max_steps,
                     CameraMotorLearner(3) if condition == 'fresh' else controller,
                     goal_memory, replay=replay and course == 'both' and index == 0)
                for index, spec in enumerate(tests)]
        windows[course] = by_condition
    return {'warmup': history, 'warmup_model': {'delay': fitted_model.delay, 'effects': fitted_model.effects().tolist()},
            'courses': windows}


def _summarize(trials, max_steps):
    hits = sum(trial['success'] for trial in trials)
    rejections = [event for trial in trials for event in trial['rejections']]
    return {
        'trials': len(trials), 'physical_successes': hits, 'timeouts': len(trials) - hits,
        'first_physical_hold_paid': sum(bool(trial['contacts'] and trial['contacts'][0]['reward']) for trial in trials),
        'correct_first_choice': sum(trial['first_choice'] == trial['paying'] for trial in trials),
        'actual_commands': sum(trial['commands'] for trial in trials),
        'mean_capped_commands': float(np.mean([_cap(trial, max_steps) for trial in trials])),
        'probe_commands': sum(trial['probes'] for trial in trials),
        'lost_frames': sum(trial['lost_frames'] for trial in trials),
        'delay_correct_at_end': sum(trial['inferred_delay'] == trial['actual_delay'] for trial in trials),
        'recalibrations': sum(len(trial['adaptations']) for trial in trials),
        'rejections': len(rejections),
        'unverified_rejections': sum(not event['physically_verified'] for event in rejections),
        'rejections_of_paying_goal': sum(event['rejected_paying_goal'] for event in rejections),
    }


def _cap(trial, max_steps):
    return trial['commands'] if trial['success'] else max_steps + 1


def _paired(values):
    values = np.asarray(values)
    bootstrap = np.random.default_rng(647133).choice(values, (4000, len(values)), replace=True).mean(axis=1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'ahead': int((values > 0).sum()), 'tied': int((values == 0).sum()), 'behind': int((values < 0).sum())}


def run(seeds, *, max_steps=DEFAULT_MAX_STEPS, transfer=True, courses=COURSES):
    seeds = list(map(int, seeds))
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError('provide distinct nonnegative stream seeds')
    if max_steps < SUCCESS_TICKS:
        raise ValueError('provide enough steps for a physical hold')
    started = time.perf_counter()
    streams = []
    for seed in seeds:
        parts = np.random.SeedSequence(seed).spawn(3)
        rule, memory, source = _source(parts[0], max_steps)
        capture = not streams
        stream = {'seed': seed, 'source': source}
        if transfer:
            stream['transfer'] = _transfer(parts[1], rule, memory, source, max_steps, replay=capture)
        if courses:
            stream['motor'] = _motor(parts[2], rule, memory.export_prior(), max_steps, courses, replay=capture)
        streams.append(stream)
    summary = {'source': _summarize([trial for row in streams for trial in row['source']], max_steps)}
    if transfer:
        rows = [stream['transfer']['target'] for stream in streams]
        summary['transfer'] = {
            'groups': {condition: _summarize([row[condition] for row in rows], max_steps) for condition in TRANSFER},
            'paired_savings': {condition: _paired([_cap(row[condition], max_steps) - _cap(row['carried'], max_steps)
                                                  for row in rows]) for condition in TRANSFER[1:]},
        }
    if courses:
        summary['warmup'] = _summarize([trial for stream in streams for trial in stream['motor']['warmup']], max_steps)
        summary['courses'] = {}
        for course in courses:
            rows = [stream['motor']['courses'][course] for stream in streams]
            summary['courses'][course] = {}
            for period, window in (('first', slice(0, 1)), ('later', slice(1, None)), ('all', slice(None))):
                groups = {condition: _summarize([trial for row in rows for trial in row[condition][window]], max_steps)
                          for condition in MOTOR}
                pairs = {}
                for condition in MOTOR[1:]:
                    pairs[condition] = _paired([
                        np.mean([_cap(second, max_steps) - _cap(first, max_steps)
                                 for first, second in zip(row['adaptive'][window], row[condition][window], strict=True)])
                        for row in rows])
                summary['courses'][course][period] = {'groups': groups, 'paired_savings': pairs}
    files = ('run_camera_arm_transfer.py', 'camera_motor_learner.py', 'camera_reaching_arena.py',
             'body_reaching_arena.py', 'body_strategy_transfer.py', 'planar_position_arm.py')
    return {
        'schema': 'rgb-camera-arm-v1', 'tick_seconds': TICK_SECONDS, 'max_steps': max_steps,
        'calibration_commands': CALIBRATION_TICKS, 'visual_dwell_ticks': VISUAL_DWELL, 'visual_radius_px': VISUAL_RADIUS,
        'physical_dwell_ticks': SUCCESS_TICKS, 'physical_radius': GOAL_TOLERANCE,
        'source_reaches_per_stream': SOURCE_REACHES, 'warmup_reaches_per_stream': WARMUP_REACHES,
        'elapsed_seconds': time.perf_counter() - started,
        'source_sha256': {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files},
        'summary': summary, 'streams': streams,
    }


def write_replay(receipt, path, *, comparison):
    """Recreate two panel streams from their exact recorded RGB camera trajectories."""
    from PIL import Image, ImageDraw

    stream = receipt['streams'][0]
    if comparison == 'transfer':
        labels = ('Carried reward memory', 'Fresh reward memory')
        cases = [stream['transfer']['target'][name] for name in ('carried', 'reset')]
    else:
        labels = ('Online motor updates', 'Fixed motor model')
        cases = [stream['motor']['courses']['both'][name][0] for name in ('adaptive', 'fixed')]
    clips = []
    for case in cases:
        rep = case['replay']
        camera = CameraView(**{key: tuple(value) if isinstance(value, list) else value
                               for key, value in rep['camera'].items()})
        drive = Drive(**rep['drive'])
        arena = CameraArena(CueLayout(**rep['layout']), RELATIONS.index(case['paying']), camera,
                            max_steps=receipt['max_steps'], drive=drive, wrist=rep['wrist'])
        digest = sha256(np.asarray(rep['actions'], dtype='<f8').tobytes()).hexdigest()
        if digest != case['actions_sha256']:
            raise AssertionError('replay actions differ from the recorded controller run')
        images = [arena.frame()]
        for action in rep['actions']:
            image, _ = arena.step(action)
            images.append(image)
        if _physical_holds(arena) != case['contacts']:
            raise AssertionError('replay contacts differ from the recorded controller run')
        clips.append(images)
    frames = []
    for tick in range(max(map(len, clips))):
        canvas = Image.new('RGB', (334, 201), (15, 22, 33))
        draw = ImageDraw.Draw(canvas)
        for x, label, clip, trial in zip((5, 169), labels, clips, cases, strict=True):
            index = min(tick, len(clip) - 1)
            canvas.paste(Image.fromarray(clip[index]), (x, 20))
            draw.text((x + 2, 5), label, fill=(245, 248, 251))
            state = 'Reaching' if tick < trial['commands'] else 'Rewarded hold' if trial['success'] else 'Timeout'
            if tick in trial['replay']['probe_ticks']:
                state = 'Motor probe'
            draw.text((x + 2, 184), f'Command {index:02d}  |  {state}', fill=(231, 237, 243))
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[70] * (len(frames) - 1) + [1500], loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--first-seed', type=int, required=True)
    parser.add_argument('--streams', type=int, default=64)
    parser.add_argument('--mode', choices=('all', 'transfer', 'motor'), default='all')
    parser.add_argument('--courses', choices=COURSES, nargs='+', default=COURSES)
    parser.add_argument('--replay-dir', type=Path)
    args = parser.parse_args()
    if args.first_seed < 0 or args.streams < 1:
        parser.error('provide a nonnegative seed and at least one stream')
    result = run(range(args.first_seed, args.first_seed + args.streams), transfer=args.mode != 'motor',
                 courses=() if args.mode == 'transfer' else args.courses)
    with args.output.open('x') as output:
        json.dump(result, output, separators=(',', ':'), allow_nan=False)
    if args.replay_dir:
        args.replay_dir.mkdir(parents=True, exist_ok=True)
        if args.mode != 'motor':
            write_replay(result, args.replay_dir / 'rgb-arm-reward-transfer.gif', comparison='transfer')
        if args.mode != 'transfer' and 'both' in args.courses:
            write_replay(result, args.replay_dir / 'rgb-arm-motor-adaptation.gif', comparison='motor')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
