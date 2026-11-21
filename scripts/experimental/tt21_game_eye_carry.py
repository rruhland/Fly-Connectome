"""Carry only the delay learned in ordinary play to a different physical camera."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt20_camera_carry import summarize
from .tt20_curriculum import CHECKPOINT, OUTPUT as TT20, configuration, model_arena
from .tt20_fixture import TT20Arena
from .tt20_latency import CommandPipe
from .tt20_natural import NaturalViewTransport
from .tt21_game_delay import GameDelayPlayer
from .tt21_game_delay_trial import OUTPUT as PARENT, SEEDS, play_point


OUTPUT = ROOT / 'artifacts' / 'tt21' / 'game-eye-carry.json.gz'
TARGET_SEEDS = {'left': 794313037, 'wrist': 1862730206}
SOURCE_EYES = {'left': 'wrist', 'wrist': 'left'}
CONDITIONS = ('discard', 'carry')


def source_checkpoint(receipt, eye):
    """Stop at the first ordinary point played under changed transport."""
    game_seed = SEEDS[0]
    run = next(pair for pair in receipt['results'][eye] if pair['seed'] == game_seed)
    index = receipt['switch_after']
    stop = index + 1
    adaptive = run['games']['adaptive']['points'][:stop]
    frozen = run['games']['frozen']['points'][:stop]
    for source, control in zip(adaptive, frozen, strict=True):
        keys = ('seed', 'role', 'actions', 'applied', 'rgb_sha256', 'events', 'outcome', 'score')
        if any(source[key] != control[key] for key in keys):
            raise ValueError('source branches did not share exactly the same ordinary experience')
    point = adaptive[-1]
    learned = point['learning']
    before = point['lead']
    if (len(adaptive) != stop or point['evaluator_delay'] == adaptive[0]['evaluator_delay']
            or any(p['evaluator_delay'] != adaptive[0]['evaluator_delay'] for p in adaptive[:-1])
            or learned['status'] != 'accepted' or learned['delay'] == before):
        raise ValueError('the first changed point did not identify a new delay')
    return dict(eye=eye, game_seed=game_seed, first_changed_point=stop, point_seed=point['seed'],
                initial_lead=before, learned_lead=learned['delay'], evaluator_delay=point['evaluator_delay'],
                same_exposure=True, ordinary_points=stop, ordinary_frames=sum(len(p['actions']) for p in adaptive),
                changed_frames=len(point['actions']), changed_rendered=learned['rendered'],
                changed_inference_seconds=learned['wall_seconds'],
                prefix_nominal_renders=sum(p['learning']['rendered'] for p in adaptive),
                prefix_inference_seconds=sum(p['learning']['wall_seconds'] for p in adaptive),
                changed_rgb_sha256=point['rgb_sha256'])


def target_schedule(receipt, count):
    source_seeds = {point['seed'] for pairs in receipt['results'].values() for run in pairs
                    for game in run['games'].values() for point in game['points']}
    plan = {eye: np.random.default_rng(seed).integers(0, 2**31, count).tolist() for eye, seed in TARGET_SEEDS.items()}
    starts = [seed for seeds in plan.values() for seed in seeds]
    if len(set(starts)) != len(starts) or source_seeds.intersection(starts):
        raise ValueError('the prechosen target starts overlap each other or the ordinary source games')
    return plan


def action_digest(tape):
    return sha256(np.asarray(tape, dtype='<f8').tobytes()).hexdigest()


def verify_points(data):
    """Reconstruct the physical state and fixed FIFO in fresh evaluator arenas."""
    started = perf_counter()
    replays = 0
    rendered = 0
    for eye, section in data['destinations'].items():
        arena = TT20Arena(**data['arena'], view_mode=eye)
        hidden = section['source']['evaluator_delay']
        try:
            for condition, points in section['conditions'].items():
                for index, row in enumerate(points):
                    arena.reset(seed=row['seed'], server=-1, render=False)
                    controls = len(row['actions'])
                    if action_digest(row['actions']) != row['request_sha256'] or action_digest(row['applied']) != row['applied_sha256']:
                        raise ValueError('the saved request or actuation tape changed')
                    rgb = sha256() if index == 0 else None
                    for tick, (request, applied) in enumerate(zip(row['actions'], row['applied'], strict=True)):
                        expected = np.zeros(arena.action_size) if tick < hidden else np.clip(row['actions'][tick - hidden], -1., 1.)
                        if expected.tolist() != applied:
                            raise ValueError('the applied tape changed the fixed evaluator FIFO')
                        if rgb is not None:
                            images = arena.render_cameras()
                            if tuple(images) != (eye,):
                                raise ValueError('the replay received the wrong camera stream')
                            rgb.update(eye.encode())
                            rgb.update(images[eye])
                        _, reward, done, capped, _ = arena.step(applied, render=False)
                        if (done or capped) != (tick == controls - 1):
                            raise ValueError('termination changed on physical replay')
                    if (arena.point_result() != row['outcome'] or reward != row['reward']
                            or digest_json(arena.events) != digest_json(row['events'])):
                        raise ValueError(f"the verified score or event trace changed at {row['seed']}")
                    if rgb is not None:
                        if rgb.hexdigest() != row['rgb_sha256']:
                            raise ValueError('the independently rendered destination RGB tape changed')
                        rendered += controls
                    replays += 1
                print('replayed', eye, condition, len(points), flush=True)
        finally:
            arena.close()
    return dict(points=replays, exact_rgb_frames=rendered, wall_seconds=perf_counter() - started)


def summarize_target(points):
    result = summarize(points)
    duration = sum(point['wall_seconds'] for point in points)
    result.update(contacts=sum(point['outcome']['contacts'] for point in points),
                  wall_seconds=duration, median_point_wall_seconds=float(np.median([point['wall_seconds'] for point in points])),
                  wall_ms_per_tick=1000 * duration / result['frames'])
    return result


def run(source_path, output, count):
    receipt = _read(source_path)
    for source_key, path in (('factory_sha256', TT20/'curriculum-factory.json.gz'),
                             ('atlas_sha256', TT20/'curriculum-dry-catalog.npz'), ('source_sha256', CHECKPOINT)):
        if _hash(path) != receipt[source_key]:
            raise ValueError('the common controller source differs from the ordinary source games')
    schedule = target_schedule(receipt, count)
    programs, catalogue, law, response, candidate = configuration(TT20)
    if int(candidate) != receipt['candidate']:
        raise ValueError('the source and destination use different factory serve banks')
    data = dict(schema='tt21-ordinary-game-eye-carry-1', source=source_path.name,
                source_sha256=_hash(source_path), arena=receipt['arena'], held_starts_per_eye=count,
                schedule=schedule, schedule_generators=TARGET_SEEDS, target_nominal_renders=0,
                target_calibration_points=0, destination_learning=False, destinations={})
    player = GameDelayPlayer(programs, catalogue, law, response, model_arena())
    try:
        for eye, seeds in schedule.items():
            source = source_checkpoint(receipt, SOURCE_EYES[eye])
            section = data['destinations'][eye] = dict(source=source, conditions={i: [] for i in CONDITIONS})
            arena = TT20Arena(**receipt['arena'], view_mode=eye)
            try:
                lens = arena.lens_intrinsics()[eye]
                leads = dict(discard=source['initial_lead'], carry=source['learned_lead'])
                for index, seed in enumerate(seeds):
                    order = CONDITIONS if index % 2 == 0 else CONDITIONS[::-1]
                    pair = {}
                    for condition in order:
                        player.lead = leads[condition]
                        # Every target starts with the same fresh single-eye registration.
                        # The source's only live contribution is the installed scalar lead.
                        camera = NaturalViewTransport(eye, lens_fov={eye: lens}, law=law)
                        row = play_point(arena, player, camera, eye, seed, -1, int(candidate),
                                         CommandPipe(source['evaluator_delay']), None)
                        row.update(request_sha256=action_digest(row['actions']), applied_sha256=action_digest(row['applied']))
                        section['conditions'][condition].append(row)
                        pair[condition] = row
                    discard, carry = (pair[branch] for branch in CONDITIONS)
                    if discard['optical'][0] != carry['optical'][0]:
                        raise ValueError('the opening optical observation differs across the matched pair')
                    print(eye, index + 1, seed, tuple((point['reward'], point['outcome']['legal_returns'])
                                                     for point in (discard, carry)), flush=True)
                    _write(output, data)
            finally:
                arena.close()
            section['summary'] = {name: summarize_target(points) for name, points in section['conditions'].items()}
            section['matched'] = dict(action_changes=sum(a['actions'] != b['actions'] for a, b in
                                                         zip(*section['conditions'].values(), strict=True)))
    finally:
        player.player.close()
    _write(output, data)
    data['replay'] = verify_points(data)
    _write(output, data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PARENT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--count', type=int, choices=(8, 12), default=12)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.verify:
        print(verify_points(_read(args.output)))
    else:
        completed = run(args.source, args.output, args.count)
        print({eye: section['summary'] for eye, section in completed['destinations'].items()})
