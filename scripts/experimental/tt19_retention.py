"""Replay the old gantry and native Pong policies before and after SO-101 play."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt3_numeric_retention import _hashes as pong_hashes
from .tt3_pong import assess_retention, state_sha256
from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt7_flight import FlightMemory
from .tt8_pong_retention import evaluate as evaluate_pong, portable as portable_pong
from .tt8_service_arena import ServiceScore
from .tt10_series_trial import PHASES, _game_summary, _restore
from .tt10_window_trial import _point_row
from .tt14_gain_trial import memory_from_checkpoint
from .tt16_transfer_game import ROW_FIELDS, game_clock
from .tt17_allocation import NativeTravelServo
from .tt17_motor_protocol import make_arena, setup_path, wiring_for
from .tt17_motor_trial import verify_physics


TT17 = ROOT / 'artifacts/tt17'
SOURCE = TT17 / 'motor-gantry-coupling-travel-player.json.gz'
SOURCE_GAME = TT17 / 'motor-gantry-coupling-travel.json.gz'
GANTRY_FIXTURE = setup_path(TT17, 'gantry-coupling')
PONG_REFERENCE = ROOT / 'artifacts/tt6/pong-retention.json.gz'
GANTRY_SEED = 732404787


def gantry_spec(seed=GANTRY_SEED):
    spec = _read(GANTRY_FIXTURE)
    spec.update(game_seed=seed, starts=_seeds(seed, 6011, 0, 65)['tests'],
                phases=match_phases(PHASES, 65, [seed, 6011]))
    return spec


def audit_installation(state, index=None):
    """Hash the body and optics that the explicit gantry address selects."""
    stored = state['gain_memory']
    index = stored['active'] if index is None else index
    active = stored['maps'][index]
    series = state['series']
    return dict(active=index, model=active['model'],
                map_sha256=state_sha256(active), body_sha256=state_sha256(state['models'][active['model']]),
                all_gain_memory_sha256=state_sha256(stored), camera_sha256=state_sha256(series['tt16_camera']),
                optics_sha256=state_sha256(series['optics']), arena_sha256=state_sha256(series['arena']))


def run_gantry(checkpoint, spec, *, identity_transform=False, use_gantry_world=False):
    arena = make_arena(spec)
    score = ServiceScore(first_server=-1)
    rows = []
    try:
        state, library, player = _restore(checkpoint, arena)
        world = state.get('tt19_online', {}).get('gantry_world') if use_gantry_world else None
        if use_gantry_world:
            if world is None:
                raise ValueError('the training endpoint has no addressed gantry table evidence')
            player.prior.bounce_values = list(world['bounce'])
            player.memory = FlightMemory('fixed', world['flight_history'], world['flight_impulses'])
        memory = memory_from_checkpoint(state, library, 'hybrid')
        selected = 0 if identity_transform else memory.active
        motor = memory.maps[selected]
        if motor.coupled == identity_transform or state['series']['tt16_camera'] != spec['camera']:
            raise ValueError('checkpoint did not contain the addressed gantry and its matching optics')
        player.adapter = motor.base
        player._make_servo = lambda: NativeTravelServo(motor, player.timing_memory, learn=player.adjust)
        cache_hall_stereo(arena, reflections=True)
        wiring = wiring_for(arena, spec)
        render = arena.render_cameras

        def digest_frames(*args, **kwargs):
            frames = render(*args, **kwargs)
            for frame in frames:
                digest.update(np.ascontiguousarray(frame))
            return frames

        arena.render_cameras = digest_frames
        for start, phase in zip(spec['starts'], spec['phases'], strict=True):
            digest = sha256()
            raw = _point_row(player, arena, wiring, spec['game_seed'], start, phase, score.server)
            rows.append(dict({key: raw[key] for key in ROW_FIELDS}, rgb_sha256=digest.hexdigest()))
            score.record(raw['evaluator'])
            if score.finished:
                break
        if not score.finished:
            raise ValueError('the held gantry game exhausted its point schedule')
    finally:
        arena.close()
    game = dict(seed=spec['game_seed'], points=rows, score=list(score.score), metrics=_game_summary(rows),
                clock=game_clock(rows, 0, spec['control_dt']))
    receipt = dict(schema='tt19-gantry-retention-game-v1', checkpoint_sha256=_hash(checkpoint),
                   physical_setup_sha256=_hash(GANTRY_FIXTURE), address='gantry',
                   identity_transform=identity_transform, world_address='gantry' if use_gantry_world else 'shared',
                   stored_world_sha256=None if world is None else state_sha256(world),
                   selected_installation=audit_installation(state, selected), games=[game])
    receipt = json.loads(json.dumps(receipt, allow_nan=False))
    verify_physics(receipt, spec)
    receipt['replayed_points'] = len(rows)
    return receipt


def compare_gantry(before, after):
    if before['physical_setup_sha256'] != after['physical_setup_sha256']:
        raise ValueError('the gantry installation differs before and after training')
    first, second = before['games'][0], after['games'][0]
    rows = list(zip(first['points'], second['points']))
    if first['seed'] != second['seed'] or any((a['seed'], a['phase']) != (b['seed'], b['phase']) for a, b in rows):
        raise ValueError('the gantry starts or disturbance schedule changed')
    matched = [(a, b) for a, b in rows if a['server'] == b['server']]
    return dict(before={'score': first['score'], **first['metrics']}, after={'score': second['score'], **second['metrics']},
                same_address=before['selected_installation'] == after['selected_installation'],
                overlapping_starts=len(rows), matched_servers=len(matched),
                exact_rgb_tapes=sum(a['rgb_sha256'] == b['rgb_sha256'] for a, b in matched),
                exact_action_tapes=sum(a['action_sha256'] == b['action_sha256'] for a, b in matched),
                exact_outcomes=sum(a['evaluator'] == b['evaluator'] for a, b in matched),
                replayed_points=[before['replayed_points'], after['replayed_points']])


def run_pong(checkpoint):
    reference = _read(PONG_REFERENCE)
    hashes = pong_hashes()
    if any(reference['implementation_sha256'][name] != digest for name, digest in hashes.items()):
        raise ValueError('the native Pong implementation differs from the previously qualified schedule')
    portable = portable_pong(checkpoint)
    schedule = reference['schedule']
    routed = evaluate_pong(schedule, portable)
    span = portable.get('motor_span', {})
    shared = dict(memory=span.get('shared', portable['memory']), roles=portable['roles'])
    profiles = span.get('profiles', {})
    common = evaluate_pong(schedule, shared)
    return dict(schema='tt19-pong-retention-run-v1', checkpoint_sha256=_hash(checkpoint),
                schedule_sha256=state_sha256(schedule), full_state_sha256=state_sha256(portable),
                shared_state_sha256=state_sha256(shared), rank_one_sha256=state_sha256(profiles.get(1, profiles.get('1'))),
                routed=routed, shared_only=common)


def compare_pong(before, after):
    if before['schedule_sha256'] != after['schedule_sha256']:
        raise ValueError('Pong controls differ before and after training')
    result = {}
    for name in ('routed', 'shared_only'):
        original, carried = before[name]['points'], after[name]['points']
        receipt = assess_retention(original, carried)
        receipt['exact_action_tapes'] = sum(old['action_sha256'] == new['action_sha256']
                                            for old, new in zip(original, carried, strict=True))
        receipt['exact_outcomes'] = sum((old['score'], old['contacts']) == (new['score'], new['contacts'])
                                        for old, new in zip(original, carried, strict=True))
        result[name] = receipt
    result.update(same_full_state=before['full_state_sha256'] == after['full_state_sha256'],
                  same_shared_state=before['shared_state_sha256'] == after['shared_state_sha256'],
                  same_rank_one_profile=before['rank_one_sha256'] == after['rank_one_sha256'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('gantry', 'pong', 'gantry-source-audit'))
    parser.add_argument('checkpoint', nargs='?', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--identity-transform', action='store_true',
                        help='On gantry, select its original identity map with the same servo and task state')
    parser.add_argument('--use-gantry-world', action='store_true',
                        help='Select table evidence from the addressed gantry slot in the training endpoint')
    args = parser.parse_args()
    if args.kind == 'gantry-source-audit':
        archived = _read(SOURCE_GAME)
        if archived['checkpoint'] != _hash(SOURCE):
            raise ValueError('source player does not match the previously won gantry game')
        verify_physics(archived, _read(GANTRY_FIXTURE))
        print(f"{len(archived['games'][0]['points'])} archived source tapes replayed", flush=True)
        return
    if args.output is None:
        parser.error('--output is required for an evaluation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == 'gantry':
        result = run_gantry(args.checkpoint, gantry_spec(), identity_transform=args.identity_transform,
                            use_gantry_world=args.use_gantry_world)
        print(result['games'][0]['score'], result['games'][0]['metrics'], flush=True)
    else:
        import torch

        torch.set_num_threads(1)
        result = run_pong(args.checkpoint)
        print({key: result[key]['summary'] for key in ('routed', 'shared_only')}, flush=True)
    _write(args.output, result)


if __name__ == '__main__':
    main()
