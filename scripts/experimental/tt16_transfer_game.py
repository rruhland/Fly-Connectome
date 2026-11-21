"""Qualify frozen transfer, the same probe alone, and direct online remapping."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import check_tapes
from .tt10_series_trial import _game_summary
from .tt10_window_trial import _point_row
from .tt12_motor_recall import rewire
from .tt13_live_trial import save_player
from .tt16_transfer_bridge import as_json, bridge_path, costs
from .tt16_transfer_protocol import FIXTURES, load_optical_player, make_arena, setup_path


REQUIRED_MODES = ('search_pitch', 'probe_only', 'table_online')
ROW_FIELDS = (
    'seed', 'server', 'phase', 'steps', 'public_reward', 'events', 'evaluator', 'applied', 'actions',
    'action_sha256', 'points_for', 'points_against', 'player_returns', 'rival_returns', 'first_legal_return_step',
    'summary',
)


def case_prefix(folder, fixture, mode):
    return folder / f'transfer-{fixture}-{mode}'


def game_clock(rows, prior_frames, dt):
    before = 0
    first = None
    for row in rows:
        step = row['first_legal_return_step']
        if first is None and step is not None:
            if not 1 <= step <= row['steps']:
                raise ValueError('first return event is outside its point tape')
            first = before + step
        before += row['steps']
    total = prior_frames + before
    points = sum(row['points_for'] for row in rows)
    returns = sum(row['player_returns'] for row in rows)
    return dict(game_frames=before, prep_frames=prior_frames, total_frames=total, total_seconds=total * dt,
                table_frames_to_first_return=first, frames_to_first_return=None if first is None else prior_frames + first,
                seconds_to_first_return=None if first is None else (prior_frames + first) * dt,
                points_per_1000_frames=1000 * points / total,
                legal_returns_per_1000_frames=1000 * returns / total)


def select_key(memory, key):
    normalized = tuple(tuple(part) for part in key)
    if normalized not in memory.keys:
        memory.models.append(rewire(memory.source, *normalized))
        memory.keys.append(normalized)
    memory.active = memory.keys.index(normalized)


def verify(receipt, spec, normalized_key=None):
    check_tapes(receipt)
    for row, seed, phase in zip(receipt['points'], spec['starts'], spec['phases']):
        if (row['seed'], row['phase']) != (seed, phase):
            raise ValueError('a point delivered a different ball or phase')
    if receipt['metrics'] != _game_summary(receipt['points']):
        raise ValueError('game summary does not follow the action tape')
    clock = game_clock(receipt['points'], receipt['clock']['prep_frames'], spec['control_dt'])
    if receipt['clock'] != clock:
        raise ValueError('accounting differs from frame and event tape')
    if receipt['mode'] != 'table_online':
        if any(row['model_used'] != receipt['points'][0]['model_used'] for row in receipt['points']):
            raise ValueError('frozen game switched motor assignments')
        return

    arena = make_arena(spec)
    try:
        _, _, _, memory = load_optical_player(spec, arena, 'remap')
        if normalized_key is not None:
            select_key(memory, normalized_key)
        for row in receipt['points']:
            if row['model_used'] != memory.active:
                raise ValueError('scored point used the wrong optical remap')
            trace = [dict(blade=xyz, blade_pitch=pitch) for xyz, pitch in row['motor_sightings']]
            if as_json(memory.observe(row['actions'], trace)) != row['choice']:
                raise ValueError('optical remapping choice differs on replay')
    finally:
        arena.close()


def play_case(folder, fixture, mode, *, key=None, prep_frames=0, prep_effort=None, prep_receipt=None,
              extra_preparation=None):
    setup = setup_path(folder, fixture)
    spec = _read(setup)
    preparation = dict(optics=costs([action for row in spec['optics_probes'] for action in row['actions']]))
    provenance = {'setup': _hash(setup), 'donor': spec['donor_sha256']}
    if mode in ('search_pitch', 'probe_only'):
        bridge = bridge_path(folder, fixture)
        evidence = _read(bridge)
        if evidence['setup_sha256'] != provenance['setup']:
            raise ValueError('search and pitch evidence uses different optics')
        key = evidence['keys'][mode]
        if mode == 'search_pitch':
            preparation['search'] = costs(evidence['search']['actions'])
        preparation['pitch'] = costs(evidence['pitch']['actions'])
        provenance['bridge'] = _hash(bridge)
    elif prep_frames:
        preparation['calibration'] = dict(prep_effort or {}, frames=prep_frames)
    if extra_preparation:
        preparation.update(extra_preparation)
    if prep_receipt is not None:
        provenance['calibration'] = _hash(prep_receipt)
    total_prep = sum(cost['frames'] for cost in preparation.values())
    prefix = case_prefix(folder, fixture, mode)
    path = prefix.with_name(prefix.name + '-player.json.gz')
    arena = make_arena(spec)
    adaptive = mode == 'table_online'
    try:
        cache_hall_stereo(arena, reflections=True)
        state, library, player, memory = load_optical_player(spec, arena, 'remap' if adaptive else 'fixed')
        if key is not None:
            select_key(memory, key)
        initial = memory.keys[memory.active]
        wiring = TT6Wiring(arena, spec['wiring'])
        scorer = ServiceScore(first_server=-1)
        rows = []
        for seed, phase in zip(spec['starts'], spec['phases'], strict=True):
            model = memory.active
            player.adapter = memory.models[model]
            raw = _point_row(player, arena, wiring, spec['game_seed'], seed, phase, scorer.server)
            row = {name: raw[name] for name in ROW_FIELDS}
            row['model_used'] = model
            if adaptive:
                row['choice'] = memory.observe(raw['actions'], player.trace)
                row['motor_sightings'] = [[frame['blade'], frame['blade_pitch']] for frame in player.trace]
            rows.append(row)
            scorer.record(row['evaluator'])
            if scorer.finished:
                break
        if not scorer.finished:
            raise ValueError('the score did not finish within the 65-start budget')
        endpoint = memory.keys[memory.active]
        save_player(path, state, library, player, memory, 1)
    finally:
        arena.close()
    receipt = dict(schema='tt16-transfer-game-v1', fixture=fixture, mode=mode,
                   seed=spec['game_seed'], installation_seed=spec['wiring'], arena=spec['arena'],
                   camera=spec['camera'], start=spec['donor_sha256'], provenance=provenance,
                   initial_key=initial, final_key=endpoint, checkpoint=_hash(path),
                   preparation=preparation, game_cost=costs([action for row in rows for action in row['actions']]),
                   clock=game_clock(rows, total_prep, spec['control_dt']),
                   score=list(scorer.score), metrics=_game_summary(rows),
                   first_server=-1, race_to=11, finished=True, practice=False, points=rows)
    output = prefix.with_suffix('.json.gz')
    _write(output, receipt)
    receipt = _read(output)
    verify(receipt, spec, key)
    receipt['replayed_points'] = len(rows)
    _write(output, receipt)
    print(fixture, mode, scorer.score, receipt['metrics']['total_legal_returns'],
          receipt['clock']['frames_to_first_return'], flush=True)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('fixture', choices=FIXTURES)
    p.add_argument('mode', choices=REQUIRED_MODES)
    args = p.parse_args()
    play_case(args.folder, args.fixture, args.mode)
