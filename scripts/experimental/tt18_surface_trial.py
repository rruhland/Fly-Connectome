"""Sequential first-to-eleven games across unseen tables with two bounce regions."""
import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt10_series_trial import PHASES, _game_summary
from .tt10_window_trial import _point_row
from .tt13_live_trial import load_player, save_player
from .tt16_transfer_game import ROW_FIELDS, game_clock
from .tt17_motor_protocol import setup_path
from .tt18_surface import SurfaceMemory, SurfacePlayer


def environments(site, source):
    if source:
        pairs = [(19., 70.), (60., 25.)]
    else:
        pairs = [(90., 15.), (17., 84.)]
    if site == 'five':
        pairs = [(right, left) for left, right in pairs]
    return [dict(left_damping=left, right_damping=right) for left, right in pairs]


def verify(spec, game):
    from .tt18_plant import make_arena
    arena = make_arena(spec)
    try:
        wire = TT6Wiring(arena, spec['wiring'])
        score = ServiceScore(first_server=-1)
        for row in game['points']:
            interface = PointInterface(wire, prepare_point(arena, row['phase']),
                                       seed=game['seed'], server=score.server)
            if row['server'] != score.server:
                raise ValueError('service mismatch')
            interface.reset(seed=row['seed'], render=False)
            for action in row['actions']:
                interface.step(action, render=False)
            if arena.point_result() != row['evaluator'] or interface.applied != row['applied']:
                raise ValueError(f"physical replay diverged on point {row['seed']}")
            if json.dumps(arena.events, sort_keys=True) != json.dumps(row['events'], sort_keys=True):
                raise ValueError(f"event replay diverged on point {row['seed']}")
            score.record(arena.point_result())
        if not score.finished or list(score.score) != game['score']:
            raise ValueError('incomplete or different game score')
    finally:
        arena.close()


def fit_rule(folder):
    # Optimize prediction of the next optical measurement on source streams only.
    streams = [_read(folder / f'surface-{site}-source.json.gz') for site in ('gantry','five')]
    losses = []
    for width in (.075, .20, .42, None):
        for lifetime in (3., 8., 24.):
            errors = []
            for stream in streams:
                memory = SurfaceMemory(width, lifetime)
                for game in stream['games']:
                    for point in game['points']:
                        for event in point['surface_events']:
                            prediction = memory.predict(event['xy'], event['inherited'])
                            errors.append(abs(prediction - event['ratio']))
                            memory.samples.append([*event['xy'], event['ratio']])
            if not errors:
                raise ValueError('source streams contain no visible bounces')
            losses.append(dict(width=width, lifetime=lifetime, loss=float(np.mean(errors)), samples=len(errors)))
    choice = min(losses, key=lambda row: row['loss'])
    _write(folder/'surface-source-rule.json.gz', dict(choice=choice, candidates=losses,
           sources=[_hash(folder/f'surface-{site}-source.json.gz') for site in ('gantry','five')]))
    return choice


def run(folder, site, mode, replica=0):
    from .tt18_plant import make_arena
    source = mode == 'source'
    config = (_read(folder/'surface-source-rule.json.gz')['choice'] if mode == 'transfer'
              else dict(width=.20 if mode == 'local' else None, lifetime=8.))
    memory = SurfaceMemory(config['width'], config['lifetime'])
    setup = setup_path(ROOT/'artifacts/tt17', site+'-stable')
    donor = ROOT / f'artifacts/tt17/motor-{site}-stable-bounded-player.json.gz'
    spec = _read(setup)
    suffix = '' if replica == 0 else f'-{replica}'
    stem = f'surface-{site}-{mode}{suffix}'
    games = []
    folder.mkdir(exist_ok=True, parents=True)
    for index, plant in enumerate(environments(site, source)):
        physical = dict(spec, plant=plant)
        arena = make_arena(physical)
        try:
            cache_hall_stereo(arena, reflections=True)
            if index == 0:
                state, library, player, ports = load_player(donor, arena, 'fixed')
                player.__class__ = SurfacePlayer
                player.surface = memory
                player.surface_mode = 'inherited' if mode in ('source','inherited') else mode
            seed = int(np.random.default_rng([1987403, int(site=='five'), int(source), index, replica]).integers(2**30))
            wire = TT6Wiring(arena, spec['wiring'])
            score = ServiceScore(first_server=-1)
            rows = []
            for start, phase in zip(_seeds(seed, 781, 0, 80)['tests'], match_phases(PHASES, 80, [seed,781])):
                row = _point_row(player, arena, wire, seed, start, phase, score.server)
                rows.append(dict(**{k: row[k] for k in ROW_FIELDS}, surface_events=player.surface_events))
                score.record(row['evaluator'])
                if score.finished:
                    break
            game = dict(seed=seed, plant=plant, score=list(score.score), points=rows,
                        metrics=_game_summary(rows), clock=game_clock(rows, 0, arena.control_dt))
            if not score.finished:
                raise ValueError('point budget exhausted')
        finally:
            arena.close()
        verify(physical, game)
        games.append(game)
        print(stem,index,game['score'],game['metrics']['total_legal_returns'],flush=True)
    checkpoint = folder / f'{stem}-player.json.gz'
    save_player(checkpoint, state, library, player, ports, len(games))
    saved = _read(checkpoint)
    saved['surface_memory'] = memory.state()
    saved['surface_mode'] = player.surface_mode
    _write(checkpoint, saved)
    _write(folder/f'{stem}.json.gz', dict(mode=mode, site=site, games=games,
           donor=_hash(donor), setup=_hash(setup), checkpoint=_hash(checkpoint), replica=replica,
           source_rule=_hash(folder/'surface-source-rule.json.gz') if mode=='transfer' else None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('site', choices=('gantry','five','fit'))
    parser.add_argument('mode', nargs='?', choices=('inherited','global','local','source','transfer'))
    parser.add_argument('--replica', type=int, default=0)
    args = parser.parse_args()
    if args.site == 'fit':
        print(fit_rule(args.folder))
    elif args.mode is None:
        parser.error('select a controller')
    else:
        run(args.folder, args.site, args.mode, args.replica)
