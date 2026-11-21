"""Check optical-policy replay and resume the five-axis player for a second full game."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import check_tapes
from .tt10_series_trial import PHASES, _game_summary
from .tt10_window_trial import _point_row
from .tt13_live_trial import load_player, save_player
from .tt16_transfer_game import ROW_FIELDS, case_prefix, select_key
from .tt16_transfer_protocol import load_optical_player, make_arena, setup_path


def play(folder):
    fixture = 'five-0'
    setup = _read(setup_path(folder, fixture))
    primary = case_prefix(folder, fixture, 'coded')
    game = _read(primary.with_suffix('.json.gz'))
    checkpoint = primary.with_name(primary.name+'-player.json.gz')
    saved = _read(checkpoint)
    if (game['checkpoint'] != _hash(checkpoint) or saved['series']['tt16_camera'] != setup['camera']
            or saved['series']['optics'] != setup['optics']):
        raise ValueError('inconsistent saved optical setup')
    arena = make_arena(dict(camera=saved['series']['tt16_camera'], arena=saved['series']['arena']))
    try:
        cache_hall_stereo(arena, reflections=True)
        wiring = TT6Wiring(arena, game['installation_seed'])
        _, _, live, learned = load_optical_player(setup, arena)
        select_key(learned, game['initial_key'])
        live.adapter = learned.models[learned.active]
        for row in game['points']:
            old = _point_row(live, arena, wiring, game['seed'], row['seed'], row['phase'], row['server'])
            if old['action_sha256'] != row['action_sha256'] or old['evaluator'] != row['evaluator']:
                raise ValueError('original closed-loop policy diverged from saved game')

        state, library, resumed, memory = load_player(checkpoint, arena, 'fixed')
        seed = 56327811
        starts = _seeds(seed, 618, 0, 65)['tests']
        phases = match_phases(PHASES,65,[seed,618])
        score = ServiceScore(first_server=-1)
        rows = []
        for i, (start, phase) in enumerate(zip(starts, phases, strict=True)):
            raw = _point_row(resumed, arena, wiring, seed, start, phase, score.server)
            if i < 4:
                ref = _point_row(live, arena, wiring, seed, start, phase, score.server)
                if raw['action_sha256'] != ref['action_sha256'] or raw['evaluator'] != ref['evaluator']:
                    raise ValueError('reloaded strategy differs on a continuation start')
            rows.append({name: raw[name] for name in ROW_FIELDS})
            score.record(raw['evaluator'])
            if score.finished:
                break
        if not score.finished:
            raise ValueError('new full game did not finish')
        endpoint = folder / 'coded-five-resumed-player.json.gz'
        save_player(endpoint,state,library,resumed,memory,1)
    finally:
        arena.close()
    name = folder / 'coded-five-resumed-game.json.gz'
    report = dict(seed=seed, arena=setup['arena'], camera=setup['camera'], installation_seed=game['installation_seed'],
                  start=_hash(checkpoint), checkpoint=_hash(endpoint), verified_rgb_points=len(game['points']),
                  verified_continuation=4, calibration_frames=0, points=rows, score=list(score.score),
                  race_to=11, first_server=-1, finished=True, practice=False, metrics=_game_summary(rows))
    _write(name,report)
    check_tapes(_read(name))
    print('resumed', score.score, report['metrics'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    play(parser.parse_args().folder)
