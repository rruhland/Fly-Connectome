"""Train on ordinary rival-served points and test the learned choice in full games."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import check_tapes
from .tt10_series_trial import PHASES, _game_summary
from .tt10_window_trial import _point_row
from .tt13_live_trial import load_player, save_player
from .tt16_transfer_game import ROW_FIELDS
from .tt16_transfer_protocol import make_arena, setup_path
from .tt17_strike import OFFSETS, StrikeMemory, StrikePlayer


def origin(fixture):
    root = Path('artifacts/tt16')
    if fixture == 'five-0':
        return root/'coded-five-resumed-player.json.gz'
    return root/f'transfer-{fixture}-coded-player.json.gz'


def run(folder, fixture, mode, replica):
    site = fixture.split('-')[0]
    path = origin(fixture)
    spec = _read(setup_path(Path('artifacts/tt16'), fixture))
    arena = make_arena(spec)
    seed = int(np.random.default_rng([818265398, int(site=='five'), int(fixture[-1]), replica,
                                      int(mode=='practice')]).integers(0, 2**31-1))
    prefix = folder/f'strike-{fixture}-{mode}-{replica}'
    practice = mode == 'practice'
    source = folder/f'strike-{site}-0-practice-0.json.gz'
    if mode == 'optical':
        source = folder/f'strike-{site}-optical-practice.json.gz'
    try:
        cache_hall_stereo(arena,reflections=True)
        state, library, player, memory = load_player(path,arena,'fixed')
        player.__class__ = StrikePlayer
        player.strikes = (StrikeMemory() if practice or mode=='base'
                          else StrikeMemory(**_read(source)['learned']))
        player.strike_mode = mode
        wiring = TT6Wiring(arena,spec['wiring'])
        score = ServiceScore(first_server=-1)
        n = 36 if practice else 65
        phases = match_phases(PHASES,n,[seed,3739])
        actions = []
        if practice:
            rng = np.random.default_rng([seed,7351])
            for _ in range(n//len(OFFSETS)):
                actions.extend(rng.permutation(len(OFFSETS)).tolist())
        rows = []
        for i, start in enumerate(_seeds(seed,2841,0,n)['tests']):
            player.practice_action = actions[i] if practice else None
            raw = _point_row(player,arena,wiring,seed,start,phases[i],-1 if practice else score.server)
            row = {k:raw[k] for k in (*ROW_FIELDS, 'strike_choice')}
            if mode == 'optical':
                row['visual_crossings'] = raw['visual_crossings']
            rows.append(row)
            if not practice:
                score.record(raw['evaluator'])
                if score.finished:
                    break
        if not practice and not score.finished:
            raise ValueError('game incomplete')
        checkpoint = prefix.with_name(prefix.name+'-player.json.gz')
        save_player(checkpoint,state,library,player,memory,int(not practice))
        saved = _read(checkpoint)
        saved['strike_memory'] = asdict(player.strikes)
        saved['strike_mode'] = mode
        _write(checkpoint,saved)
    finally:
        arena.close()
    result = dict(mode=mode,seed=seed,source=_hash(path),fixture=fixture,arena=spec['arena'],
                  installation_seed=spec['wiring'],camera=spec['camera'],points=rows,practice=practice,
                  learned=asdict(player.strikes),metrics=_game_summary(rows),checkpoint=_hash(checkpoint),
                  practice_sha256=_hash(source) if not practice and mode!='base' else None)
    if not practice:
        result.update(first_server=-1,race_to=11,score=list(score.score),finished=True)
    receipt=prefix.with_suffix('.json.gz')
    _write(receipt,result)
    check_tapes(_read(receipt))
    print(prefix.name,len(player.strikes.observations),result.get('score'),
          result['metrics']['total_legal_returns'],result['metrics']['two_return_points'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('fixture',choices=('gantry-0','gantry-1','five-0','five-1'))
    p.add_argument('mode',choices=('practice','base','global','frozen','online','optical'))
    p.add_argument('--replica',type=int,default=0)
    a=p.parse_args()
    run(a.folder,a.fixture,a.mode,a.replica)
