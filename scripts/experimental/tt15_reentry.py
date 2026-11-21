"""Continue optical motor adaptation after transfer back into table tennis."""
import argparse
from pathlib import Path


from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_trial import check_tapes
from .tt10_series_trial import PHASES, _game_summary
from .tt10_window_trial import _point_row
from .tt12_motor_recall import rewire
from .tt13_live_trial import load_player, save_player


def run(folder, site, index, carry):
    source = folder/f'back-{site}-{index}-motion-fit.json.gz'
    fit = _read(source)
    donor = Path('artifacts/tt13')/f'live-{site}-fixed-player.json.gz'
    settings = _read(donor)['series']['arena']
    start = _read(folder/f'back-{site}-{index}-fixed.json.gz')
    seed = start['seed']
    arena = TT8ServiceArena(**settings)
    name = 'search' if carry else 'original'
    prefix = folder / f'reentry-{site}-{index}-{name}'
    try:
        cache_hall_stereo(arena,reflections=True)
        state, library, player, memory = load_player(donor,arena,'remap')
        if carry:
            order, signs = fit['maps']['motion']['key']
            key = (tuple(order),tuple(signs))
            if key not in memory.keys:
                memory.keys.append(key)
                memory.models.append(rewire(memory.source,*key))
            memory.active = memory.keys.index(key)
        wiring = TT6Wiring(arena,fit['ports'])
        starts = _seeds(seed,618,0,65)['tests']
        phases = match_phases(PHASES,65,[seed,618])
        score = ServiceScore(first_server=-1)
        rows = []
        for point,phase in zip(starts,phases,strict=True):
            before = memory.active
            player.adapter = memory.models[before]
            row = _point_row(player,arena,wiring,seed,point,phase,score.server)
            row.update(model_used=before,choice=memory.observe(row['actions'],player.trace))
            rows.append(row)
            score.record(row['evaluator'])
            if score.finished:
                break
        if not score.finished:
            raise ValueError('game incomplete')
        if not carry and rows[0]['action_sha256'] != start['points'][0]['action_sha256']:
            raise ValueError('control began from a different policy')
        path = prefix.with_name(prefix.name+'-player.json.gz')
        save_player(path,state,library,player,memory,1)
        data = dict(seed=seed,arena=settings,installation_seed=fit['ports'],start=_hash(donor),fit=_hash(source),
                    mode=name,checkpoint=_hash(path),score=list(score.score),finished=True,race_to=11,
                    first_server=-1,practice=False,points=rows,metrics=_game_summary(rows))
        _write(prefix.with_suffix('.json.gz'),data)
        check_tapes(_read(prefix.with_suffix('.json.gz')))
        print(site,index,name,score.score,data['metrics']['total_legal_returns'],flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('site',choices=('gantry','five'))
    p.add_argument('index',type=int)
    p.add_argument('mode',choices=('search','original'))
    a=p.parse_args()
    run(a.folder,a.site,a.index,a.mode=='search')
