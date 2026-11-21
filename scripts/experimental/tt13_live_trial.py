"""Play through unannounced motor rewiring and restoration in real scored rallies."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_trial import check_tapes
from .tt10_series_trial import PHASES, _game_summary, _restore, _save
from .tt10_window_trial import _point_row
from .tt13_live_ports import PortMemory


def load_player(path, arena, mode):
    state, library, player = _restore(path,arena)
    memory = PortMemory(player.adapter,mode)
    saved = state.get('port_memory')
    if saved:
        memory.models = [library.models[index] for index in saved['indices']]
        memory.source = memory.models[0]
        memory.keys = [(tuple(k[0]),tuple(k[1])) if isinstance(k,list) else k for k in saved['keys']]
        memory.active = saved['active']
    return state, library, player, memory


def save_player(path, state, library, player, memory, count):
    entry = {k:v for k,v in state['series'].items() if k not in ('completed','role_rows','flight_history','flight_impulses')}
    indices = []
    for model in memory.models:
        index = next((i for i,m in enumerate(library.models) if m is model),None)
        if index is None:
            index = len(library.models)
            library.models.append(model)
        indices.append(index)
    entry['model'] = indices[memory.active]
    _save(path,state,library,player,entry,state['series']['completed']+count)
    result = _read(path)
    result['port_memory'] = dict(active=memory.active, indices=indices, keys=memory.keys)
    _write(path,result)


def verify(data):
    for g in data['games']:
        score = ServiceScore(first_server=-1)
        for row in g['points']:
            if row['server'] != score.server:
                raise ValueError('service order diverged')
            score.record(row['evaluator'])
        if not score.finished or list(score.score) != g['score']:
            raise ValueError('incomplete game')
        for installation in {r['installation'] for r in g['points']}:
            points = [r for r in g['points'] if r['installation']==installation]
            check_tapes(dict(seed=g['seed'],arena=data['arena'],installation_seed=installation,points=points))


def run(start, output, checkpoint, mode, seed, new_ports, games, stable):
    arena_info = _read(start)['series']['arena']
    arena = TT8ServiceArena(**arena_info)
    data = dict(arena=arena_info, start=_hash(start), mode=mode, seed=seed, stable=stable, new_ports=new_ports,
                implementation=_hash(Path(__file__).with_name('tt13_live_ports.py')), games=[])
    try:
        cache_hall_stereo(arena,reflections=True)
        state, library, player, memory = load_player(start,arena,mode)
        native = arena_info['seed']
        for j,game_seed in enumerate(_seeds(seed,9387,0,games)['tests']):
            starts = _seeds(game_seed,6011,0,65)['tests']
            phases = match_phases(PHASES,65,[game_seed,6011])
            score = ServiceScore(first_server=-1)
            rows = []
            for i,(point_seed,phase) in enumerate(zip(starts,phases,strict=True)):
                # Only the environment knows the switch and the physical port map.
                changed = not stable and ((j%2==0) == (i>=4))
                wires = TT6Wiring(arena,new_ports if changed else native)
                player.adapter = memory.models[memory.active]
                old = memory.active
                row = _point_row(player,arena,wires,game_seed,point_seed,phase,score.server)
                choice = memory.observe(row['actions'],player.trace)
                row.update(choice=choice, model_used=old, installation=new_ports if changed else native)
                rows.append(row)
                score.record(row['evaluator'])
                if score.finished:
                    break
            if not score.finished:
                raise ValueError('game exceeded 65 points')
            data['games'].append(dict(seed=game_seed,score=score.score,points=rows,metrics=_game_summary(rows)))
            _write(output,data)
            if checkpoint:
                save_player(checkpoint,state,library,player,memory,1)
                # Resume from actual storage for the next game.
                state,library,player,memory = load_player(checkpoint,arena,mode)
            print(j,score.score,len(rows),len(memory.models),memory.active,flush=True)
    finally:
        arena.close()
    verify(_read(output))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('start',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--mode',required=True,choices=('fixed','refit','remap'))
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--ports',required=True,type=int)
    p.add_argument('--games',type=int,default=2)
    p.add_argument('--stable',action='store_true')
    a=p.parse_args()
    run(a.start,a.output,a.checkpoint,a.mode,a.seed,a.ports,a.games,a.stable)
