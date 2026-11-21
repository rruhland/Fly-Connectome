"""Test whether the learned assignment of actuator timing to service roles matters."""
import argparse
from copy import deepcopy
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_trial import check_tapes
from .tt12_online_timing import ROLES, TimingChoice
from .tt12_online_timing_trial import audit_choice, play, restore


def run(site, folder):
    source = folder/f'online-{site}-learned.json.gz'
    original = _read(folder/f'online-{site}-games.json.gz')
    held = next(g for g in original['games'] if g['stage']=='qualification' and g['policy']=='retained')
    if held['start_sha256'] != _hash(source):
        raise ValueError('held game used a different checkpoint')
    arena = TT8ServiceArena(**original['arena'])
    try:
        cache_hall_stereo(arena,reflections=True)
        wiring = TT6Wiring(arena,original['installation_seed'])
        _,_,player = restore(source,arena,'retained')
        before = player.choice.state()
        state = deepcopy(before)
        for field in ('attempts','scored','wins'):
            state[field] = {key:deepcopy(before[field][other]) for key,other in zip(ROLES,reversed(ROLES),strict=True)}
        player.choice = TimingChoice(**state)
        game = play(arena,wiring,player,held['seed'],held['schedule'],'qualification')
    finally:
        arena.close()
    path = folder/f'role-reversal-{site}.json.gz'
    _write(path,dict(arena=original['arena'],installation_seed=original['installation_seed'],checkpoint=_hash(source),
                     original_choice=before,game=game,implementation=_hash(Path(__file__))))
    receipt = _read(path)
    audit_choice(receipt['game'])
    check_tapes(dict(receipt,**receipt['game']))
    print(game['score'],game['metrics'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('site',choices=('c','d'))
    p.add_argument('folder',type=Path)
    a=p.parse_args()
    run(a.site,a.folder)
