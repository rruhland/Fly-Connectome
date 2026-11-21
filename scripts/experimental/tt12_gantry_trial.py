"""Compare serving policies at the installation that previously failed every toss."""
import argparse
import json

import numpy as np
from copy import deepcopy
from pathlib import Path

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import TT8ServiceArena
from .tt7_flight import FlightMemory
from .tt8_service_drive import ServiceDrive
from .tt9_precontact_registration import RegistrationMemory
from .tt8_service_player import ServiceMemory, ServicePlayer
from .tt8_service_trial import check_tapes
from .tt10_series_trial import PHASES, SOURCE, _restore_tracker
from .tt10_window_trial import _player, _point_row
from .tt12_feasible import FeasiblePlayer


HISTORY = ROOT/'artifacts/tt11/stroke-game-gantry.json.gz'


def run(start, output, seed, modes, baseline):
    state = _read(start)
    entry = state['series']
    archive = _read(HISTORY)
    historical = archive['sections']['fixed']['points'] if entry['arena'] == archive['arena'] else []
    arena = TT8ServiceArena(**entry['arena'])
    wiring = TT6Wiring(arena, entry['arena']['seed'])
    schedule = _seeds(seed, 8793, 0, 16)['tests']
    held = list(zip(schedule, match_phases(PHASES, len(schedule), [seed, 8793]), strict=True))
    cases = dict(archive=[(r['seed'], r['phase'], r['server']) for r in historical],
                 fresh=[(s, p, 1) for s, p in held])
    result = dict(arena=entry['arena'], installation_seed=entry['arena']['seed'], source=_hash(start),
                  held_seed=seed, study={})
    try:
        cache_hall_stereo(arena, reflections=True)
        for section, starts in cases.items():
            if not starts:
                continue
            run_seed = archive['seed'] if section == 'archive' else seed
            for mode in modes:
                _, prior, response, timing = load(SOURCE)
                fit = load(start)[0].models[entry['model']]
                factory = _restore_tracker(arena.cameras, entry['optics'])
                player = _player('fixed', fit, factory, prior, response, timing, _read(SOURCE))
                if mode == 'feasible':
                    old = _read(SOURCE)
                    player = FeasiblePlayer(fit,prior,factory,response,timing,predictive=True,
                                            flight_memory=FlightMemory('fixed'),drive=ServiceDrive(**old['service_drive']),
                                            learn=False,registration=RegistrationMemory(**old['service_registration']),
                                            learn_registration=True)
                elif mode == 'drive':
                    player.intervention = player.mode = 'fixed'
                    player.learn_registration = False
                elif mode == 'classic':
                    fixed = ServicePlayer(fit, deepcopy(player.prior),
                                          _restore_tracker(arena.cameras, entry['optics']), deepcopy(player.response),
                                          deepcopy(player.timing_memory), predictive=True, flight_memory=deepcopy(player.memory),
                                          service_memory=ServiceMemory(), fixed=3)
                    fixed.prior = player.prior
                    fixed.role_rows = player.role_rows
                    player = fixed
                rows = []
                for i, (s, phase, server) in enumerate(starts):
                    row = _point_row(player, arena, wiring, run_seed, s, phase, server)
                    if baseline and section == 'archive' and mode == 'registered':
                        if (row['evaluator'] != historical[i]['evaluator'] or
                                not np.allclose(row['actions'],historical[i]['actions'],atol=1e-10,rtol=0.)):
                            raise ValueError(f'historical baseline differs at point {i}')
                    rows.append(row)
                check_tapes(json.loads(json.dumps(dict(result, seed=run_seed, points=rows))))
                own = [r['evaluator'] for r in rows if r['server'] == 1]
                count = {k:sum(r[k] for r in own) for k in ('service_contact','legal_serve','points_for')}
                result['study'].setdefault(section,{})[mode] = dict(points=rows, own=len(own), **count)
                print(section, mode, len(own), count, flush=True)
                _write(output,result)
    finally:
        arena.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('start',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=32978121)
    p.add_argument('--modes',nargs='+',choices=['registered','drive','classic','feasible'],default=['registered','drive','classic'])
    p.add_argument('--historical-control',action='store_true')
    args = p.parse_args()
    run(args.start,args.output,args.seed,args.modes,args.historical_control)
