"""Compare signed motor-map recall and fresh calibration over disjoint RGB probes."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _read, _write, _hash, _seeds
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import fit_camera_body, _samples
from .tt6_body_probe import probe_commands
from .tt6_body_vision import body_tracker_factory
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_probe_cache import ProbeCache
from .tt8_service_arena import TT8ServiceArena
from .tt10_series_trial import _capture_tracker, _save
from .tt10_window_trial import _player
from .tt12_motor_recall import align, loss


def run(start, donor, report, checkpoint, cache):
    state = _read(start)
    info = state['series']
    schedule = info['calibration']
    arena = TT8ServiceArena(**info['arena'])
    previous = load(donor)[0].models
    local = load(start)[0].models[info['model']]
    try:
        cache_hall_stereo(arena, reflections=True)
        hardware = DelayedWiring(TT6Wiring(arena, info['arena']['seed']))
        reader = ProbeCache(cache, hardware, arena.cameras, schedule, 34)
        histories, probes = [], []
        for i, seed in enumerate(schedule['calibration']):
            commands = probe_commands(schedule['player_seed']+i,34,hardware.action_size)
            h = reader.collect(hardware,seed,commands)
            histories.append(h)
            probes.append(dict(seed=seed, actions=[a.tolist() for a, _ in h[1:]]))
        factory = body_tracker_factory(arena.cameras,histories[:2])
        support = _samples(histories[:2],factory)
        hold = _samples(histories[2:],factory)
        fits, rows = [], []
        for idx, donor_model in enumerate(previous):
            if donor_model.action_size != hardware.action_size:
                continue
            fitted, stats = align(donor_model,support)
            rows.append(dict(source_model=idx, **stats, original_error_m=loss(donor_model,support)))
            fits.append(fitted)
        best = min(range(len(fits)),key=lambda i:rows[i]['error_m'])
        fresh = fit_camera_body(histories[:2],factory)
        models = dict(recalled=fits[best], fresh_two=fresh, commissioned=local,
                      unchanged=previous[rows[best]['source_model']])
        errors = {name:loss(model,hold) for name,model in models.items()}
        audit_schedule = _seeds(4217831, 1705, 3, 0)
        audit = [reader.collect(hardware, seed, probe_commands(audit_schedule['player_seed']+i,34,hardware.action_size))
                 for i,seed in enumerate(audit_schedule['calibration'])]
        samples = _samples(audit,factory)
        audit_errors = {name:loss(model,samples) for name,model in models.items()}
        # Budget both short-calibration players equally and build them directly
        # from the donor, before observing any points on the new installation.
        entry = {k:info[k] for k in ('arena','donor_sha256')}
        entry.update(optics=_capture_tracker(factory), probes=probes[:2],
                     calibration=dict(schedule,calibration=schedule['calibration'][:2]))
        destinations = {'recalled':checkpoint, 'fresh_two':checkpoint.with_name('fresh-'+checkpoint.name)}
        for name,path in destinations.items():
            library, prior, response, timing = load(donor)
            model = models[name]
            entry.update(model=len(library.models),attempts=[dict(probes=2,method=name)])
            library.models.append(model)
            old = _read(donor)
            player = _player('fixed',model,factory,prior,response,timing,old)
            _save(path,old,library,player,entry,0)
        _write(report,dict(source=_hash(donor), original_start=_hash(start),
                           starts={name:_hash(path) for name,path in destinations.items()},
                           candidates=rows, selected=best, errors_m=errors, audit_errors_m=audit_errors,
                           audit=audit_schedule, held_episodes=[p['seed'] for p in probes[2:]], seen=len(support[2]),
                           observed_held=len(hold[2]), implementation=_hash(Path(__file__).with_name('tt12_motor_recall.py'))))
        print(rows[best],errors,flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('start','donor','report','checkpoint'):
        p.add_argument(name,type=Path)
    p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args()
    run(a.start,a.donor,a.report,a.checkpoint,a.cache)
