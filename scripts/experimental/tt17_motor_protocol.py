"""Matched optical preparation and private hardware fixtures for TT17."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .tt4_camera import cameras
from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _samples
from .tt6_body_probe import collect_probe, probe_commands
from .tt6_body_vision import body_tracker_factory
from .tt6_continuous_trial import match_phases
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt10_series_trial import PHASES, _capture_tracker
from .tt14_gain_hardware import GainWiring, perturbation
from .tt15_motor_probe import measure
from .tt16_continuous_probe import continued_tape
from .tt16_transfer_protocol import load_optical_player, make_arena


# Camera, continuous perturbation and full-game seed. The final two use TT14 mixers.
CASES = {
    'gantry-gain': (65143227, 913050467, 507062311),
    'gantry-coupling': (102665183, 740652019, 383641949),
    'gantry-stable': (61867001, 1, 47620141),
    'five-gain': (78609391, 406291067, 206784133),
    'five-coupling': (173657357, 593711701, 604438757),
    'five-stable': (432198109, 1, 79167721),
    'gantry-coupling-tt14': (31971947, 58214071, 415268863),
    'five-gain-tt14': (591188357, 94140583, 9267151),
}


def setup_path(folder, case):
    return folder / f'motor-{case}-setup.json.gz'


def fixture(case):
    site, profile, *_ = case.split('-')
    view_seed, mixer_seed, game_seed = CASES[case]
    donor = Path('artifacts/tt13') / f'live-{site}-remap-player.json.gz'
    state = _read(ROOT / donor)
    earlier = state['series']['arena']
    wiring = earlier['seed']
    rng = np.random.default_rng([view_seed, 4972])
    shift = rng.uniform((-.025, -.028, -.022), (.025, .028, .022))
    baseline, fovy = rng.uniform(.272, .299), rng.uniform(64.3, 65.7)
    eyes = cameras(width=earlier['width'], height=earlier['height'], baseline=float(baseline))
    rig = [asdict(replace(eye, eye=tuple(np.array(eye.eye) + shift), target=tuple(np.array(eye.target) + shift),
                          fovy=float(fovy))) for eye in eyes]
    arena = dict(earlier, view_seed=view_seed, baseline=float(baseline))
    matrix = perturbation(4 if site == 'gantry' else 5, mixer_seed, profile)
    return dict(case=case, origin='TT14 matrix' if case.endswith('tt14') else 'fresh', site=site,
                profile=profile, wiring=wiring, original_wiring=wiring, mixer_seed=mixer_seed,
                camera_seed=view_seed, game_seed=game_seed, donor=str(donor),
                donor_sha256=_hash(ROOT / donor), arena=arena, camera=rig, matrix=matrix.tolist(),
                starts=_seeds(game_seed, 6011, 0, 65)['tests'],
                phases=match_phases(PHASES, 65, [game_seed, 6011]),
                followups=_seeds(game_seed, 540917, 0, 65)['tests'])


def wiring_for(arena, spec):
    return GainWiring(arena, spec['wiring'], spec['matrix'])


def observed(history, factory):
    _, (_, ticks), poses = _samples([history], factory)
    return dict(ticks=ticks.tolist(),
                seen=[[float(value) if np.isfinite(value) else None for value in row] for row in poses])


def prepare(folder, case):
    spec = fixture(case)
    arena = make_arena(spec)
    try:
        cache_hall_stereo(arena, reflections=True)
        donor = ROOT / spec['donor']
        archive = _read(donor)
        source = load(donor)[0].models[archive['port_memory']['indices'][0]]
        commands = continued_tape(source)
        schedule = _seeds(spec['camera_seed'], 3819, 4, 0)
        known = TT6Wiring(arena, spec['wiring'])
        tapes = [np.pad(commands, ((0, 10), (0, 0))),
                 probe_commands(schedule['player_seed'] + 1, 34, source.action_size)]
        histories = [collect_probe(known, seed, tape) for seed, tape in zip(schedule['calibration'][:2], tapes)]
        if any(len(history) != 35 for history in histories):
            raise ValueError('a common 34-frame optical movement ended early')
        factory = body_tracker_factory(arena.cameras, histories)
        spec['optics'] = _capture_tracker(factory)
        spec['optics_probes'] = [dict(seed=seed, actions=tape.tolist())
                                 for seed, tape in zip(schedule['calibration'][:2], tapes)]
        spec['optics_frames'] = 68
        spec['control_dt'] = arena.control_dt
        spec['reference'] = observed(histories[0][:25], factory)
        spec['actions'] = commands.tolist()
        spec['active_seed'] = schedule['calibration'][2]
        changed = wiring_for(arena, spec)
        history = collect_probe(changed, spec['active_seed'], commands)
        if len(history) != 25:
            raise ValueError('a new-wiring optical movement ended early')
        spec['active'] = observed(history, factory)
        replay = collect_probe(changed, spec['active_seed'], commands)
        if spec['active'] != observed(replay, factory):
            raise ValueError('physical replay changed the active optical history')
        spec['tt15_seed'] = schedule['calibration'][3]
        spec['tt15_reference'] = measure(known, spec['tt15_seed'], factory)
    finally:
        arena.close()
    folder.mkdir(exist_ok=True, parents=True)
    _write(setup_path(folder, case), spec)
    print(case, 'common and active pairs', len(spec['reference']['ticks']), len(spec['active']['ticks']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('case', choices=CASES)
    a = p.parse_args()
    prepare(a.folder, a.case)
