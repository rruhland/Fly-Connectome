"""Frozen fixture and camera preparation for held-out search-to-table transfer."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .tt4_camera import PerspectiveCamera, cameras
from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_probe import collect_probe, probe_commands
from .tt6_body_vision import body_tracker_factory
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import TT8ServiceArena
from .tt10_series_trial import PHASES, _capture_tracker, _restore_tracker
from .tt13_live_trial import load_player


# Each tuple is the new wiring, new camera, search and full-game seed.
FIXTURES = {
    'gantry-0': (422499613, 193908619, 843265901, 651303241),
    'gantry-1': (769853819, 51704171, 241983107, 690778681),
    'five-0': (834688853, 657917517, 453588113, 352615669),
    'five-1': (113634167, 527080267, 917368519, 762573443),
}


def setup_path(folder, fixture):
    return folder / f'transfer-{fixture}-setup.json.gz'


def specification(fixture, *, seeds=None):
    wiring, camera_seed, search_seed, game_seed = FIXTURES[fixture] if seeds is None else seeds
    site, replica = fixture.split('-')
    donor = Path('artifacts/tt13') / f'live-{site}-fixed-player.json.gz'
    original = _read(ROOT / donor)['series']['arena']
    rng = np.random.default_rng(np.random.SeedSequence([camera_seed, 9417]))
    shift = rng.uniform((-.035, -.035, -.028), (.035, .035, .028))
    baseline = float(rng.uniform(.262, .302))
    fovy = float(rng.uniform(64., 66.))
    eyes = cameras(width=original['width'], height=original['height'], baseline=baseline)
    camera = [asdict(replace(eye, eye=tuple(np.asarray(eye.eye) + shift),
                             target=tuple(np.asarray(eye.target) + shift), fovy=fovy)) for eye in eyes]
    arena = dict(original, seed=wiring, view_seed=camera_seed, baseline=baseline)
    return dict(fixture=fixture, site=site, replica=int(replica), wiring=wiring, camera_seed=camera_seed,
                search_seed=search_seed, game_seed=game_seed, donor=str(donor), donor_sha256=_hash(ROOT / donor),
                original_wiring=original['seed'], arena=arena, camera=camera,
                starts=_seeds(game_seed, 618, 0, 65)['tests'],
                phases=match_phases(PHASES, 65, [game_seed, 618]))


def make_arena(spec):
    arena = TT8ServiceArena(**spec['arena'])
    arena._rig = tuple(PerspectiveCamera(**eye) for eye in spec['camera'])
    return arena


def load_optical_player(spec, arena, mode='fixed'):
    """Load the archived player with the common, newly observed table optics."""
    donor = ROOT / spec['donor']
    if _hash(donor) != spec['donor_sha256']:
        raise ValueError('the archived table player changed')
    state, library, player, memory = load_player(donor, arena, mode)
    player.tracker_factory = _restore_tracker(arena.cameras, spec['optics'])
    state['series']['arena'] = spec['arena']
    state['series']['optics'] = spec['optics']
    state['series']['tt16_camera'] = spec['camera']
    return state, library, player, memory


def prepare(folder, fixture, *, seeds=None):
    spec = specification(fixture, seeds=seeds)
    schedule = _seeds(spec['camera_seed'], 3819, 2, 0)
    arena = make_arena(spec)
    try:
        cache_hall_stereo(arena, reflections=True)
        original_wiring = TT6Wiring(arena, spec['original_wiring'])
        histories, probes = [], []
        for i, seed in enumerate(schedule['calibration']):
            actions = probe_commands(schedule['player_seed'] + i, 34, arena.action_size)
            history = collect_probe(original_wiring, seed, actions)
            histories.append(history)
            probes.append(dict(seed=seed, actions=[action.tolist() for action, _ in history[1:]]))
        spec['optics'] = _capture_tracker(body_tracker_factory(arena.cameras, histories))
        spec['optics_probes'] = probes
        spec['optics_frames'] = sum(len(row['actions']) for row in probes)
        spec['control_dt'] = arena.control_dt
    finally:
        arena.close()
    folder.mkdir(exist_ok=True, parents=True)
    _write(setup_path(folder, fixture), spec)
    print(fixture, 'shared optics frames', spec['optics_frames'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('fixture', choices=FIXTURES)
    args = p.parse_args()
    prepare(args.folder, args.fixture)
