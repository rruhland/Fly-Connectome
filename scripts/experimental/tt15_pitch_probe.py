"""Resolve the weakest position channel with one RGB pitch probe before table play."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _samples
from .tt6_body_probe import collect_probe
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_trial import check_tapes
from .tt12_motor_recall import predicted, rewire
from .tt13_live_trial import load_player
from .tt15_back_transfer import game


def old_receipt(folder, site, index):
    return folder / f'back-{site}-{index}-motion-fit.json.gz'


def pitch_path(folder, site, index, name):
    return folder / f'pitch-{site}-{index}-{name}.json.gz'


def motion_test(source, order):
    position_gain = np.linalg.norm(source.jacobian(np.zeros(source.action_size))[:3], axis=0)
    source_axis = int(np.argmin(position_gain))
    port = order.index(source_axis)
    commands = np.zeros((12, source.action_size))
    commands[2:5, port] = .70
    return source_axis, port, position_gain, commands


def optical_error(source, key, samples):
    commands, indices, observed = samples
    observed = observed.reshape(-1, 4)
    pitch_frames = np.isfinite(observed[:, 3])
    estimate = predicted(rewire(source, *key), commands, indices)
    # A constant registration offset can differ between positions at rest.
    # Remove it so the decision uses optical motion relative to that offset.
    errors = estimate - observed
    xyz = errors[:, :3]
    pitch = errors[pitch_frames, 3]
    return dict(xyz_error_m=float(np.mean(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)))
                if len(xyz) else None,
                pitch_error_rad=float(np.mean(np.abs(pitch - np.median(pitch)))) if len(pitch) else None,
                pose=estimate.tolist())


def select(source, old_key, port, samples):
    """Use the same observed pitch sequence for both sign hypotheses."""
    order, old_signs = old_key
    alternate = old_signs.copy()
    alternate[port] *= -1
    keys = {'old': [order, old_signs], 'opposite': [order, alternate]}
    observed = samples[2].reshape(-1, 4)
    pitch_frames = np.isfinite(observed[:, 3])
    fitted = {name: optical_error(source, key, samples) for name, key in keys.items()}
    observed_pitch = observed[pitch_frames, 3]
    sufficient = len(observed_pitch) >= 5 and bool(np.ptp(observed_pitch) >= .06)
    selected = 'old'
    if sufficient and fitted['old']['pitch_error_rad'] - fitted['opposite']['pitch_error_rad'] > .025:
        selected = 'opposite'
    return keys[selected], dict(port=port, selected=selected, enough_optical_motion=sufficient,
                                candidates=fitted, pose_frames=int(len(observed)), pitch_frames=int(pitch_frames.sum()),
                                ticks=samples[1][1].tolist(),
                                seen=[[float(x) if np.isfinite(x) else None for x in row] for row in observed])


def probe(folder, site, index):
    receipt = old_receipt(folder, site, index)
    old = _read(receipt)
    donor = ROOT / 'artifacts/tt13' / f'live-{site}-fixed-player.json.gz'
    settings = _read(donor)['series']['arena']
    arena = TT8ServiceArena(**settings)
    try:
        cache_hall_stereo(arena, reflections=True)
        _, _, player, memory = load_player(donor, arena, 'fixed')
        old_key = old['maps']['motion']['key']
        axis, port, gain, commands = motion_test(memory.source, old_key[0])
        wiring = TT6Wiring(arena, old['ports'])
        seed = int(np.random.default_rng([old['seed'], 27318]).integers(0, 2**31-1))
        history = collect_probe(wiring, seed, commands)
        samples = _samples([history], player.tracker_factory)
        selected, evidence = select(memory.source, old_key, port, samples)
        control_dt = arena.control_dt
    finally:
        arena.close()
    fit = dict(old, maps={'motion': dict(old['maps']['motion'], key=selected)})
    fit['pitch_probe'] = dict(evidence, seed=seed, used_commands=len(history)-1,
                              stereo_pairs=len(history), dt=control_dt,
                              actions=[action.tolist() for action, _ in history[1:]],
                              source_axis=axis, source_gain_m_per_unit=gain.tolist(), old_key=old_key,
                              source_sha256=_hash(donor), prior_fit_sha256=_hash(receipt))
    _write(pitch_path(folder, site, index, 'fit'), fit)
    print(site, index, axis, port, evidence['selected'], evidence['pitch_frames'],
          {k: v['pitch_error_rad'] for k, v in evidence['candidates'].items()}, flush=True)


def qualify(folder, site, index, control=False):
    """Use the exact existing game loop and store the result under a new prefix."""
    tag = 'control-' if control else ''
    fit = pitch_path(folder, site, index, f'{tag}fit')
    with TemporaryDirectory(dir=folder) as isolated:
        workspace = Path(isolated)
        old_receipt(workspace, site, index).write_bytes(fit.read_bytes())
        game(workspace, site, index, 'motion')
        prefix = workspace / f'back-{site}-{index}-motion'
        prefix.with_suffix('.json.gz').replace(pitch_path(folder, site, index, f'{tag}game'))
        prefix.with_name(prefix.name + '-player.json.gz').replace(pitch_path(folder, site, index, f'{tag}player'))


def probe_samples(size, evidence):
    commands = np.vstack((np.zeros(size), np.asarray(evidence['actions'])))
    ticks = np.asarray(evidence['ticks'], dtype=int)
    observed = np.asarray(evidence['seen'], dtype=float)
    return commands[None], (np.zeros_like(ticks), ticks), observed


def one_port_fit(source, evidence):
    samples = probe_samples(source.action_size, evidence)
    commands = samples[0][0]
    port, = np.flatnonzero(np.any(commands != 0., axis=0)).tolist()
    identity = [list(range(source.action_size)), [1] * source.action_size]
    candidates = []
    for axis in range(source.action_size):
        for sign in (1, -1):
            order, signs = [part.copy() for part in identity]
            order[port], order[axis] = order[axis], order[port]
            signs[port] = sign
            loss = optical_error(source, [order, signs], samples)
            position, pitch = loss['xyz_error_m'], loss['pitch_error_rad']
            cost = float(np.hypot(position, .12 * pitch)) if pitch is not None else position
            candidates.append(dict(axis=axis, sign=sign, key=[order, signs], loss_m=cost,
                                   position_error_m=position, pitch_error_rad=pitch))
    observed_candidates = [candidate for candidate in candidates if candidate['loss_m'] is not None]
    best = min((c['loss_m'] for c in observed_candidates), default=None)
    tied = [] if best is None else [c for c in observed_candidates if c['loss_m'] <= best + .001]
    choice = min(tied, key=lambda c: (c['axis'] != port, c['sign'] < 0, c['axis'])) if tied else None
    return dict(port=port, candidates=candidates, tied=len(tied), choice=choice,
                key=identity if choice is None else choice['key'])


def control(folder, site, index):
    evidence_path = pitch_path(folder, site, index, 'fit')
    recorded = _read(evidence_path)['pitch_probe']
    evidence = {key: recorded[key] for key in ('actions', 'ticks', 'seen')}
    donor = ROOT / 'artifacts/tt13' / f'live-{site}-fixed-player.json.gz'
    state = _read(donor)
    memory = state['port_memory']
    source = load(donor)[0].models[memory['indices'][0]]
    fitted = one_port_fit(source, evidence)
    original = _read(old_receipt(folder, site, index))
    # Only the installation and fixture identifiers route the evaluator.
    # The controller's key was selected solely from the shared probe.
    receipt = dict(seed=original['seed'], ports=original['ports'], maps={'motion': {'key': fitted['key']}},
                   control=fitted, evidence_sha256=_hash(evidence_path))
    _write(pitch_path(folder, site, index, 'control-fit'), receipt)
    print(site, index, 'one-port control', fitted['choice'], flush=True)
    prefixes = [folder / f'back-{site}-{index}-{mode}' for mode in ('fixed', 'carried', 'cold', 'motion')]
    games = [(prefix.with_suffix('.json.gz'), prefix.with_name(prefix.name + '-player.json.gz')) for prefix in prefixes]
    games.append((pitch_path(folder, site, index, 'game'), pitch_path(folder, site, index, 'player')))
    for archived_game, archived_player in games:
        if not archived_game.exists() or not archived_player.exists():
            continue
        saved = _read(archived_player)['port_memory']
        if saved['keys'][saved['active']] == fitted['key']:
            check_tapes(_read(archived_game))
            pitch_path(folder, site, index, 'control-game').write_bytes(archived_game.read_bytes())
            pitch_path(folder, site, index, 'control-player').write_bytes(archived_player.read_bytes())
            _write(pitch_path(folder, site, index, 'control-reuse'), dict(game=archived_game.name,
                      game_sha256=_hash(archived_game), checkpoint_sha256=_hash(archived_player)))
            print(site, index, 'verified existing tape', archived_game.name, flush=True)
            return
    qualify(folder, site, index, control=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('site', choices=('five', 'gantry'))
    parser.add_argument('index', type=int, choices=(0, 1))
    parser.add_argument('command', choices=('probe', 'qualify', 'control'))
    cli = parser.parse_args()
    {'probe': probe, 'qualify': qualify, 'control': control}[cli.command](cli.folder, cli.site, cli.index)
