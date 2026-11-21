"""Check pitch probe decisions and compare full games across the same fixtures."""

import argparse
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt8_service_trial import check_tapes
from .tt15_pitch_probe import old_receipt, one_port_fit, pitch_path, probe_samples, select


def active_key(path):
    memory = _read(path)['port_memory']
    return memory['keys'][memory['active']]


def signatures(receipt):
    rows = []
    for row in receipt['points']:
        digest = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
        rows.append((row['seed'], row['phase'], row['server'], digest))
    return rows


def audit(folder, replay):
    results = []
    replayed_points = 0
    for site in ('gantry', 'five'):
        donor = ROOT / 'artifacts/tt13' / f'live-{site}-fixed-player.json.gz'
        state = _read(donor)
        source = load(donor)[0].models[state['port_memory']['indices'][0]]
        for index in (0, 1):
            prefix = folder / f'back-{site}-{index}-motion'
            old_fit_path = old_receipt(folder, site, index)
            fit_path = pitch_path(folder, site, index, 'fit')
            control_path = pitch_path(folder, site, index, 'control-fit')
            fit, control = _read(fit_path), _read(control_path)
            evidence = fit['pitch_probe']
            redacted = {key: evidence[key] for key in ('actions', 'ticks', 'seen')}
            old_key = _read(old_fit_path)['maps']['motion']['key']
            new_key, measured = select(source, old_key, evidence['port'], probe_samples(source.action_size, redacted))
            control_fit = one_port_fit(source, redacted)
            if (any(evidence[key] != value for key, value in measured.items()) or control_fit != control['control']
                    or evidence['prior_fit_sha256'] != _hash(old_fit_path)
                    or control['evidence_sha256'] != _hash(fit_path)):
                raise ValueError('optical decision cannot be reproduced from the recorded evidence')

            # Evaluator wiring is consulted after both decisions, for audit only.
            arena = SimpleNamespace(action_size=source.action_size)
            native = TT6Wiring(arena, state['series']['arena']['seed']).state()
            current = TT6Wiring(arena, fit['ports']).state()
            order = [native['order'].index(axis) for axis in current['order']]
            signs = [native['signs'][axis] * sign for axis, sign in zip(order, current['signs'], strict=True)]
            truth = [order, signs]
            active = evidence['port']
            keys = {'search': old_key, 'search_pitch': new_key, 'probe_only': control_fit['key']}
            validated = {name: dict(complete=key == truth,
                                    positions=sum(a == b for a, b in zip(key[0], truth[0], strict=True)),
                                    signed_ports=sum(a == b and s == t for a, b, s, t in
                                                     zip(key[0], truth[0], key[1], truth[1], strict=True)),
                                    probed_port_correct=key[0][active] == truth[0][active]
                                    and key[1][active] == truth[1][active])
                         for name, key in keys.items()}
            games = {'search': (prefix.with_suffix('.json.gz'), prefix.with_name(prefix.name + '-player.json.gz')),
                     'search_pitch': (pitch_path(folder, site, index, 'game'), pitch_path(folder, site, index, 'player')),
                     'probe_only': (pitch_path(folder, site, index, 'control-game'),
                                    pitch_path(folder, site, index, 'control-player'))}
            plays = {name: _read(game_path) for name, (game_path, _) in games.items()}
            baseline = plays['search']
            tapes = signatures(baseline)
            metrics = {}
            for name, (game_path, player_path) in games.items():
                trial = plays[name]
                fit_source = {'search': old_fit_path, 'search_pitch': fit_path, 'probe_only': control_path}[name]
                if (trial['start'] != _hash(donor) or trial['seed'] != baseline['seed']
                        or trial['installation_seed'] != fit['ports'] or not trial['finished']
                        or trial['checkpoint'] != _hash(player_path) or active_key(player_path) != keys[name]):
                    raise ValueError('a fixture changed its initial state, environment or frozen map')
                reuse_path = pitch_path(folder, site, index, 'control-reuse')
                if name == 'probe_only' and reuse_path.exists():
                    reuse = _read(reuse_path)
                    if reuse['game_sha256'] != _hash(game_path) or reuse['checkpoint_sha256'] != _hash(player_path):
                        raise ValueError('reused tape differs from the archived output')
                elif trial['maps'] != _hash(fit_source):
                    raise ValueError('a game used a different model receipt')
                if replay:
                    check_tapes(trial)
                    replayed_points += len(trial['points'])
                matched = list(zip(tapes, signatures(trial)))
                if not all(a[:2] == b[:2] for a, b in matched):
                    raise ValueError('comparison used different point seeds or phases')
                metrics[name] = dict(score=trial['score'], counts=trial['metrics'], tape_count=len(trial['points']),
                                     same_role=sum(a[2] == b[2] for a, b in matched), shared_starts=len(matched),
                                     exact_actions=sum(a == b for a, b in matched), game=game_path.name)
            results.append(dict(site=site, index=index, control_ties=control_fit['tied'],
                                weak_port=active, costs={key: evidence[key] for key in
                                                       ('dt', 'used_commands', 'stereo_pairs', 'pitch_frames')},
                                errors={label: {k: v for k, v in row.items() if k != 'pose'}
                                        for label, row in evidence['candidates'].items()},
                                truth=truth, validated=validated, qualification=metrics))
    _write(folder / 'pitch-comparison.json.gz', dict(fixtures=results, replayed_points=replayed_points))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    audit(args.folder, args.replay)
