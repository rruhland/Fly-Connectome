"""Freeze fixture metadata and aggregate full-game frame and return accounting."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt16_transfer_bridge import as_json, bridge_path
from .tt16_transfer_game import REQUIRED_MODES, case_prefix, game_clock
from .tt16_transfer_protocol import FIXTURES, setup_path, specification


INHERITED = ('tt15_motion_transfer', 'tt15_pitch_probe', 'tt15_discovery', 'tt15_context_trial',
             'tt13_live_ports', 'tt6_continuous_trial')


def freeze(folder):
    folder.mkdir(exist_ok=True, parents=True)
    protocol = dict(schema='tt16-transfer-protocol-v1', parent_revision='cca1694',
                    fixtures={name: specification(name) for name in FIXTURES}, modes=list(REQUIRED_MODES),
                    inherited_implementation={p: _hash(Path(__file__).with_name(p + '.py')) for p in INHERITED},
                    optical_preparation=dict(probes=2, commands_per_probe=34, wiring='original'),
                    search=dict(stop='end of first changed-wiring block', fit_window=96),
                    queried_port=dict(commands=12, condition='ordinary', selector='TT15 frozen pitch'),
                    game=dict(first_server=-1, max_starts=65, target=11, lead=2),
                    time=dict(first_return='preparation commands plus chronological legal-return event frame',
                              efficiency='points or returns divided by total control frames',
                              empty_return='right-censored at completion'))
    path = folder / 'transfer-protocol.json.gz'
    if path.exists():
        if _read(path) != as_json(protocol):
            raise ValueError('the frozen holdout definition changed')
    else:
        _write(path, protocol)


def key_accuracy(own, truth):
    return dict(matched_axes=sum(a == b for a, b in zip(own[0], truth[0], strict=True)),
                matched_signed_axes=sum(a == b and p == q for a, b, p, q in
                                       zip(own[0], truth[0], own[1], truth[1], strict=True)),
                complete=own == truth)


def aggregate(rows):
    successes = [row['clock']['frames_to_first_return'] for row in rows
                 if row['clock']['frames_to_first_return'] is not None]
    frames = sum(row['clock']['total_frames'] for row in rows)
    points = sum(row['score'][0] for row in rows)
    returns = sum(row['metrics']['total_legal_returns'] for row in rows)
    return dict(games=len(rows), wins=sum(row['score'][0] > row['score'][1] for row in rows),
                points=points, legal_returns=returns, total_frames=frames,
                returned_in_games=len(successes), median_elapsed_frames_for_successes=float(np.median(successes))
                if successes else None, points_per_1000_frames=1000 * points / frames,
                legal_returns_per_1000_frames=1000 * returns / frames)


def run(folder):
    freeze(folder)
    protocols = _read(folder / 'transfer-protocol.json.gz')['fixtures']
    summary = {}
    per_mode = {}
    coded_ready = all(case_prefix(folder, fixture, 'coded').with_suffix('.json.gz').is_file()
                      for fixture in FIXTURES)
    modes = (*REQUIRED_MODES, 'coded') if coded_ready else REQUIRED_MODES
    for fixture in FIXTURES:
        setup = setup_path(folder, fixture)
        spec = _read(setup)
        expected = protocols[fixture]
        if any(spec[k] != value for k, value in expected.items()):
            raise ValueError('prepared fixture differs from protocol')
        bridge = bridge_path(folder, fixture)
        fit = _read(bridge)
        if fit['setup_sha256'] != _hash(setup):
            raise ValueError('search and probe were performed with different setup')
        games = {}
        online = _read(case_prefix(folder, fixture, 'table_online').with_suffix('.json.gz'))
        for mode in modes:
            prefix = case_prefix(folder, fixture, mode)
            receipt = _read(prefix.with_suffix('.json.gz'))
            checkpoint = prefix.with_name(prefix.name + '-player.json.gz')
            player = _read(checkpoint)
            memory = player['port_memory']
            end_key = memory['keys'][memory['active']]
            if (receipt['checkpoint'] != _hash(checkpoint) or receipt['final_key'] != end_key
                    or receipt['start'] != spec['donor_sha256'] or receipt['camera'] != spec['camera']
                    or receipt['arena'] != spec['arena'] or receipt['provenance']['setup'] != _hash(setup)
                    or receipt['replayed_points'] != len(receipt['points'])):
                raise ValueError('game, portable player, or shared setup failed verification')
            prep = sum(cost['frames'] for cost in receipt['preparation'].values())
            if game_clock(receipt['points'], prep, spec['control_dt']) != receipt['clock']:
                raise ValueError('game clock differs from the chronological tape')
            case = {k: receipt[k] for k in ('score', 'metrics', 'clock', 'preparation', 'game_cost', 'replayed_points')}
            case.update(game=prefix.name + '.json.gz', checkpoint=checkpoint.name,
                        initial=key_accuracy(receipt['initial_key'], fit['truth']),
                        final=key_accuracy(end_key, fit['truth']))
            if mode in ('search_pitch', 'probe_only') and receipt['provenance']['bridge'] != _hash(bridge):
                raise ValueError('game used different search or one-port observations')
            if any((row['seed'], row['phase']) != (s, p) for row, s, p in
                   zip(receipt['points'], spec['starts'], spec['phases'])):
                raise ValueError('matched precommitted point and disturbance schedule diverged')
            case['shared_starts_with_online'] = min(len(online['points']), len(receipt['points']))
            case['same_service_role_with_online'] = sum(a['server'] == b['server'] for a, b in
                                                       zip(online['points'], receipt['points']))
            games[mode] = case
            per_mode.setdefault(mode, []).append(case)
        summary[fixture] = dict(camera=spec['camera'], wiring=spec['wiring'], clock_dt=spec['control_dt'],
                                search_frames=fit['search']['represented_frames'], shifted_search_frames=fit['search']['changed_frames'],
                                key_before_pitch=key_accuracy(fit['keys']['search'], fit['truth']),
                                pitch=dict(selected=fit['pitch']['selected'], port=fit['pitch']['port'],
                                           enough_optical_motion=fit['pitch']['enough_optical_motion']),
                                games=games)
    totals = {mode: aggregate(rows) for mode, rows in per_mode.items()}
    _write(folder / 'transfer-summary.json.gz', dict(fixtures=summary, totals=totals))
    return totals


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--freeze', action='store_true')
    args = p.parse_args()
    if args.freeze:
        freeze(args.folder)
    else:
        print(run(args.folder))
