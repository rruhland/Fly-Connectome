"""Test search-guided two-port optical calibration on fresh motor wiring."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _response_bank, _samples
from .tt6_body_probe import collect_probe
from .tt6_render import cache_hall_stereo
from .tt13_live_audit import assignment
from .tt14_signal_search import SignalArena
from .tt15_context_trial import play, schedule
from .tt15_discovery import DiscoveryPlayer
from .tt15_motion_transfer import Recorder
from .tt16_coded_trial import fit as coded_fit, fit_path as coded_path
from .tt16_sparse_model import QUERY_COUNT, QUERY_SCALE, SEARCH_SCALE, best_keys, motion_ranking, next_port, port_scores
from .tt16_transfer_bridge import Playback, bridge_path, costs
from .tt16_transfer_game import play_case
from .tt16_transfer_protocol import FIXTURES, load_optical_player, make_arena, prepare, setup_path, specification


# Wiring, camera, search and game seeds. None are shared with transfer qualification.
HOLDOUT = {
    'gantry-2': (392634881, 856797343, 523016739, 670518293),
    'five-2': (190813711, 31920659, 459673289, 780221501),
}


def search_data(spec, reference):
    arena = SignalArena(**dict(spec['arena'], render_mode='projected'))
    try:
        seed = spec['search_seed']
        blocks, metric = schedule(arena, seed)
        last = next(i + 1 for i, block in enumerate(blocks) if block['motor'] == 1)
        blocks = blocks[:last]
        wires = [TT6Wiring(arena, port) for port in (spec['original_wiring'], spec['wiring'])]
        engine = Recorder(DiscoveryPlayer(reference))
        args = arena, wires, seed, blocks, metric, bool(spec['replica'] % 2)
        result = play(*args, player=engine)
        replay = Recorder(Playback(result['actions']))
        if play(*args, player=replay) != result or not np.array_equal(engine.positions, replay.positions):
            raise ValueError('sparse search replay diverged')
        return dict(result, positions=np.asarray(engine.positions).tolist(), schedule=blocks,
                    metric=metric.tolist(), cost=costs(result['actions']), replayed=True)
    finally:
        arena.close()


def query(source, port, wires, seed, tracker_factory):
    commands = np.zeros((12, source.action_size))
    commands[2:5, port] = .70
    history = collect_probe(wires, seed, commands)
    if len(history) != 13:
        raise ValueError('sparse command tape did not complete')
    samples = _samples([history], tracker_factory)
    repeated = _samples([collect_probe(wires, seed, commands)], tracker_factory)
    if not (np.array_equal(samples[1][1], repeated[1][1]) and
            np.array_equal(samples[2], repeated[2], equal_nan=True)):
        raise ValueError('sparse RGB query replay diverged')
    return port_scores(source, port, samples), dict(port=port, seed=seed, actions=commands.tolist(),
           ticks=samples[1][1].tolist(),
           observations=[[float(v) if np.isfinite(v) else None for v in row] for row in samples[2]],
           replayed=True)


def evaluate(folder, fixture, development=False):
    spec_path = setup_path(folder, fixture)
    spec = _read(spec_path)
    arena = make_arena(spec)
    seed = spec['search_seed']
    try:
        cache_hall_stereo(arena, reflections=True)
        _, _, player, memory = load_optical_player(spec, arena)
        source = memory.source
        reference = source.jacobian(np.zeros(source.action_size))[:3]
        search = _read(bridge_path(folder, fixture))['search'] if development else search_data(spec, reference)
        orders, signs, loss = motion_ranking(reference, search['actions'], search['positions'],
                                             _response_bank()[0].mean(axis=0))
        costs_before = loss / SEARCH_SCALE
        scores, records = [], []
        wiring = TT6Wiring(arena, spec['wiring'])
        cost = costs_before
        for index in range(QUERY_COUNT):
            port = next_port(orders, signs, cost, [probe['port'] for probe in records])
            sample_seed = int(np.random.default_rng([seed, 52033, index]).integers(0, 2**31 - 1))
            measured, record = query(source, port, wiring, sample_seed, player.tracker_factory)
            scores.append((port, measured))
            records.append(dict(record, losses=measured.tolist()))
            selected, control, cost = best_keys(orders, signs, loss, scores)
        truth = assignment(source.action_size, spec['original_wiring'], spec['wiring'])
        exact, = np.flatnonzero(np.all(orders == truth[0], axis=1) & np.all(signs == truth[1], axis=1))
        rank = int(np.count_nonzero(loss < loss[exact]) + 1)
        final_rank = int(np.count_nonzero(cost < cost[exact]) + 1)
    finally:
        arena.close()

    output = dict(fixture=fixture, donor=spec['donor_sha256'], setup_sha256=_hash(spec_path), search=search,
                  queries=records, truth=truth, blind=control, search_sparse=selected,
                  true_rank_from_search=rank, true_rank_with_queries=final_rank,
                  implementation=_hash(Path(__file__).with_name('tt16_sparse_model.py')))
    print(fixture, 'ports', [q['port'] for q in records], 'true ranks', rank, final_rank,
          'exact', selected == truth, control == truth, flush=True)
    if not development:
        _write(folder / f'sparse-{fixture}-evidence.json.gz', output)
    return output


def freeze(folder):
    folder.mkdir(exist_ok=True, parents=True)
    manifest = dict(fixtures={fixture: specification(fixture, seeds=seeds) for fixture, seeds in HOLDOUT.items()},
                    ports=QUERY_COUNT, search_error_scale_m=SEARCH_SCALE, query_error_scale_m=QUERY_SCALE,
                    games=['search_sparse', 'sparse_only', 'coded'],
                    model_sha256=_hash(Path(__file__).with_name('tt16_sparse_model.py')))
    path = folder / 'sparse-protocol.json.gz'
    if path.exists():
        if _read(path) != json.loads(json.dumps(manifest)):
            raise ValueError('the frozen sparse qualification changed')
    else:
        _write(path, manifest)


def qualify(folder, fixture, mode):
    freeze(folder)
    case = folder / f'sparse-{fixture}-evidence.json.gz'
    measured = _read(case)
    probe_cost = costs([a for q in measured['queries'] for a in q['actions']])
    if mode in (None, 'search_sparse'):
        play_case(folder, fixture, 'search_sparse', key=measured['search_sparse'], prep_receipt=case,
                  extra_preparation={'search': measured['search']['cost'], 'sparse': probe_cost})
    if mode in (None, 'sparse_only'):
        play_case(folder, fixture, 'sparse_only', key=measured['blind'], prep_receipt=case,
                  extra_preparation={'sparse': probe_cost})
    if mode in (None, 'coded'):
        coded_file = coded_path(folder, fixture)
        coded_fit(folder, fixture)
        coded = _read(coded_file)
        play_case(folder, fixture, 'coded', key=coded['key'], prep_receipt=coded_file,
                  extra_preparation={'calibration': costs(coded['actions'])})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('fixture', choices=(*FIXTURES, *HOLDOUT))
    p.add_argument('command', choices=('develop', 'collect', 'qualify'))
    p.add_argument('--mode', choices=('search_sparse', 'sparse_only', 'coded'))
    a = p.parse_args()
    if a.command == 'develop':
        evaluate(a.folder, a.fixture, development=True)
    elif a.command == 'collect':
        freeze(a.folder)
        prepare(a.folder, a.fixture, seeds=HOLDOUT[a.fixture])
        evaluate(a.folder, a.fixture)
    else:
        qualify(a.folder, a.fixture, a.mode)
