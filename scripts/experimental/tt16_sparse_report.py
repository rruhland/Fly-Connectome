"""Reconstruct sparse port decisions and summarize held-out games."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_control import _response_bank
from .tt6_memory import load
from .tt16_sparse_model import SEARCH_SCALE, best_keys, motion_ranking, next_port, port_scores
from .tt16_sparse_trial import HOLDOUT, freeze
from .tt16_transfer_game import case_prefix, game_clock
from .tt16_transfer_protocol import setup_path
from .tt16_transfer_report import aggregate, key_accuracy


def independently_fit(spec, evidence):
    path = ROOT / spec['donor']
    state = _read(path)
    source = load(path)[0].models[state['port_memory']['indices'][0]]
    spatial = source.jacobian(np.zeros(source.action_size))[:3]
    search = evidence['search']
    orders, signs, losses = motion_ranking(spatial, search['actions'], search['positions'],
                                           _response_bank()[0].mean(axis=0))
    current = losses / SEARCH_SCALE
    observed = []
    for q in evidence['queries']:
        port = next_port(orders, signs, current, [prior for prior, _ in observed])
        if port != q['port']:
            raise ValueError('optical evidence led to a different query')
        actions = np.vstack([np.zeros(source.action_size), q['actions']])
        ticks = np.asarray(q['ticks'], dtype=int)
        samples = actions[None], (np.zeros_like(ticks), ticks), np.asarray(q['observations'], dtype=float)
        cost = port_scores(source, port, samples)
        if not np.array_equal(cost, q['losses']):
            raise ValueError('RGB query loss could not be reproduced')
        observed.append((port, cost))
        assisted, control, current = best_keys(orders, signs, losses, observed)
    if (assisted, control) != (evidence['search_sparse'], evidence['blind']):
        raise ValueError('signed axis assignment differs on replay')


def report(folder):
    freeze(folder)
    results, per_mode = {}, {}
    for fixture in HOLDOUT:
        setup = setup_path(folder, fixture)
        spec = _read(setup)
        raw = _read(folder / f'sparse-{fixture}-evidence.json.gz')
        if raw['setup_sha256'] != _hash(setup) or raw['donor'] != spec['donor_sha256']:
            raise ValueError('different optics or archived source')
        independently_fit(spec, raw)
        games = {}
        tapes = {}
        for mode in ('search_sparse', 'sparse_only', 'coded'):
            prefix = case_prefix(folder, fixture, mode)
            receipt = _read(prefix.with_suffix('.json.gz'))
            checkpoint = prefix.with_name(prefix.name + '-player.json.gz')
            saved = _read(checkpoint)['port_memory']
            if (receipt['checkpoint'] != _hash(checkpoint) or receipt['final_key'] != saved['keys'][saved['active']]
                    or receipt['provenance']['setup'] != _hash(setup)
                    or receipt['replayed_points'] != len(receipt['points'])):
                raise ValueError('scored evidence or portable player failed qualification')
            prep = sum(section['frames'] for section in receipt['preparation'].values())
            if game_clock(receipt['points'], prep, spec['control_dt']) != receipt['clock']:
                raise ValueError('scored elapsed time ledger differs on replay')
            row = {k: receipt[k] for k in ('score', 'preparation', 'clock', 'metrics', 'replayed_points')}
            row.update(key_accuracy(receipt['final_key'], raw['truth']),
                       game=prefix.name+'.json.gz', checkpoint=checkpoint.name)
            tapes[mode] = [p['action_sha256'] for p in receipt['points']]
            games[mode] = row
            per_mode.setdefault(mode, []).append(row)
        results[fixture] = dict(true_rank_from_search=raw['true_rank_from_search'],
                                true_rank_with_queries=raw['true_rank_with_queries'],
                                queried_ports=[q['port'] for q in raw['queries']],
                                exact_coded_tapes=tapes['search_sparse'] == tapes['coded'], games=games)
    totals = {mode: aggregate(rows) for mode, rows in per_mode.items()}
    _write(folder / 'sparse-summary.json.gz', dict(fixtures=results, totals=totals))
    print(totals)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    report(p.parse_args().folder)
