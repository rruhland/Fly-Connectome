"""Ablate the learned gantry world in complete, zero-target-practice SO-101 games."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt7_flight import FlightMemory
from .tt19_factory_serve_trial import held
from .tt19_flight import FlightLaw


def run(folder, catalog_path, seeds, resume=False):
    proposals_path = folder / 'factory-serve-proposals.json.gz'
    proposals = _read(proposals_path)
    # Select the first factory proposal without consulting any target practice.
    start = dict(factory_pick=0, seeds=[])
    with np.load(catalog_path) as archive:
        catalog = dict(archive)
    report = dict(catalog_sha256=_hash(catalog_path), proposals_sha256=_hash(proposals_path),
                  seeds=seeds, target_practice_points=0, prior='nominal', games=[])
    response = FlightMemory('impulse', (), (), frozen=True).response_matrix
    output = folder / 'factory-serve-nominal.json.gz'
    if resume and output.exists():
        earlier = _read(output)
        completed = [g['seed'] for g in earlier['games']]
        if completed != seeds[:len(completed)] or any(earlier[k] != report[k] for k in
                    ('catalog_sha256', 'proposals_sha256', 'prior', 'target_practice_points')):
            raise ValueError('Saved nominal games differ from requested initial sequence or player')
        report['games'] = earlier['games']
    for seed in seeds[len(report['games']):]:
        game = held(folder, proposals, start, catalog, FlightLaw(), response, seed, 'factory')
        report['games'].append(game)
        _write(output, report)
        print(seed, game['score'], 'returns', game['returns'], 'serves', game['serves'], flush=True)


def summarize(folder):
    sources = ['factory-serve-held', 'factory-serve-1529880313-held',
               'factory-serve-82617492-distinct-held', 'factory-serve-335280197-distinct-held']
    training = _read(folder / 'factory-serve-distinct-practice.json.gz')
    earlier = _read(folder / 'factory-serve-practice.json.gz')
    excluded = {p['seed'] for record in (training, earlier) for p in record['trials']}
    if earlier['corrected_pick'] != training['corrected_pick']:
        raise ValueError('The repeated-toss and distinct-toss selectors chose different strokes')
    results = {}
    for label, files, mode in [('gantry_factory', sources, 'factory'), ('gantry_corrected', sources, 'corrected'),
                               ('nominal_factory', ['factory-serve-nominal'], 'factory')]:
        rows = []
        for name in files:
            source = _read(folder / (name + '.json.gz'))
            for g in (game for game in source['games'] if game['mode'] == mode):
                points = g['points']
                if len(points) != g['replayed_points'] or not g['finished'] or any(p['seed'] in excluded for p in points):
                    raise ValueError('An incomplete, overlapping or unverified game entered the comparison')
                rows.append(dict(seed=g['seed'], score=g['score'], win=int(g['score'][0] > g['score'][1]),
                                 returns=g['returns'], serves=g['serves'], own_serves=g['serve_attempts'],
                                 multiple=g['multiple'], scored_attempts=len(points), sha256=_hash(folder / (name + '.json.gz'))))
        if len({row['seed'] for row in rows}) != 4:
            raise ValueError('Missing or duplicate game seeds in the final comparison')
        results[label] = dict(games=rows, totals={key:sum(row[key] for row in rows) for key in
                                                ('win','returns','serves','own_serves','multiple','scored_attempts')})
    _write(folder / 'factory-serve-summary.json.gz', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[930477119, 1529880313, 82617492, 335280197])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary:
        summarize(args.folder)
    else:
        run(args.folder, args.catalog, args.seeds, args.resume)
