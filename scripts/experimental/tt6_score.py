"""Compare played courses, requiring the same installations, serves and disturbances."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _read


def course_rows(receipt):
    if not receipt['visits']:
        raise ValueError('no played installations')
    rows = []
    for visit in receipt['visits']:
        if not visit['points'] or ('match' not in receipt and len(visit['points']) != len(receipt['surfaces'])):
            raise ValueError(f"incomplete points at {visit['install']}")
        station = visit['installation']
        for point in visit['points']:
            identity = (receipt['seed'], visit['body'], station['arena_seed'], station['view_seed'],
                        point['seed'], point['phase'])
            rows.append((identity, point))
    return rows


def metrics(points):
    return dict(points=len(points), openings=sum(p['player_returns'] > 0 for p in points),
                legal_returns=sum(p['player_returns'] for p in points),
                sustained=sum(p['player_returns'] >= 2 for p in points),
                wins=sum(p['points_for'] for p in points), losses=sum(p['points_against'] for p in points),
                capped=sum(p['truncated'] for p in points))


def compare(paths):
    baseline = None
    for path in paths:
        receipt = _read(path)
        rows = course_rows(receipt)
        ids, points = zip(*rows, strict=True)
        if baseline is not None and baseline != ids:
            raise ValueError(f'{path} has different physical starts or disturbances')
        baseline = ids
        probes = sum(len(v['probes']) for v in receipt['visits'])
        print(path.name, f'probes={probes}', *(f'{k}={v}' for k, v in metrics(points).items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', nargs='+', type=Path)
    compare(parser.parse_args().receipts)
