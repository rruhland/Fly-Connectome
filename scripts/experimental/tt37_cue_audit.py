"""Verify paired physical outcomes and no use of a prep command after visual handoff."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena
from .tt37_opponent_benchmark import BASE, OUT


def first_delivery(row):
    delivered = row['placements']['delivered']
    if not delivered:
        return None
    return {key: delivered[0][key] for key in ('target', 'far_contact', 'near_bounce')}


def matching_delivery(left, right):
    if left is None or right is None:
        return left is right
    if left['target'] != right['target'] or left['far_contact'] != right['far_contact']:
        return False
    a, b = left['near_bounce'], right['near_bounce']
    if set(a) != set(b):
        return False
    # The native coupled solver can shift the billiard rounding on the near
    # table bounce when the arm has a different pose. Compare physical units.
    return all(np.allclose(a[k], b[k], rtol=0., atol=1e-8)
               if isinstance(a[k], (list, float)) else a[k] == b[k] for k in a)


def check(receipt):
    if not receipt.get('complete') or receipt['protocol'] != 'random':
        raise ValueError('Incomplete or incompatible deployment')
    expected_launches = 0
    for row in receipt['rows']:
        if set(row) != set(receipt['modes']):
            raise ValueError('Unpaired arm')
        deliveries = [first_delivery(a) for a in row.values()]
        if not all(matching_delivery(deliveries[0], x) for x in deliveries[1:]):
            raise ValueError('First delivered ball differs across arms')
        for arm in row.values():
            expected_launches += 1
            checked = audit(arm['events'], opening_side=1)
            assert digest_json(checked) == digest_json(arm['score'])
            assert arm['launches'] == expected_launches
            actions = np.asarray(arm['actions'])
            assert actions.ndim == 2 and np.isfinite(actions).all() and np.max(abs(actions)) <= 1.
            assert np.max(arm['motor_audit']['max_measured_force_nm']) <= 3.35 + 1e-8
            n = sum(g.get('prepared', False) for g in arm['gates'])
            for t in arm['handoffs']:
                assert 0 < t['blind_s'] <= receipt['blind_limit_s'] + (1. / 60) + 1e-8
            if n:
                assert arm['early_ticks'] > 0
            else:
                assert arm['early_ticks'] == 0


def summary(rows):
    hits = [g for r in rows for g in r['gates'] if g.get('prepared')]
    leads = [g['lead_s'] for g in hits if g['lead_s'] is not None]
    points = [r['score']['learner'] for r in rows]
    offered = [r['score']['opponent'] for r in rows]
    return dict(points=len(rows), own=sum(points), first=sum(x > 0 for x in points),
                later=sum(max(0, x - 1) for x in points),
                longest_own=max(points), offered=sum(offered), first_offered=sum(x > 0 for x in offered),
                total_legal=sum(r['score']['returns'] for r in rows),
                longest_alternating=max(r['score']['returns'] for r in rows),
                prepared_gates=len(hits), earlier_than_far_contact=sum(x > 0. for x in leads),
                after_far_contact=sum(x <= 0. for x in leads),
                with_no_far_contact=len(hits)-len(leads),
                prepared_lead_ms=None if not leads else (1000. * np.quantile(leads, [.10, .50, .90])).tolist(),
                early_ticks=sum(r['early_ticks'] for r in rows),
                median_blind_ms=None if not any(r['handoffs'] for r in rows) else float(
                    1000. * np.median([x['blind_s'] for r in rows for x in r['handoffs']])))


def paired(rows, treatment, baseline):
    result = np.array([r[treatment]['score']['learner'] - r[baseline]['score']['learner'] for r in rows])
    return dict(net=int(result.sum()), improved=int((result > 0).sum()),
                unchanged=int((result == 0).sum()), worse=int((result < 0).sum()))


def physically_replay(a, card, row):
    a.launch(row['shot'], card=card)
    a.partner.reset(seed=card['seed'])
    for x in row['actions']:
        a.exchange(x, FAR_COMMAND)
    if digest_json(a.events) != digest_json(row['events']):
        raise ValueError('Saved action tape diverged in fresh physical replay')


def summarize(paths, output, replay):
    results = {}
    for path in paths:
        trial = _read(path)
        check(trial)
        rows = trial['rows']
        modes = trial['modes']
        result = dict(source_sha256=_hash(path), models={k: summary([r[k] for r in rows]) for k in modes},
                      paired={f'{b}_vs_{a}': paired(rows, b, a) for i, a in enumerate(modes) for b in modes[i+1:]},
                      events_independently_scored=len(rows)*len(modes), fresh_physical_replays=0)
        if replay:
            sample = set(np.linspace(0, len(rows)-1, min(len(rows), replay), dtype=int))
            sample.update(i for i, r in enumerate(rows) if {'cue', 'off'} <= set(r)
                          and r['cue']['score']['learner'] != r['off']['score']['learner'])
            with centered(_read(BASE)['ready']):
                a = arena(protocol='random')
                try:
                    for i, (c, arms) in enumerate(zip(trial['cards'], rows, strict=True)):
                        if i not in sample:
                            continue
                        for row in arms.values():
                            physically_replay(a, c, row)
                            result['fresh_physical_replays'] += 1
                finally:
                    a.close()
        results[trial['pool']] = result
    _write(output, results)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', type=Path, nargs='+')
    p.add_argument('--output', type=Path, default=OUT / 'cue-motor-summary.json')
    p.add_argument('--replay', type=int, default=0,
                   help='replay this many evenly spaced pairs plus all cue-versus-off score changes')
    args = p.parse_args()
    summarize(args.inputs, args.output, args.replay)
