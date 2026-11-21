"""Compare destination, transferred and factory strokes on the same new own serves."""

import argparse
import numpy as np
from scipy.stats import binomtest

from .tt4_transfer_trial import _hash, _read, _write
from .tt20_camera_carry import EYES, OUTPUT, SETTINGS, Station, summarize, verify

SEED = 208393128


def paired(rows, baseline, alternative, field):
    a, b = [], []
    for group in rows:
        for values, index in ((a, baseline), (b, alternative)):
            row = group[str(index)]
            values.append(row['reward'] > 0 if field == 'wins' else bool(row['outcome']['legal_serve']))
    gains = sum(y and not x for x, y in zip(a,b,strict=True))
    losses = sum(x and not y for x, y in zip(a,b,strict=True))
    return dict(base=sum(a), selected=sum(b), gained=gains, lost=losses,
                exact_p=1. if gains+losses == 0 else binomtest(gains, gains+losses).pvalue)


def run(eye, count):
    checkpoints = {e: OUTPUT/f'camera-carry-{e}-practice.json.gz' for e in EYES}
    source = {e: _read(p) for e,p in checkpoints.items()}
    choices = {e: data['learning'][-1]['selected'] for e,data in source.items()}
    choices['factory'] = source[eye]['factory_pick']
    seeds = np.random.default_rng(SEED).integers(2**31, size=count).tolist()
    if any(r['seed'] in seeds for data in source.values() for r in data['points']):
        raise ValueError('some test tosses occurred during source practice')
    branches = {}
    for pick in sorted(set(choices.values())):
        station = Station(eye)
        try:
            group = [station.point(s,1,pick) for s in seeds]
        finally:
            station.close()
        branches[str(pick)] = group
        print(eye, pick, summarize(group), flush=True)
        verify(dict(arena=SETTINGS,eye=eye,points=group))
    rows = [{key: group[i] for key,group in branches.items()} for i in range(count)]
    alternatives = {label: {metric:paired(rows,choices['factory'],pick,metric) for metric in ('wins','legal')}
                    for label,pick in choices.items() if label != 'factory'}
    _write(OUTPUT/f'camera-carry-{eye}-paired-serves.json.gz',
           dict(schema='tt20-camera-carry-paired-serves-1', eye=eye, arena=SETTINGS,
                training_sha256={e:_hash(p) for e,p in checkpoints.items()}, choices=choices, seed=SEED,
                seeds=seeds, rows=rows, metrics=alternatives, physical_replays=count*len(branches)))
    print(eye, alternatives, flush=True)


def continued(eye, count):
    from .tt20_stroke_carry import SWITCH

    saved = OUTPUT / f'stroke-carry-{eye}-{SWITCH}.json.gz'
    checkpoint = _read(saved)
    initial_eval = OUTPUT / f'camera-carry-{eye}-paired-serves.json.gz'
    previous = _read(initial_eval)
    if len(previous['seeds']) != count:
        raise ValueError('continued learner requires the same held tosses')
    rows = [dict(r) for r in previous['rows']]
    update = checkpoint['learning'][-1]
    control, trained = (update[k]['pick'] for k in ('destination_only','carried'))
    extra = {}
    for candidate in {control,trained}:
        if str(candidate) in rows[0]:
            continue
        station = Station(eye)
        try:
            group = [station.point(s,1,candidate) for s in previous['seeds']]
        finally:
            station.close()
        verify(dict(eye=eye,arena=SETTINGS,points=group))
        extra[str(candidate)] = group
        for row, point in zip(rows,group,strict=True):
            row[str(candidate)] = point
    scores = {field:paired(rows,control,trained,field) for field in ('wins','legal')}
    _write(OUTPUT/f'stroke-carry-{eye}-held.json.gz',
           dict(eye=eye,seed=SEED,training_sha256=_hash(saved),comparison_sha256=_hash(initial_eval),
                destination_only=control,carried=trained,extra=extra,metrics=scores))
    print(eye,'after switch',scores,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eye',choices=EYES)
    parser.add_argument('--starts',type=int,default=16)
    parser.add_argument('--continued',action='store_true')
    a=parser.parse_args()
    if a.continued:
        continued(a.eye,a.starts)
    else:
        run(a.eye,a.starts)
