"""Audit independent schedules and tabulate censored time-to-ten for actual online agents."""
import itertools
import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import STREAMS
from .tt22_joint_practice import draw
from .tt23_implicit_online import CONFIRM_SEEDS, LIVE_SEED, OUT, SOURCE
from .tt23_jog_bank import GENERATORS
from .tt23_sequence import summarize


def starts(seed):
    rng = np.random.default_rng(seed)
    strata = [draw(rng, 4, band) for band in STREAMS.values()]
    return [(s, lane) for pair in zip(*strata, strict=True) for s, lane in pair]


def main():
    prior = {s for generator in (*GENERATORS, LIVE_SEED) for s, _ in starts(generator)}
    rows = []
    for i, seed in enumerate(CONFIRM_SEEDS, 1):
        schedule = starts(seed)
        claimed = {s for s, _ in schedule}
        if len(claimed) != 8 or prior & claimed:
            raise ValueError('Duplicate destination starts')
        prior.update(claimed)
        path = OUT / f'implicit-confirm-{i}.json.gz'
        doc = _read(path)
        if doc['source'] != _hash(SOURCE):
            raise ValueError('Unexpected source checkpoint')
        cases = doc['cases']
        if [(c['seed'], c['lane']) for c in cases] != schedule[:len(cases)]:
            raise ValueError('Schedule mismatch')
        summary = {}
        for mode in ('learned', 'flat'):
            included = [n for n,c in enumerate(cases) if mode in c['results']]
            if included != list(range(len(included))):
                raise ValueError('Censored stream has a gap before its endpoint')
            scored = summarize([cases[n]['results'][mode] for n in included])
            if doc['summary'][mode] != scored:
                raise ValueError('Recorded outcomes failed to regenerate')
            event = scored['first_ten']
            if len(included) != (event['point'] if event else 8):
                raise ValueError('Stopped at an unexpected point')
            summary[mode] = dict(point=event['point'] if event else None,
                                 seconds=event['simulated_seconds'] if event else None,
                                 observation_seconds=scored['control_frames']/30,
                                 legal_returns=scored['learner_returns'])
        if doc['verified'] != sum(len(c['results']) for c in cases):
            raise ValueError('Unverified physical alternative')
        rows.append(dict(stream=i, seed=seed, receipt=_hash(path), paired=summary, verified=doc['verified']))
    difference = np.array([(r['paired']['flat']['point'] or 9) - (r['paired']['learned']['point'] or 9) for r in rows])
    nonzero = difference[difference != 0]
    signs = np.array(list(itertools.product((-1,1), repeat=len(nonzero))), dtype=int)
    probability = float(np.mean((signs @ nonzero) >= difference.sum()))
    report = dict(rows=rows, mean_launches_saved=float(difference.mean()), exact_one_sided_p=probability,
                  verified=sum(r['verified'] for r in rows))
    for mode in ('learned','flat'):
        sample = [r['paired'][mode] for r in rows]
        report[mode] = dict(solved=sum(r['point'] is not None for r in sample),
                            solved_by_40=sum(r['seconds'] is not None and r['seconds']<=40 for r in sample),
                            censored_before_40=sum(r['seconds'] is None and r['observation_seconds']<40 for r in sample),
                            capped_mean=float(np.mean([r['point'] or 9 for r in sample])))
    _write(OUT/'implicit-confirm-summary.json.gz', report)
    print({k:v for k,v in report.items() if k!='rows'})


if __name__ == '__main__':
    main()
