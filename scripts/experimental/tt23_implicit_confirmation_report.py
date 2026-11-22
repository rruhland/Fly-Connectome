"""Audit independent schedules and tabulate censored time-to-ten for actual online agents."""
import argparse
import itertools
import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_implicit_online import CONFIRM_SEEDS, LIVE_SEED, OUT, SOURCE, WIDE_SEEDS, live_schedule
from .tt23_jog_bank import GENERATORS
from .tt23_sequence import summarize


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(rows), 0, -1)
    counts = {k: np.array([r['paired'][k]['point'] or 9 for r in rows]) for k in ('learned', 'flat')}
    ax.axvspan(8.6, 9.4, color='#f4ece5', zorder=0)
    ax.hlines(y, np.minimum(counts['flat'], counts['learned']), np.maximum(counts['flat'], counts['learned']),
              color='#939d9b', lw=1.4)
    ax.scatter(counts['flat'], y, marker='o', edgecolors='#824222', facecolors='white', label='Flat prior',
               s=46, zorder=3)
    ax.scatter(counts['learned'], y, marker='x', color='#007465', label='Pong first', s=50, zorder=4)
    ax.set(yticks=y, yticklabels=np.arange(1, len(rows)+1), ylim=(.4,len(rows)+.6), xlim=(.65,9.45),
           xticks=np.arange(1,10), xticklabels=[*range(1,9),'Missed'],
           xlabel='First target point with ten alternating legal returns', ylabel='Independent training stream')
    ax.grid(axis='x', alpha=.15)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(only_wide=False):
    series = WIDE_SEEDS if only_wide else CONFIRM_SEEDS
    prefix = 'implicit-wide' if only_wide else 'implicit-confirm'
    prior = {s for generator in (*GENERATORS, LIVE_SEED) for s, _ in live_schedule(generator)}
    if only_wide:
        prior.update(s for generator in CONFIRM_SEEDS for s, _ in live_schedule(generator))
    rows = []
    for i, seed in enumerate(series, 1):
        schedule = live_schedule(seed, only_wide=only_wide)
        claimed = {s for s, _ in schedule}
        if len(claimed) != 8 or prior & claimed:
            raise ValueError('Duplicate destination starts')
        prior.update(claimed)
        path = OUT / f'{prefix}-{i}.json.gz'
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
    _write(OUT/f'{prefix}-summary.json.gz', report)
    plot(rows, OUT/f'{prefix}-learning.png')
    print({k:v for k,v in report.items() if k!='rows'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wide', action='store_true')
    main(only_wide=parser.parse_args().wide)
