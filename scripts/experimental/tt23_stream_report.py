"""Summarize paired chronological training streams, including points that never rally."""

import json

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt23_live_learning import COACH, MOTOR_PRACTICE, OUT, STREAM_SEEDS, stream_schedule
from .tt23_sequence import summarize


def summarize_streams():
    streams = []
    prior_starts = {row['seed'] for path, key in ((MOTOR_PRACTICE, 'schedule'), (COACH, 'rows'))
                    for row in _read(path)[key]}
    all_starts = set()
    for index in range(len(STREAM_SEEDS)):
        receipt = _read(OUT / f'online-rally-stream-{index + 1}.json.gz')
        cases = receipt['cases']
        planned, _ = stream_schedule(index)
        if receipt.get('verified') != len(planned) * 2:
            raise ValueError('Every arm of every point must pass physical replay')
        if [point['seed'] for point in cases] != [point['seed'] for point in planned]:
            raise ValueError('Incomplete or reordered training stream')
        new_starts = {point['seed'] for point in cases}
        if new_starts & (all_starts | prior_starts):
            raise ValueError('Test start overlapped another stream or target practice')
        all_starts.update(new_starts)
        methods = {}
        for method in ('pong', 'cold'):
            rows = [point['results'][method] for point in cases]
            scores = summarize(rows)
            if scores != receipt['summary'][method]:
                raise ValueError('Replay summary changed')
            methods[method] = dict(scores, choices=[point['choices'][method] for point in cases],
                                   arm_curve=np.cumsum([row['outcome']['legal_returns'] for row in rows]).tolist())
        streams.append(methods)
    return streams


def plot(streams, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(streams), sharey=True, figsize=(10, 3.3))
    for index, (ax, pair) in enumerate(zip(axes, streams, strict=True)):
        for method, color, label in [('pong', '#007966', 'Pong first'), ('cold', '#a74c2a', 'Table tennis only')]:
            counts = np.r_[0, pair[method]['arm_curve']]
            ax.step(np.arange(len(counts)), counts, where='post', color=color, label=label)
            first = pair[method]['first_ten']
            if first:
                x = first['point']
                ax.scatter(x, counts[x], color=color, s=30, marker='D', zorder=3)
        ax.set(title=f'Stream {index+1}', xticks=[0, 4, 8, 12], xlabel='Target points played')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Cumulative legal returns by the arm')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    fig.text(.5, .01, 'Diamond marks the first point with ten alternating legal returns.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .89))
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    pairs = summarize_streams()
    _write(OUT / 'online-rally-summary.json.gz', dict(streams=pairs))
    plot(pairs, OUT / 'online-rally-learning.png')
    for index, pair in enumerate(pairs, 1):
        print(index, json.dumps({key: {k: row[k] for k in ('first_ten', 'learner_returns', 'ten_or_more',
                                                           'twenty_or_more', 'longest')}
                                for key, row in pair.items()}))
