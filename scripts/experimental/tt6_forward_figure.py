"""Plot held-out receiving performance from the matched forward-course receipts."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .tt4_transfer_trial import ROOT, _read
from .tt6_score import course_rows


def plot(output):
    source = ROOT / 'artifacts/tt6'
    labels = ('Mini only', 'Ordinary arm practice', 'Varied arm practice')
    colors = ('#586879', '#977044', '#007479')
    arms = []
    counts = []
    for condition in ('cold', 'ordinary', 'experienced'):
        condition_counts = []
        names = []
        for prefix in ('forward-', 'forward-repeat-'):
            course = _read(source / f'{prefix}{condition}.json.gz')
            course_rows(course)
            for visit in course['visits']:
                names.append('Gantry' if visit['body'] == 'gantry' else '5-axis arm')
                points = visit['points']
                condition_counts.append(sum(p['player_returns'] > 0 for p in points))
        if arms and arms != names:
            raise ValueError('unequal courses')
        arms = names
        counts.append(condition_counts)

    fig, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
    y = np.arange(len(arms))
    for i, (label, color, values) in enumerate(zip(labels, colors, counts, strict=True)):
        offset = .19 * (i - 1)
        ax.scatter(values, y + offset, s=52, color=color, label=label, zorder=3)
    ax.invert_yaxis()
    ax.set_yticks(y, [f'Course {course}{j}   {arm}' for course in ('A', 'B')
                      for j, arm in enumerate(arms[:3] if course == 'A' else arms[3:], 1)])
    ax.set_xticks(range(8))
    ax.set_xlim(-.1, 7.2)
    ax.set_xlabel('Serves with a legal opening return, out of seven')
    ax.set_title('Opening returns on six new installations', loc='left', fontsize=13)
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.23), frameon=False, ncols=3)
    fig.savefig(output, dpi=155)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    plot(parser.parse_args().output)
