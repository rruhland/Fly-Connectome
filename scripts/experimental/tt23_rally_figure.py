"""Show every matched start in the two successful target-practice experiments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .tt4_transfer_trial import _read
from .tt23_optical_contact import OUT


def render():
    untrained = _read(OUT / 'after-service-zero.json.gz')['rows']
    practiced = _read(OUT.parent / 'tt22' / 'after-service.json.gz')['rows']['neutral']
    coaches = _read(OUT / 'coach-held.json.gz')['rows']
    comparisons = [('Arm practice', 'Without corrections', 'After four practice points', untrained, practiced),
                   ('Opponent practice', 'Original opponent', 'Trained opponent', coaches['stock'], coaches['learned'])]
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(9, 4))
    for ax, (title, old, new, before, after) in zip(axes, comparisons, strict=True):
        if [r['seed'] for r in before] != [r['seed'] for r in after]:
            raise ValueError('Start order differs between conditions')
        y = np.arange(len(before), 0, -1)
        left, right = [[r['longest'] for r in group] for group in (before, after)]
        ax.axvspan(10, 20, color='#ecefe8', zorder=0)
        ax.hlines(y, np.minimum(left, right), np.maximum(left, right), color='#8c978e', lw=1.5)
        ax.scatter(left, y, marker='o', facecolors='white', edgecolors='#3a5366', s=49, label=old, zorder=3)
        ax.scatter(right, y, marker='x', color='#ad4726', s=47, label=new, zorder=4)
        ax.set(yticks=y, yticklabels=[str(i + 1) for i in range(len(y))], xlabel='Consecutive legal returns',
               title=title, ylim=(.5,len(y)+.5), xlim=(-1.6, 44.5), xticks=[0,10,20,30,40])
        ax.legend(loc='upper left', bbox_to_anchor=(0, -.21), frameon=False, fontsize=9)
        ax.grid(axis='x', alpha=.18)
    axes[0].set_ylabel('Matched start within each experiment')
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(OUT / 'rally-practice.png', dpi=155)
    plt.close(fig)


if __name__ == '__main__':
    render()
