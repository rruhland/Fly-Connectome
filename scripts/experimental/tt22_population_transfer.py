"""Compare native Pong performance ranks with actual SO-101 returns across all 16 options."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.special import betaincinv
from scipy.stats import rankdata

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt22_duration_transfer import MACRO, PONG, RECEIPT, STREAMS
from .tt22_joint_practice import draw, point
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer_trial import MotorArena

SEED = 178329541
BUDGETS = (0, 4, 8, 16, 32, 48)
OUTPUT = ROOT / 'artifacts/tt22/population-transfer.json.gz'


def games(arena, template, options, optics):
    return [[point(arena, template, k, obs) for k in options] for obs in optics]


def learning_curve(source_score, practice, held):
    count = len(source_score)
    rng = np.random.default_rng(391884)
    native_rank = (rankdata(source_score) - 1) / (count - 1)
    methods = ('pong', 'cold', 'permuted')
    curves = {k: [] for k in methods}
    returns = {k: [] for k in methods}
    for _ in range(256):
        order = rng.permutation(len(practice))
        # Inverse-CDF sampling pairs exploration noise for corresponding options.
        uniforms = rng.uniform(.00001, .99999, size=(len(order), count))
        tie = rng.random((len(order)+1, count)) * 1.e-10
        for name, ranks in [('pong', native_rank), ('cold', np.full(count, .5)),
                            ('permuted', rng.permutation(native_rank))]:
            a = .5 + 2 * ranks
            b = 3 - a
            actual, checks = [], []
            for i in range(len(order)+1):
                if i in BUDGETS:
                    idx = int(np.argmax(a/(a+b) + tie[i]))
                    checks.append(dict(option=idx, legal=int(held[:, idx].sum())))
                if i == len(order):
                    break
                idx = int(np.argmax(betaincinv(a, b, uniforms[i])))
                good = practice[order[i], idx]
                actual.append(int(good))
                a[idx] += good
                b[idx] += 1-good
            curves[name].append(checks)
            returns[name].append(sum(actual))
    return dict(budgets=BUDGETS, checkpoints=curves, earned_returns=returns)


def analyze(report):
    n = len(report['options'])
    def matrix(key, measure):
        return np.array([[int(row[measure] > 0) for row in batch] for batch in report[key]])
    held = matrix('held', 'legal')
    practice = matrix('practice', 'legal')
    source = np.array(report['source_scores'])
    target = held[:, :n].mean(axis=0)
    winner = int(report['winner'])
    other = np.arange(n) != winner
    rng = np.random.default_rng(805064)
    rival_ranks = rankdata(target)
    source_ranks = rankdata(source)
    corr = np.corrcoef(source_ranks, rival_ranks)[0, 1]
    x = (source_ranks - source_ranks.mean())/np.linalg.norm(source_ranks - source_ranks.mean())
    y = (rival_ranks - rival_ranks.mean())/np.linalg.norm(rival_ranks - rival_ranks.mean())
    null = np.stack([rng.permutation(x) for _ in range(20000)]) @ y
    pairs = {}
    for name, rival in [('unselected',held[:,:n][:, other].mean(axis=1)), ('neutral', held[:,n])]:
        diff = held[:,winner] - rival
        # Keep the two sampling strata's sizes fixed and all corresponding options together.
        pool = np.split(diff, 2)
        samples = np.column_stack([group[rng.integers(len(group),size=(8192,len(group)))].mean(axis=1)
                                   for group in pool]).mean(axis=1)
        pairs[name] = dict(delta=float(diff.mean()), ci95=np.quantile(samples,[.025,.975]).tolist())
    return dict(rates=target.tolist(), neutral=float(held[:,n].mean()),
                contacts=matrix('held','contact').sum(axis=0).tolist(),
                selected=int(held[:,winner].sum()), points=len(held), top_rank=int(1+sum(target>target[winner])),
                tied=int(sum(target==target[winner])), spearman=float(corr),
                permutation_p=float((1+sum(np.abs(null)>=abs(corr)-1e-12))/(1+len(null))),
                contrasts=pairs, bandit=learning_curve(source,practice[:,:n],held[:,:n]))


def run(path):
    native = _read(PONG)
    macro = np.array(_read(MACRO)['parameters'])
    prev = _read(RECEIPT)
    n = len(native['candidates'])
    options = [np.r_[native['candidates'][str(i)][1:], np.zeros(4)] for i in range(n)]
    theta = options + [np.zeros(6)]
    source = [native['screen_duration_ticks'][str(i)] for i in range(n)]
    practice = [r['optical'] for group in prev['streams'].values() for r in group]
    rng = np.random.default_rng(SEED)
    starts = [pair for span in STREAMS.values() for pair in draw(rng,32,span)]
    if len(set(s for s,_ in starts)) != len(starts) or {s for s,_ in starts} & {r['seed'] for r in practice}:
        raise ValueError('held overlap')
    doc = dict(seed=SEED, source_sha256=_hash(PONG), macro_sha256=_hash(MACRO), old_sha256=_hash(RECEIPT),
               options=[o.tolist() for o in options], source_scores=source, winner=int(native['selected']),
               practice_optics=practice, held_optics=[], practice=[], held=[])
    arena = MotorArena()
    try:
        for key in ('practice','held'):
            if key == 'held':
                doc['held_optics'] = [prefix(arena,s,lane) for s,lane in starts]
            optics = doc[key+'_optics']
            for j in range(0,len(optics),8):
                doc[key].extend(games(arena,macro,theta,optics[j:j+8]))
                _write(path,doc)
                print(key, len(doc[key]), flush=True)
        doc['analysis'] = analyze(doc)
        for j, sight in enumerate(doc['held_optics']):
            for i in (doc['winner'], n):
                if point(arena,macro,theta[i],sight) != doc['held'][j][i]:
                    raise ValueError('physical score replay changed')
        doc['verified_held_endpoints'] = 2*len(starts)
        _write(path,doc)
    finally:
        arena.close()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    run(parser.parse_args().output)
