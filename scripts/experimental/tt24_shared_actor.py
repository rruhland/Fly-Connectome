"""Transfer one framewise joint policy across table tennis and paddle bouncing.

The optional instructor supplies actuator labels only while collecting explicitly
assisted practice. Held control uses one RGB image, own joints and own commands.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt19_so101 import RATES
from .tt23_sequence import course
from .tt24_bounce import BounceArena
from .tt24_motor import optical_sightings
from .tt24_rally_eye import DATA, SingleStation
from .tt24_rally_policy import JointPolicy, ViewHistory
from .tt24_rally_trial import COACHED, STEP_SIZE, _read, returns


DIR = Path('artifacts/tt24')
BUNDLE = DIR / 'shared-bounce-examples.npz'
RATE = np.array(RATES)


def source(data):
    return (data['left'], data['postures'], data['rates'])


def rows(samples):
    if not samples:
        return np.empty((0,38)), np.empty((0,6)), np.empty((0,6))
    return tuple(map(np.array, zip(*samples)))


def fit(groups, dt, *, body=.5, mix=.35):
    return JointPolicy(*(np.concatenate(part) for part in zip(*groups, strict=True)), RATE*dt,
                       neighbors=24, body=body, mix=mix)


class Observation:
    def __init__(self, obs, dt):
        self.history = ViewHistory()
        self.last = None
        self.target = np.r_[obs['joints'], -.10]
        self.dt = dt

    def state(self, obs, blind=False):
        ball, _ = (None, None) if blind else optical_sightings(obs['rgb'])
        center = None
        if ball is not None:
            h, w = obs['rgb'].shape[:2]
            center = (ball.center*h + np.array([(w-1)/2,(h-1)/2]) + .5)*480/w-.5
        return self.history.observe_pixels((center, None), np.r_[obs['joints'], -.1], self.target)

    def issued(self, command):
        self.history.issued(command)
        self.target += self.dt * RATE * command


def play(arena, seed, policy, *, collect=False, query=False, blind=False, split='held', regime='standard'):
    obs = arena.reset(seed=seed, split=split, regime=regime)
    interface = Observation(obs, arena.dt)
    labels = []
    teacher = None
    if collect or query:
        from .tt24_bounce_ceiling import PrivilegedCeiling
        teacher = PrivilegedCeiling(arena)
    n = 0
    done = False
    while not done:
        state = interface.state(obs, blind=blind)
        if collect or (query and n%2==0):
            expert = np.r_[teacher.command(), 0.]
            labels.append((state.copy(), interface.target + expert * RATE * arena.dt, expert.copy()))
        if collect:
            command = expert
        elif policy is None:
            command = np.zeros(6)
        else:
            command = policy.command(state, interface.target)
            command[5] = 0.
        obs, _, done = arena.step(command[:5], render=not blind)
        interface.issued(command)
        n += 1
    metrics = arena.judge_metrics()
    return dict(rebounds=metrics['rebounds'], consecutive=metrics['longest_consecutive_rebounds'], ticks=n), labels


def gather(arena, path):
    seeds = np.random.default_rng(231485).integers(1000000, 2**31-1, 8).tolist()
    all_rows = []
    lengths = []
    for s in seeds:
        result, samples = play(arena, s, None, collect=True, split='practice')
        all_rows.extend(samples)
        lengths.append(len(samples))
        print('instructor', s, result, flush=True)
    x, p, a = rows(all_rows)
    np.savez_compressed(path, x=x, p=p, a=a, lengths=lengths, seeds=seeds)


def group(data, demos):
    n = sum(data['lengths'][:demos])
    return tuple(data[k][:n] for k in ('x','p','a'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--starts', default=12, type=int)
    parser.add_argument('--seed', default=3378915, type=int)
    parser.add_argument('--continued', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--retention', action='store_true')
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    arena = BounceArena(view='wrist',width=96,height=72)
    try:
        if args.collect:
            gather(arena, BUNDLE)
            return
        with np.load(DATA) as matches, np.load(BUNDLE) as demos:
            table = source(matches)
            examples = {d: group(demos,d) for d in (1,4,8)}
            budget = dict(teacher_ticks=demos['lengths'].tolist(), teacher_seeds=demos['seeds'].tolist(),
                          table_labels=len(table[0]))
        if args.continued:
            with np.load(args.continued) as extra:
                corrected = tuple(extra[k] for k in ('x','p','a'))
        else:
            corrected = rows([])
        mode_groups = dict(table_only=[table], one=[examples[1]], shared_one=[table,examples[1]],
                           four=[examples[4]], shared_four=[table,examples[4]],
                           shared_eight=[table,examples[8]])
        if len(corrected[0]):
            mode_groups['shared_corrected'] = [table,examples[4],corrected]
        if args.extended:
            mode_groups = {k:v for k,v in mode_groups.items() if k in ('table_only','shared_four','shared_corrected')}
        starts = np.random.default_rng(args.seed).integers(1000000,2**31-1,args.starts).tolist()
        results = {}
        if args.retention:
            from .tt24_rally_eye import SingleEye
            station = SingleStation(_read(COACHED)['metadata']['opening']['far_profile'],'left')
            try:
                for label, groups in mode_groups.items():
                    pupil = fit(groups, 1/30.)
                    scores = []
                    for row in course({'i':3190547,'j':9317339})[:args.starts]:
                        r = station.play(row,'demo_full',pupil,new_history=SingleEye,live_opening=True)
                        scores.append(dict(seed=row['seed'],legal=len(returns(r,1)),longest=r['longest']))
                    results[label] = scores
                    print(label,scores,flush=True)
            finally:
                station.arena.close()
        else:
            for label, groups in mode_groups.items():
                pupil = fit(groups, arena.dt)
                scores = [play(arena, s, pupil)[0] for s in starts]
                results[label] = scores
                print(label,[r['consecutive'] for r in scores],flush=True)
            chosen = 'shared_corrected' if 'shared_corrected' in mode_groups else 'shared_four'
            pupil = fit(mode_groups[chosen], arena.dt)
            results['blind'] = [play(arena,s,pupil,blind=True)[0] for s in starts]
            results['hold'] = [play(arena,s,None)[0] for s in starts]
        with gzip.open(args.output,'wt') as out:
            json.dump(dict(budget=budget,starts=starts,labels=len(corrected[0]),results=results),out)
    finally:
        arena.close()


if __name__ == '__main__':
    main()
