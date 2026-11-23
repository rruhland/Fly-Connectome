"""Learn cross-view visual coordinates from paired images and measured joints.

The moving camera may use silhouette angular size. Neither camera calibration,
world coordinates nor simulator object dimensions are available to the mapper.
"""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt23_coach_trial import PRACTICE
from .tt23_sequence import CONFIG, SequenceArena, course, verify
from .tt24_motor import optical_sightings
from .tt24_rally_eye import DATA, DIMS, SingleEye, SingleStation
from .tt24_rally_policy import JointPolicy, orange_center
from .tt24_rally_trial import COACHED, STEP_SIZE, PublicBody, archive_schedule, demo_schedule, returns


OPTICS = Path('artifacts/tt24/eye-depth-pairs.npz')


def measure(frame):
    mover, _ = optical_sightings(frame)
    if mover is None:
        return None
    return np.r_[mover.center, np.log(mover.radius)]


def optical_bank():
    values = {key: [] for key in ('wrist', 'target', 'joints', 'seed', 'clip')}
    arena = SequenceArena(**dict(CONFIG, width=DIMS[0], height=DIMS[1]))
    arena.streams = ('wrist', 'left')
    try:
        for clip, row in enumerate(demo_schedule(archive_schedule())):
            arena.lane = row['lane']
            arena.reset(seed=row['seed'], server=-1, render=False)
            previous = None
            stop = returns(row, 1)[-1] + 1
            for near, far in zip(row['actions'][:stop], row['rival_actions'][:stop], strict=True):
                frames = arena.render_cameras()
                feature = measure(frames['wrist'])
                target = orange_center(frames['left'], previous)
                if target is not None:
                    previous = target
                values['wrist'].append(np.full(3, np.nan) if feature is None else feature)
                values['target'].append(np.full(2, np.nan) if target is None else target)
                values['joints'].append(arena.joints.copy())
                values['seed'].append(row['seed'])
                values['clip'].append(clip)
                arena._aim = arena._command_to = lambda *args: far
                _, _, done, cap, _ = arena.step(near, render=False)
                if done or cap:
                    break
            assert digest_json(arena.events) == digest_json([v for v in row['events'] if v['step'] <= arena.steps])
        np.savez_compressed(OPTICS, **values)
    finally:
        arena.close()


class LensRegression:
    def __init__(self, raw, output):
        self.origin = np.median(raw, axis=0)
        self.scale = np.maximum(.05, np.std(raw, axis=0))
        forms = self.features(raw)
        penalty = np.eye(forms.shape[-1]) * 2.
        penalty[0,0] = .0001
        self.weights = np.linalg.solve(forms.T@forms + penalty, forms.T@output)

    def features(self, raw):
        x = np.clip((np.atleast_2d(raw) - self.origin)/self.scale, -4, 4)
        a, b = np.triu_indices(x.shape[-1])
        return np.column_stack((np.ones(len(x)), x, x[:,a]*x[:,b]))

    def predict(self, raw):
        return self.features(raw) @ self.weights


class MappedHistory(SingleEye):
    def __init__(self, lens):
        super().__init__()
        self.lens = lens

    def observe(self, frames, joints, issued_position):
        v = measure(frames['left'])
        pos = None if v is None else self.lens.predict(np.r_[v, joints[:5]])[0]
        return self.observe_pixels((pos, None), joints, issued_position)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect', action='store_true')
    p.add_argument('--limit', type=int, default=16)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.collect:
        optical_bank()
        return
    with np.load(OPTICS) as data, np.load(DATA) as motor:
        valid = np.isfinite(data['wrist']).all(axis=1) & np.isfinite(data['target']).all(axis=1)
        raw = np.column_stack((data['wrist'], data['joints'][:,:5]))
        output = data['target']
        chosen = data['clip']==0
        source = motor['seed']!=motor['seed'][0]
        policy = JointPolicy(motor['left'][source], motor['postures'][source], motor['rates'][source], STEP_SIZE)
        rng = np.random.default_rng(189311)
        perm = output.copy()
        perm[valid] = output[rng.permutation(np.flatnonzero(valid))]
        lenses = dict(one=LensRegression(raw[chosen&valid], output[chosen&valid]),
                      full=LensRegression(raw[valid], output[valid]), shuffled=LensRegression(raw[valid], perm[valid]))
        folds = {}
        for seed in np.unique(data['seed']):
            train = valid & (data['seed'] != seed)
            test = valid & ~train
            mapped = LensRegression(raw[train], output[train]).predict(raw[test])
            errors = np.linalg.norm(mapped - output[test], axis=1)
            folds[int(seed)] = dict(paired=int(test.sum()), median_error=float(np.median(errors)),
                                    median_constant=float(np.median(np.linalg.norm(output[test]-output[train].mean(0),axis=1))))
        budgets = dict(actor_samples=int(source.sum()), one=int(chosen.sum()), full=len(raw),
                       usable_one=int((chosen&valid).sum()), usable_full=int(valid.sum()))
    station = SingleStation(_read(COACHED)['metadata']['opening']['far_profile'], 'wrist')
    cases = []
    try:
        for row in course({'g': 199538051, 'h': 753140913})[:args.limit]:
            outputs = {}
            for mode in ('full', 'delayed', 'one', 'shuffled'):
                lens = lenses['full'] if mode == 'delayed' else lenses[mode]
                outputs[mode] = station.play(row, 'sight_delay' if mode=='delayed' else 'demo_full', policy,
                                             new_history=lambda: MappedHistory(lens), live_opening=True)
            print(row['seed'], [len(returns(o,1)) for o in outputs.values()], flush=True)
            cases.append(dict(seed=row['seed'], results=outputs))
        _write(args.out, dict(inputs='wrist RGB and measured joint angles; paired optical training',
                              calibration=budgets, held_lens_folds=folds, cases=cases, verified=verify(cases)))
    finally:
        station.arena.close()


if __name__ == '__main__':
    main()
