"""Train a carried five-joint head on independent physical held releases."""

import argparse
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_donor_trial import DESIGNS, ROOT, SHAPE, STATE, DonorBounce, Native, World, _exploration, _seeds
from .tt24_posture import WidePosture
from .tt4_transfer_trial import _hash, _read, _write


CPP = Path(__file__).with_name('tt24_held_donor_native.cpp')
TRAINING = ROOT / 'held-donor-training.json.gz'
CHECKPOINT = ROOT / 'held-donor-checkpoint.npz'
HELD = ROOT / 'held-donor-held.json.gz'
ROUNDS = 23
DIRECTIONS = 8
PRACTICE_CAP = 170
DEV_CAP = 240


class TrainingDonor(DonorBounce):
    sample_drop = staticmethod(WidePosture.sample_drop)


class SourceWorld(World):
    def __init__(self, native, index, bundle, fixed_dispenser):
        self.native = native
        self.arena = TrainingDonor(DESIGNS[index])
        self.regular = DonorBounce(DESIGNS[index])
        self.arena.sample_drop = lambda seed, split, regime: (
            DonorBounce.sample_drop(seed, split, regime) if self.plain else
            TrainingDonor.sample_drop(seed, split, regime))
        self.plain = False
        self.dispenser = fixed_dispenser
        self.q_reference = bundle.equilibrium.copy()
        self.mount = np.asarray([(-.014, .012, .003), (.017, -.012, -.006),
                                 (-.008, -.018, .011)][index])
        arena, model = self.arena, self.arena._model
        mass = (.88, 1., 1.13)[index]
        body = arena._ball_body
        model.body_mass[body] *= mass
        model.body_inertia[body] *= mass
        self.mass_ratio = mass
        mujoco.mj_setConst(model, arena._data)
        arena._prepare_pose(np.r_[self.q_reference, -.1])
        self.home = arena._data.geom_xpos[arena._blade].copy()
        self.handle = native.open(model._address)

    def close(self):
        super().close()
        self.regular.close()

    def plan(self, seeds, *, ordinary=False):
        arena, model = self.arena, self.arena._model
        states = np.empty((len(seeds), mujoco.mj_stateSize(model, STATE)))
        settings = np.empty((len(seeds), 26))
        notes = []
        self.plain = ordinary
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(seed ^ 253100582)
            obs = arena.reset(seed=seed, split='held', regime='changed', render=False)
            if ordinary:
                home = arena._data.geom_xpos[arena._blade].copy()
                velocity, spin = arena._ball_velocity.copy(), arena._ball_spin.copy()
                lead, issued, drag, install = 0, np.zeros(5), 1., 'initial_blade'
                changes = (-1, 1., -1, 1.)
                baseline, bias, lag = 1., np.zeros(3), 0.
                launcher_seed = None
            else:
                per_body = bool(rng.integers(2))
                dispenser = self.regular if per_body else self.dispenser
                launcher_seed = int(rng.integers(1, 2**31 - 1))
                dispenser.reset(seed=launcher_seed, split='held', regime='changed', render=False)
                offset = self.mount + rng.uniform((-.024, -.026, -.014), (.024, .026, .014))
                arena._ball_position[:] = dispenser._ball_position + offset
                velocity, spin = dispenser._ball_velocity.copy(), dispenser._ball_spin.copy()
                arena._ball_velocity[:] = 0.
                arena._ball_spin[:] = 0.
                home, issued = self.home, np.asarray(obs['joints'], float) - self.q_reference
                lead, drag = int(rng.integers(12, 41)), float(rng.uniform(.81, 1.24))
                install = 'body_served' if per_body else 'world_fixed'
                profiles = [(int(rng.integers(77, 108)), float(rng.uniform(.75, .89))),
                            (int(rng.integers(145, 195)), float(rng.uniform(1.10, 1.26)))]
                if rng.integers(2):
                    profiles = [(a[0], b[1]) for a, b in zip(profiles, profiles[::-1], strict=True)]
                changes = tuple(x for row in profiles for x in row)
                baseline = float(rng.uniform(.95, 1.07))
                bias = rng.uniform((-.006, -.008, -.006), (.006, .008, .009))
                lag = float(rng.uniform(0., .38))
                mujoco.mj_forward(model, arena._data)
            mujoco.mj_getState(model, arena._data, states[i], STATE)
            settings[i] = np.r_[arena._drag_force_scale * drag * self.mass_ratio, arena._drop.damping, lead,
                                home, velocity, spin, issued, baseline, changes, bias, lag]
            notes.append(dict(seed=seed, dispenser=install, dispenser_seed=launcher_seed, lead=lead,
                              release=arena._ball_position.tolist(), damping=arena._drop.damping,
                              velocity=velocity.tolist(), spin=spin.tolist(), gravity_initial=baseline,
                              gravity_changes=list(changes), drop_pose=list(arena._drop.pose)))
        return (states, settings), notes


def measure(worlds, batches, matrices, cap):
    return np.asarray([world.play(matrices, batch, cap)
                       for world, batch in zip(worlds, batches, strict=True)])


def utility(scores, cap):
    hits, streak, ticks = np.moveaxis(scores[..., :3], -1, 0)
    reward = .45*(hits >= 1) + .85*(streak >= 2) + .55*(streak >= 4) + 1.05*(streak >= 10)
    reward += .15*streak + .15*ticks/cap
    body = reward.mean(axis=-1)
    return .85*body.mean(axis=0) + .15*body.min(axis=0)


def local_mutations(rng, basis, turn):
    changes = rng.normal(size=(DIRECTIONS, *SHAPE)) * .03
    features = ([1,2,4,5,7,8,10,11,15,16,17,18], [0,1,2,7,8,10,11],
                [3,6,9,12,13,14,19], list(range(SHAPE[1])))
    for i, edit in enumerate(changes):
        column = (turn+i) % 5
        axes = basis[:, column]
        pool = features[i % len(features)]
        selected = rng.choice(pool, min(4, len(pool)), replace=False)
        edit[:, selected] += np.outer(axes, rng.normal(size=len(selected)))
        edit *= rng.choice((.08, .16, .25, .38)) / (1.+.01*turn)
    return changes


def restore_probe_basis(rows):
    effects = []
    for body in rows:
        vectors = np.asarray([row['normalized_change'] for row in body]).reshape(5, 2, 5)
        effects.append((vectors[:, 1] - vectors[:, 0]) / 2.)
    response = np.asarray(effects).mean(axis=0).T
    basis = np.linalg.solve(response.T @ response + .08**2 * np.eye(5), response.T)
    return basis / np.linalg.norm(basis, axis=0)


def train(resume=False):
    bundle = CarriedWristPolicy()
    native = Native(CPP)
    dispenser = BounceArena(view='none')
    worlds = [SourceWorld(native, i, bundle, dispenser) for i in range(len(DESIGNS))]
    rng = np.random.default_rng(56907643)
    base = bundle.weights.copy()
    best = base.copy()
    start = perf_counter()
    try:
        stress_seeds = _seeds(950317320, len(worlds), 10)
        ordinary_seeds = _seeds(950317321, len(worlds), 5)
        stress = [world.plan(row) for world, row in zip(worlds, stress_seeds, strict=True)]
        regular = [world.plan(row, ordinary=True) for world, row in zip(worlds, ordinary_seeds, strict=True)]
        score_held = lambda opts: measure(worlds, [case[0] for case in stress], opts, DEV_CAP)
        score_ordinary = lambda opts: measure(worlds, [case[0] for case in regular], opts, DEV_CAP)
        if resume:
            report = _read(TRAINING)
            if report['bundle_sha256'] != _hash(BUNDLE):
                raise ValueError('the carried actor has changed since source training started')
            if report.get('selected_before_target'):
                if _hash(CHECKPOINT) != report['checkpoint_sha256']:
                    raise ValueError('completed source checkpoint differs from the receipt')
                return
            basis = restore_probe_basis(report['source_motor_probes'])
            initial = np.asarray(report['initial_held'])
            floor = report['ordinary_utility_floor']
            quality = float(utility(initial, DEV_CAP)[0])
            for epoch, row in enumerate(report['updates']):
                local_mutations(rng, basis, epoch)
                launches = rng.integers(1, 2**31-1, size=(len(worlds), 2)).tolist()
                if launches != [[plan['seed'] for plan in group] for group in row['physical_plans']]:
                    raise ValueError('source proposal random stream cannot resume exactly')
                if row['selected'] is not None:
                    best = np.asarray(row['candidate_weights'][row['selected']])
                quality = row['utility']
            print('resuming source training after', len(report['updates']), 'persisted rounds', flush=True)
        else:
            for world in worlds:
                world.plain = False
            basis, probes = _exploration(worlds)
            initial, baseline = score_held(base), score_ordinary(base)
            floor = .87 * float(utility(baseline, DEV_CAP)[0])
            quality = float(utility(initial, DEV_CAP)[0])
            report = dict(actor_observations='source exact physical states with randomized optical position bias and velocity lag',
                          external_release=True, bundle_sha256=_hash(BUNDLE), cap=PRACTICE_CAP,
                          validation_cap=DEV_CAP, initial_held=initial.tolist(), initial_prior=baseline.tolist(),
                          ordinary_utility_floor=floor, source_motor_probes=probes,
                          held_validation=[case[1] for case in stress],
                          ordinary_validation=[case[1] for case in regular], updates=[])
        for epoch in range(len(report['updates']), ROUNDS):
            directions = local_mutations(rng, basis, epoch)
            actors = np.clip(np.concatenate((best[None], best+directions, best-directions)), base-.48, base+.48)
            launches = rng.integers(1, 2**31-1, size=(len(worlds), 2)).tolist()
            batches = [world.plan(row) for world, row in zip(worlds, launches, strict=True)]
            trial = measure(worlds, [case[0] for case in batches], actors, PRACTICE_CAP)
            rankings = np.argsort(-utility(trial, PRACTICE_CAP), kind='stable')
            leaders = [i for i in rankings if i][:3]
            validation = score_held(actors[leaders])
            ordinary = score_ordinary(actors[leaders])
            selected = None
            values = utility(validation, DEV_CAP)
            retention = utility(ordinary, DEV_CAP)
            for chosen in np.argsort(-values, kind='stable'):
                if float(values[chosen]) > quality and float(retention[chosen]) >= floor:
                    best, quality, selected = actors[leaders[chosen]].copy(), float(values[chosen]), int(chosen)
                    break
            report['updates'].append(dict(generation=epoch+1, physical_plans=[case[1] for case in batches],
                                          candidate_weights=actors[leaders].tolist(), candidate_indices=[int(i) for i in leaders],
                                          sampled=trial.tolist(), validation=validation.tolist(),
                                          ordinary=ordinary.tolist(), selected=selected, utility=quality))
            if selected is not None or epoch % 4 == 3 or epoch+1 == ROUNDS:
                print('held donor train', epoch+1, round(perf_counter()-start, 1), round(quality, 3),
                      'selected', selected, 'held baseline', round(float(utility(initial,DEV_CAP)[0]),3), flush=True)
            _write(TRAINING, report)
        temporary = CHECKPOINT.with_name(CHECKPOINT.stem + '.tmp.npz')
        np.savez_compressed(temporary, trained=best, original=base)
        temporary.replace(CHECKPOINT)
        report.update(checkpoint_sha256=_hash(CHECKPOINT), selected_before_target=True,
                      elapsed_seconds=perf_counter()-start)
        _write(TRAINING, report)
    finally:
        for world in worlds:
            world.close()
        dispenser.close()
        native.working.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(args.resume)


if __name__ == '__main__':
    main()
