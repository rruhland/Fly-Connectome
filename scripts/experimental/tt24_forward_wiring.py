"""Infer hidden actuator routing from a causal encoder forward model."""

import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _write
from .tt24_bounce import BounceArena
from .tt24_compound_observer import read, recover
from .tt24_effects import transmission
from .tt24_motor_recall import DELAY, TAPS, WINDOW, MapFit, MotorObserver, sensor
from .tt24_predict_witness import BODY_SEEDS, DATA as BODY_TRACE


DIR = Path('artifacts/tt24')
CALIBRATION = DIR / 'forward-wiring-model.npz'
REPORT = DIR / 'forward-wiring-identification.json.gz'
HISTORY = 6
CLOCK_TAPS = (0, 1, 2, 3)
N_MOTORS = 5
IDENTITY = np.eye(N_MOTORS)
BODY_SEED = 809803364
BUDGET = (8, 16, 32, 64)
BODY_DROPS = 12
REGIMES = ('standard', 'changed')


class BallFree(BounceArena):
    """Run the installed arm and servo physics without ball contacts."""
    def __init__(self, arm, frames):
        super().__init__(view='none', max_steps=frames, arm=arm)
        self._model.geom_contype[self._ball_geom] = 0
        self._model.geom_conaffinity[self._ball_geom] = 0

    def _judge_substep(self, _incoming):
        return 0.


def window(states, commands, index):
    """At index t the newest state is sampled before the first listed command."""
    frames = np.maximum(0, index[:, None] - np.arange(5))
    history = states[frames]
    q, v = history[:, 0, :5], history[..., 5:]
    acceleration = np.diff(-v, axis=1)
    u = np.vstack((np.zeros((1, N_MOTORS)), commands))
    actions = u[np.maximum(0, index[:, None] + 1 - CLOCK_TAPS)]
    # Public servo angles and velocities also capture pose-varying coupling.
    features = np.c_[q, v[:, 0], v[:, 0] * abs(v[:, 0]), acceleration.reshape(len(index), -1),
                     (q[:, :, None] * v[:, 0, None, :]).reshape(len(index), -1)]
    return features, actions


def examples(clips):
    numerical = []
    for states, commands in clips:
        index = np.arange(len(commands))
        x, u = window(states, commands, index)
        y = np.diff(states[:, 5:], axis=0)
        numerical.append((x, u, y))
    return tuple(np.concatenate(parts) for parts in zip(*numerical, strict=True))


def design(x, u, center, scale):
    return np.c_[np.ones(len(x)), np.clip((x-center)/scale, -7., 7.), u.reshape(len(x), -1)]


class Forward:
    """Predict one-step encoder acceleration and factor its linear motor contribution."""
    def __init__(self, x, u, y, contrasts=None):
        self.center = x.mean(axis=0)
        self.scale = np.maximum(.006, x.std(axis=0))
        data = design(x, u, self.center, self.scale)
        penalty = np.r_[.10, np.full(x.shape[1], 2.0), np.full(u.shape[1]*N_MOTORS, .24)]
        gram, cross = data.T @ data, data.T @ y
        if contrasts is not None:
            a, b = contrasts
            difference = design(*a[:2], self.center, self.scale) - design(*b[:2], self.center, self.scale)
            # Same-pose randomized contrasts carry especially direct evidence
            # about command effects, with free-motion drift largely removed.
            gram += 12. * difference.T @ difference
            cross += 12. * difference.T @ (a[2]-b[2])
        params = np.linalg.solve(gram + np.diag(penalty), cross)
        self.auto = params[:x.shape[1]+1]
        self.response = params[x.shape[1]+1:].reshape(len(CLOCK_TAPS), N_MOTORS, N_MOTORS)
        yhat = data @ params
        rms = np.sqrt(np.mean((y-yhat)**2, axis=0))
        self.noise = np.maximum(.025, rms)

    def predict(self, states, commands):
        """Compute an intercept and a causal regression for the unknown 5-by-5 routing."""
        t = len(states)-1
        x, u = window(np.asarray(states), np.asarray(commands), np.array([t]))
        autoreg = np.r_[1., np.clip((x[0]-self.center)/self.scale, -7., 7.)] @ self.auto
        readout = np.einsum('lpo,li->opi', self.response, u[0]).reshape(N_MOTORS, -1)
        return autoreg, readout

    def residuals(self, x, u, y):
        a = np.c_[np.ones(len(x)), np.clip((x-self.center)/self.scale, -7., 7.)]
        return y - a @ self.auto - np.einsum('lpo,nlp->no', self.response, u)

    def fields(self):
        return {name: getattr(self, name) for name in ('center', 'scale', 'auto', 'response', 'noise')}

    @classmethod
    def restore(cls, fields):
        obj = cls.__new__(cls)
        for key, value in fields.items():
            setattr(obj, key, value)
        return obj


def archived_clips():
    probes = read('public-body.json.gz')['data'], read('public-morph-calibration.json.gz')['data']
    games = (read('public-carry.json.gz')['modes']['carried'][:4],
             read('public-morph.json.gz')['modes']['recalibrated_public']['episodes'][:4])
    with np.load(BODY_TRACE, allow_pickle=False) as archive:
        body = [(archive[f'body_obs_{seed}'], archive[f'body_action_{seed}']) for seed in BODY_SEEDS]
    arenas = [BounceArena(view='none', max_steps=450, arm=arm) for arm in ('stock', 'alternate')]
    try:
        clips = []
        for index, arena in enumerate(arenas):
            regime = REGIMES[index]
            clips.append((recover(arena, probes[index], 'held' if index else 'practice', regime),
                          recover(arena, games[index], 'held', regime)))
    finally:
        for arena in arenas:
            arena.close()
    return body, clips


def paired_responses(arm, regime, count):
    arena = BallFree(arm, 40)
    rng = np.random.default_rng(2963342 + int(arm == 'alternate'))
    samples = ([], [])
    issued = 0
    try:
        for _ in range(count):
            seed = int(rng.integers(2**31-1))
            t = int(rng.integers(0, 14))
            controls = rng.uniform(-.16, .16, size=(t+8, N_MOTORS))
            vector = rng.normal(size=N_MOTORS)
            effect = .37 * vector / np.linalg.norm(vector)
            traces = []
            for side, sign in enumerate((1., -1.)):
                stream = controls.copy()
                stream[t] += effect * sign
                states, commands = conduct(arena, seed, regime, stream, IDENTITY, len(stream))
                samples[side].append((states, commands))
                traces.append(states[:t+1])
                issued += len(stream)
            np.testing.assert_array_equal(*traces)
    finally:
        arena.close()
    a, b = (examples(stream) for stream in samples)
    perturbed = np.any(np.abs(a[1]-b[1]) > 1.e-8, axis=(1,2))
    # Include the motor-induced response through the end of each probe packet.
    varying = np.linalg.norm(a[0]-b[0], axis=1) > 1.e-8
    included = perturbed | varying
    return tuple(tuple(x[included] for x in s) for s in (a, b)), issued


def learn_nominal(output, *, extra=0, contrast_pairs=0):
    body, clips = archived_clips()
    models, audit, inverse_models = [], [], []
    source_probes = list(zip(BODY_SEEDS, body, strict=True))
    for index, (probe_clips, ordinary) in enumerate(clips):
        arm = ('stock', 'alternate')[index]
        contrasts, contrast_frames = (paired_responses(arm, REGIMES[index], contrast_pairs)
                                      if contrast_pairs else (None, 0))
        supplemental = body if not index else []
        if index and extra:
            arena = BallFree('alternate', 120)
            supplemental = []
            try:
                for seed, (_, actions) in source_probes[:extra]:
                    obs = arena.reset(seed=seed, split='held', regime='changed', render=False)
                    nom = [sensor(obs)]
                    for command in actions[:arena.max_steps]:
                        obs, _, _ = arena.step(command, render=False)
                        nom.append(sensor(obs))
                    supplemental.append((np.asarray(nom), actions[:arena.max_steps]))
            finally:
                arena.close()
        training = supplemental + probe_clips * 2
        all_rms = []
        for i in range(len(ordinary)):
            fold = Forward(*examples(training + 2*(ordinary[:i]+ordinary[i+1:])), contrasts=contrasts)
            residual = fold.residuals(*examples(ordinary[i:i+1]))
            all_rms.append(np.sqrt(np.mean(residual**2, axis=0)))
        model = Forward(*examples(training + 2*ordinary), contrasts=contrasts)
        model.noise = np.maximum(model.noise, np.median(all_rms, axis=0))
        models.append(model)
        inverse_models.append(MotorObserver(body + 3*probe_clips + 3*ordinary))
        audit.append(dict(arm=arm, archived_game_frames=sum(len(y) for _, y in ordinary),
                          archived_probe_frames=sum(len(y) for _, y in probe_clips), forward_ball_free_frames=
                          sum(len(y) for _, y in supplemental), new_ball_free_frames=120*extra*index,
                          new_contrast_pairs=contrast_pairs, new_contrast_frames=contrast_frames,
                          held_game_fold_rms=np.asarray(all_rms).tolist(), fitted_motion_rms=model.noise.tolist(),
                          response_current_sv=np.linalg.svd(model.response[0], compute_uv=False).tolist()))
        print('nominal', audit[-1], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{f'{i}_{key}': value for i, model in enumerate(models)
                                  for key, value in model.fields().items()})
    return models, inverse_models, audit


def load_forward(path=CALIBRATION):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(Forward.restore({k[2:]: archive[k] for k in archive.files if k.startswith(f'{i}_')})
                     for i in range(2))


def whiten(auto, matrix, before, after, noise):
    return matrix/noise[:, None], (after[5:]-before[5:]-auto)/noise


def ridge_fit(rows, prior=IDENTITY, penalty=3.0):
    if not rows:
        return prior.copy()
    h, r = (np.concatenate(a) for a in zip(*rows, strict=True))
    direction = h.T @ r
    basis = h.T @ h
    solution = np.linalg.solve(basis + penalty*np.eye(N_MOTORS**2),
                               direction+penalty*prior.ravel())
    return solution.reshape(N_MOTORS, N_MOTORS)


def rating(row, guesses):
    matrix, residual = row
    maps = np.asarray(guesses).reshape(-1, N_MOTORS**2)
    return np.abs(residual - maps @ matrix.T)


def clipped_error(error):
    return float(np.sqrt(np.mean(np.square(np.minimum(error, 9.)))))


def inverse(guess, desired):
    # Near-singular guesses are cheap to produce before all directions have been seen.
    u, s, v = np.linalg.svd(guess)
    regularized = (v.T * (s/(s*s+.035))) @ u.T
    return np.clip(regularized @ desired, -.82, .82)


def probe_batch(rng, budget, *, scale=.32):
    packets = []
    while len(packets) < budget:
        basis, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        for vector in basis.T:
            packets.extend([scale*vector, scale*vector, -scale*vector, -scale*vector])
    return np.asarray(packets[:budget])


def conduct(arena, seed, regime, changes, wiring, count):
    body, commands = [sensor(arena.reset(seed=seed, split='held', regime=regime, render=False))], []
    for command in changes[:count]:
        # A dense hidden interface is applied after the learner's request.
        obs, _, _ = arena.step(wiring @ command, render=False)
        commands.append(command)
        body.append(sensor(obs))
    return np.stack(body), np.asarray(commands)


def identify(arena, example_seed, regime, wiring, forward, observer, requests, test):
    states, commands = conduct(arena, example_seed, regime, requests, wiring, max(BUDGET))
    ts, tu = conduct(arena, example_seed, regime, test, wiring, len(test))
    transitions = [whiten(*forward.predict(ts[:i+1], tu[:i+1]), ts[i], ts[i+1], forward.noise)
                   for i in range(len(tu))]
    measured, legacy, proof = [], MapFit(.993), {}
    for i in range(len(commands)):
        channel = whiten(*forward.predict(states[:i+1], commands[:i+1]), states[i], states[i+1], forward.noise)
        measured.append(channel)
        if i+1 >= max(TAPS):
            actual = observer.estimate(states[:i+2])[:5]
            age = i+1-DELAY
            move = commands[age-WINDOW:age].sum(axis=0)
            legacy.update(move, actual)
        if i+1 in BUDGET:
            freshest = ridge_fit(measured, penalty=2.0)
            candidates = np.stack((IDENTITY, legacy.transform, freshest))
            actual = np.concatenate([rating(row, candidates) for row in transitions], axis=1)
            proof[i+1] = {mode: dict(relative_wiring_error=float(np.linalg.norm(candidate-wiring)/np.linalg.norm(wiring)),
                                     held_prediction_rms=clipped_error(error))
                          for mode, candidate, error in zip(('identity','inverse_observer','forward'), candidates, actual,
                                                            strict=True)}
    return proof


def qualification(models, inverse_models, bodies, seed=BODY_SEED):
    result = []
    generator = np.random.default_rng(seed)
    plants = [BallFree(arm, max(BUDGET)) for arm in ('stock', 'alternate')]
    try:
        for i in range(bodies):
            seed = int(generator.integers(2**31-1))
            wiring = transmission(generator)
            train = probe_batch(generator, max(BUDGET))
            test = probe_batch(generator, 40, scale=.40)
            row = {}
            for index, arena in enumerate(plants):
                row[arena.arm] = identify(arena, seed, REGIMES[index], wiring, models[index], inverse_models[index],
                                          train, test)
            result.append(dict(seed=seed, wiring=wiring.tolist(), body=row))
    finally:
        for arena in plants:
            arena.close()
    aggregate = {}
    for body in ('stock', 'alternate'):
        aggregate[body] = {n: {mode: {m: float(np.mean([r['body'][body][n][mode][m] for r in result]))
                                      for m in ('relative_wiring_error', 'held_prediction_rms')}
                               for mode in ('identity', 'inverse_observer', 'forward')} for n in BUDGET}
    print('unseen nominal-free identification', aggregate, flush=True)
    return result, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPORT)
    parser.add_argument('--model', type=Path, default=CALIBRATION)
    parser.add_argument('--bodies', type=int, default=BODY_DROPS)
    parser.add_argument('--alternate-probe-replays', type=int, default=0, choices=range(0, 9))
    parser.add_argument('--contrast-pairs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=BODY_SEED)
    args = parser.parse_args()
    models, legacy, charges = learn_nominal(args.model, extra=args.alternate_probe_replays,
                                            contrast_pairs=args.contrast_pairs)
    rows, summary = qualification(models, legacy, args.bodies, args.seed)
    _write(args.out, dict(schema='tt24-forward-wiring-identification-1', nominal=charges, summary=summary, rows=rows,
                          source='own five action commands and causal encoder histories', target_reward_updates=0,
                          model=str(args.model), horizons=BUDGET))


if __name__ == '__main__':
    main()
