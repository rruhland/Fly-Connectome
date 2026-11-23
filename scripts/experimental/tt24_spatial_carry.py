"""Measure local racket-frame transition overlap between live 3D rallies and paddle bouncing."""

import json
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import BLADE
from .tt23_sequence import CONFIG, SequenceArena
from .tt24_bounce import BounceArena, mujoco
from .tt24_bounce_trial import play, verify_replay
from .tt24_recurrence_arm import EMBODIMENT, target_effects
from .tt24_rally_trial import OUTPUT
from .tt24_rally_trial import returns as native_returns


ARCHIVES = (('rally-improve-combined-held.json.gz', 'combined'), ('rally-improve-optical-held.json.gz', 'interaction'))
SOURCE = OUTPUT / 'spatial-carry-table.json.gz'
RESULT = OUTPUT / 'spatial-carry-contact.json.gz'
SCHEDULE = 867501104
TARGET_COUNT = 12
AXES = ('tangent-x', 'tangent-y', 'normal')
FEATURES = ('incoming-x', 'incoming-y', 'normal-closure', 'contact-x', 'contact-y', 'point-motion-x',
            'point-motion-y', 'point-motion-normal', 'radial-distance')


def capture_impact(model, data, geom, incoming, location, ball, contact_axis):
    """Clean diagnostic only. Resolve the face pointing at the approach."""
    rotation = data.geom_xmat[geom].reshape(3, 3).copy()
    center = data.geom_xpos[geom].copy()
    if abs(contact_axis @ rotation[:, 2]) < .94:
        return None
    if (ball - center) @ rotation[:, 2] < 0.:
        rotation[:, 1:] *= -1.
    body_twist = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, body_twist, 0)
    offset = np.asarray(location) - center
    point_velocity = body_twist[3:] + np.cross(body_twist[:3], offset)
    relative = rotation.T @ (np.asarray(incoming) - point_velocity)
    closure = -relative[2]
    if closure < .2:
        return None
    local = rotation.T @ offset / model.geom_size[geom, 0]
    motion = rotation.T @ point_velocity
    return dict(features=np.r_[relative[:2], closure, local[:2], motion, np.linalg.norm(local[:2])].tolist(),
                rotation=rotation.tolist(), contact_point_velocity=point_velocity.tolist(),
                incoming=np.asarray(incoming).tolist(), centered_contact=local.tolist(), time=float(data.time))


def capture_out(impact, velocity, time):
    rotation = np.asarray(impact['rotation'])
    out = rotation.T @ (np.asarray(velocity) - impact['contact_point_velocity'])
    return dict(impact, outgoing=out.tolist(), elapsed=float(time - impact['time']))


def racket_contact(data, ball_id, racket_id):
    for item in data.contact:
        if item.efc_address >= 0 and set(item.geom) == {ball_id, racket_id}:
            return item
    return None


class TracedMatch(SequenceArena):
    """Capture body coordinates by exact numeric replay of archived live rallies."""

    def arm_capture(self):
        self.impact = None
        self.measured = []

    def _contacts(self, incoming):
        before = len(self.events)
        super()._contacts(incoming)
        for row in self.events[before:]:
            if row['kind'] == 'contact' and row['side'] == 1:
                geom = self._racket_ids[1]
                contact = racket_contact(self.data, self._geom_ids['ball'], geom)
                if contact is not None:
                    self.impact = capture_impact(self.model, self.data, geom, incoming, row['location'],
                                                 np.asarray(row['ball']), contact.frame[:3])
                    if self.impact is not None:
                        self.impact['tick'] = row['step']
            elif row['kind'] == 'separation' and row.get('collider') == BLADE and self.impact is not None:
                measured = capture_out(self.impact, row['velocity'], row['time'])
                self.measured.append(measured)
                self.impact = None


class TracedBounce(BounceArena):
    """Clean diagnostic. The live wrist anchor remains outside the geometry observer."""

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.previous = self._ball_velocity.copy()
        self.measured = []
        self.impact = None
        return result

    def _judge_substep(self, incoming):
        contact = racket_contact(self._data, self._ball_geom, self._blade)
        if contact is not None and not self._touching:
            self.impact = capture_impact(self._model, self._data, self._blade, self.previous, contact.pos,
                                         self._ball_position, contact.frame[:3])
        if contact is None and self._touching and self.impact is not None:
            self.measured.append(capture_out(self.impact, self._ball_velocity, self._data.time))
            self.impact = None
        self.previous[:] = self._ball_velocity
        return super()._judge_substep(incoming)


def table_setup():
    seed_source = []
    for path, mode in ARCHIVES:
        archive = _read(OUTPUT / path)
        candidates = [r['results'][mode] for r in archive['cases'] if r and len(native_returns(r['results'][mode], 1)) >= 2]
        seed_source.extend((row, path, mode) for row in candidates)
    seeds = [row['seed'] for row, _, _ in seed_source]
    if len(seeds) != len(set(seeds)):
        raise ValueError('source points overlap by release')
    return seed_source


def source_rows():
    if SOURCE.exists():
        data = _read(SOURCE)
        if data['archives'] != {name: _hash(OUTPUT / name) for name, _ in ARCHIVES}:
            raise ValueError('table source changed after capture')
        return data
    arena = TracedMatch(**CONFIG)
    source = []
    try:
        for row, filename, mode in table_setup():
            arena.lane = row['lane']
            arena.reset(seed=row['seed'], server=-1, render=False)
            arena.arm_capture()
            for i, (command, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True), 1):
                rates = np.asarray(far)
                arena._aim = arena._command_to = lambda *_args, rates=rates: rates
                _, reward, done, cap, _ = arena.step(command, render=False)
                if bool(done or cap) != (i == len(row['actions'])):
                    raise ValueError('native table contact replay changed the point duration')
            if json.dumps(arena.events, sort_keys=True) != json.dumps(row['events'], sort_keys=True):
                raise ValueError('native table contact replay changed a referee event')
            if reward != row['reward'] or arena.point_result() != row['outcome']:
                raise ValueError('native table contact replay changed the terminal scoring')
            retained = []
            for effect in arena.measured:
                next_blade = next((e for e in arena.events if e['kind'] == 'contact' and e['time'] > effect['time']), None)
                stop = np.inf if next_blade is None else next_blade['time']
                legal = any(e['kind'] == 'legal_return' and e.get('side') == 1 and effect['time'] < e['time'] < stop
                            for e in arena.events)
                retained.append(dict(effect, legal=legal))
            source.append(dict(seed=row['seed'], tape=filename, mode=mode, clean=retained,
                               real_near_returns=len(native_returns(row, 1)), steps=len(row['actions'])))
            print('native', row['seed'], len(retained), sum(e['legal'] for e in retained), flush=True)
    finally:
        arena.close()
    if not source or not any(entry['clean'] for entry in source):
        raise ValueError('no physically valid native racket encounters were recorded')
    data = dict(schema='tt24-spatial-carry-table-1', points=source,
                archives={name: _hash(OUTPUT / name) for name, _ in ARCHIVES}, verified_points=len(source))
    _write(SOURCE, data)
    return data


def components(groups, *, legal=False):
    records = [row for group in groups for row in group['clean'] if not legal or row['legal']]
    return tuple(np.array([row[col] for row in records]) for col in ('features', 'outgoing'))


def reflect(x):
    return x[:, :3].copy()


def motor_design(x, state):
    x = np.clip((x - state['center']) / state['spread'], -4., 4.)
    return np.column_stack((np.ones(len(x)), x, x[:, 2:3] * x[:, :2], x[:, 2:3] * x[:, 3:5]))


def fit(x, y):
    state = dict(center=np.median(x, axis=0).tolist(), spread=np.maximum(x.std(axis=0), .06).tolist())
    design = motor_design(x, state)
    regularization = design.T @ design + np.eye(design.shape[1]) * 6.
    delta = y - reflect(x)
    rng = np.random.default_rng(345850059)
    state['weights'] = {mode: np.linalg.solve(regularization, design.T @ residual).tolist()
                        for mode, residual in (('coherent', delta), ('shuffled', delta[rng.permutation(len(delta))]))}
    state['coverage'] = dict(low=np.quantile(x, .05, axis=0).tolist(), high=np.quantile(x, .95, axis=0).tolist())
    return state


def predict(x, checkpoint, mode):
    reflected = reflect(x)
    if mode == 'cold':
        return reflected
    return reflected + motor_design(x, checkpoint) @ checkpoint['weights'][mode]


def report_on(x, y, state):
    inside = (x >= state['coverage']['low']) & (x <= state['coverage']['high'])
    clipped = np.abs((x - state['center']) / state['spread'])
    outputs = {}
    for mode in ('coherent', 'shuffled', 'cold'):
        error = predict(x, state, mode) - y
        outputs[mode] = dict(rmse=float(np.mean(error**2)**.5), normal_rmse=float(np.mean(error[:, 2]**2)**.5),
                             tangent_rmse=float(np.mean(error[:, :2]**2)**.5), bias=error.mean(axis=0).tolist())
    return dict(rows=len(x), all_inside=float(np.mean(inside.all(axis=1))),
                inside_per_feature=np.mean(inside, axis=0).tolist(),
                severe_outliers=float(np.mean(np.any(clipped > 3.5, axis=1))),
                mean=x.mean(axis=0).tolist(), median=np.median(x, axis=0).tolist(),
                outbound_median=np.median(y, axis=0).tolist(), modes=outputs)


def normal_features(x):
    closure = x[:, 2]
    return np.column_stack((np.log(closure), np.linalg.norm(x[:, :2], axis=1) / closure, x[:, -1]))


def normal_contact_model(x, y):
    features = normal_features(x)
    measured = y[:, 2] / x[:, 2]
    median, spread = np.median(features, axis=0), np.maximum(.07, features.std(axis=0))
    a = np.clip((features - median) / spread, -3., 3.)
    design = np.column_stack((np.ones(len(x)), a))
    reg = design.T @ design + np.diag((.03, 12., 12., 12.))
    shuffled = np.random.default_rng(345850059).permutation(measured)
    params = {mode: np.linalg.solve(reg, design.T @ score).tolist()
              for mode, score in (('coherent', measured), ('shuffled', shuffled))}
    return dict(median=median.tolist(), spread=spread.tolist(), weights=params, constant=float(measured.mean()),
                low=np.quantile(features, .05, axis=0).tolist(), high=np.quantile(features, .95, axis=0).tolist())


def normal_diagnostic(x, y, model):
    features = normal_features(x)
    all_inside = (features >= model['low']).all(axis=1) & (features <= model['high']).all(axis=1)
    center = np.clip((features - model['median']) / model['spread'], -3., 3.)
    design = np.column_stack((np.ones(len(x)), center))
    prediction = {mode: design @ weights for mode, weights in model['weights'].items()}
    prediction |= {'constant_source': model['constant'], 'cold': 1.}
    closure = x[:, 2]
    errors = {mode: float(np.mean((closure * values - y[:, 2])**2)**.5)
              for mode, values in prediction.items()}
    return dict(rows=len(x), support=float(all_inside.mean()), rmse=errors,
                median_closure=float(np.median(closure)), median_slip_ratio=float(np.median(features[:, 1])),
                median_restitution=float(np.median(y[:, 2] / closure)))


def target_rows():
    rng = np.random.default_rng(SCHEDULE)
    seeds = rng.choice(2147483646, TARGET_COUNT, replace=False).tolist()
    body = _read(EMBODIMENT)
    known = set(body['practice_seeds']) | {row['seed'] for branches in body['held'].values() for row in branches}
    if known.intersection(seeds):
        raise ValueError('target diagnostic overlaps the borrowed wrist policy exposure')
    weights = np.asarray(body['weights'])
    arena = TracedBounce(view='wrist', width=body['width'], height=body['height'])
    response = {}
    try:
        for mode in ('passive', 'active'):
            rows = []
            for seed in seeds:
                result = play(arena, seed, weights, target_effects(body, arena.action_size), split='held',
                              stride=body['stride'], readout=body['readout'], retinal=body['retinal'],
                              keep_actions=True, hold=mode == 'passive', blind=mode == 'passive')
                result['clean'] = arena.measured
                rows.append(result)
            response[mode] = rows
        for mode in response.values():
            for case in mode:
                verify_replay(arena, case)
    finally:
        arena.close()
    return response


def main():
    started = perf_counter()
    source = source_rows()
    cases = source['points']
    train = [row for i, row in enumerate(cases) if i % 3]
    held = [row for i, row in enumerate(cases) if not i % 3]
    x, y = components(train, legal=True)
    state = fit(x, y)
    native = report_on(*components(held, legal=True), state)
    print('held native contact law', native, flush=True)
    previous = _read(RESULT) if RESULT.exists() else None
    if previous is not None and previous['source_sha256'] != _hash(SOURCE):
        raise ValueError('previous target diagnostic uses a different source archive')
    target = target_rows() if previous is None else previous['target']
    reduced = normal_contact_model(x, y)
    elapsed = perf_counter() - started if previous is None else previous['seconds']
    result = dict(schema='tt24-spatial-carry-contact-1', source_sha256=_hash(SOURCE),
                  source_observations=len(x), features=FEATURES, goal_axes=AXES,
                  source_practice=[r['seed'] for r in train], source_held=[r['seed'] for r in held], model=state,
                  native_held=native, target=target, target_train_attempts=0,
                  anchor_sha256=_hash(EMBODIMENT), anchor_prior_attempts=_read(EMBODIMENT)['total_initial_attempts'],
                  verified_target_tapes=2 * TARGET_COUNT, seconds=elapsed,
                  normal_model=reduced)
    result['target_diagnostic'] = {mode: report_on(*components(rows), state) for mode, rows in target.items()}
    result['normal_diagnostic'] = {'native': normal_diagnostic(*components(held, legal=True), reduced)} | {
        mode: normal_diagnostic(*components(rows), reduced) for mode, rows in target.items()}
    _write(RESULT, result)
    print('target clean overlap', result['target_diagnostic'], flush=True)
    print('orientation-invariant normal contact', result['normal_diagnostic'], flush=True)


if __name__ == '__main__':
    main()
