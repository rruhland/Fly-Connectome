"""Carry a reward-trained bounce motor head into a new one-flight partner return."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena, mujoco
from .tt24_bounce_ident import CleanState, _ellipse
from .tt24_donor_trial import ROOT, SHAPE, STATE, _checkpoint, _live
from .tt24_metric_stereo import triangulate
from .tt24_predict_reference import TrainingEyes
from .tt24_public_body import FUSED_OUTPUT, load_public, stationary
from .tt24_volley_source import CONTROL_LIMIT, STATS, VolleyNative, _feed


DONOR = ROOT / 'tt24_donor-refined.npz'
DONOR_REPORT = ROOT / 'tt24_donor-refined.json.gz'
SEED = 307671122
PHASES = 4
COHORTS = 2
EVAL_SIZE = 20
RETENTION_SIZE = 12
RADII = (.43, .43, .36, .36)


class PartnerCourt(BounceArena):
    """Same stock physics and first-flight judge used by the native volley source."""

    def __init__(self, catchable=False):
        self.catchable = catchable
        self.gate_height = .118 if catchable else .165
        super().__init__(view='wrist', width=192, height=144, max_steps=CONTROL_LIMIT)

    fixed_images = TrainingEyes.fixed_images

    def reset(self, *, seed=None, render=True):
        super().reset(seed=seed, split='held', regime='standard', render=False)
        offset, velocity = _feed(int(seed), self.catchable)
        self.origin = self._data.geom_xpos[self._blade].copy()
        self.side = np.sign(offset[0])
        self._ball_position[:] = self.origin + offset
        self._ball_velocity[:] = velocity
        mujoco.mj_forward(self._model, self._data)
        self.previous = self._ball_position.copy()
        self.touching, self.sent, self.legal, self.crossing = False, False, False, False
        self.route = 0
        self.maximum, self.surface_speed, self.contact_time = 0., 0., 0.
        self.finish = np.zeros(3)
        self.crossing_vz = 0.
        self.encounters = 0
        self.decisions = 0
        return self._observation(render)

    def _contact_on_ball(self):
        own = self._ball_geom
        blade = None
        for contact in self._data.contact:
            if contact.efc_address < 0 or own not in contact.geom:
                continue
            first, second = contact.geom
            other, sign = (second, -1) if first == own else (first, 1)
            if other != self._blade:
                self._collider = self._model.geom(other).name
                self.route = 1
                self._end('non-paddle contact')
                return None
            blade = contact, sign
        return blade

    def _judge_substep(self, incoming):
        contact = self._contact_on_ball()
        if self._done:
            return 0.
        now = self._ball_position
        old = self.previous.copy()
        self.previous[:] = now
        speed = self._ball_velocity
        pose = self._data
        if contact is not None and self.sent:
            self.route, self.legal = 2, False
            self._end('early recontact')
            return 0.
        if contact is not None and not self.touching:
            self.encounters += 1
            hit, sign = contact
            normal = pose.geom_xmat[self._blade].reshape(3, 3)[:, 2]
            if sign * hit.frame[2] <= .55 or abs(hit.frame[:3] @ normal) <= .95 or incoming >= -.25:
                self.route = 3
                self._end('invalid encounter')
                return 0.
            self.touching = True
            self.contact_time = pose.time
            twist = np.empty(6)
            mujoco.mj_objectVelocity(self._model, pose, mujoco.mjtObj.mjOBJ_GEOM, self._blade, twist, 0)
            gap = hit.pos - pose.geom_xpos[self._blade]
            self.surface_speed = float(twist[5] + twist[0]*gap[1] - twist[1]*gap[0])
        if self.touching and contact is None:
            self.touching = False
            if pose.time - self.contact_time > .032 or speed[2] <= .25:
                self.route = 4
                self._end('invalid departure')
                return 0.
            self.sent = self.legal = True
        if self.sent:
            self.finish[:] = np.r_[now[:2]-self.origin[:2], speed[0]]
            self.maximum = max(self.maximum, now[2]-self.origin[2])
            plane = self.origin[2] + self.gate_height
            if old[2] < plane <= now[2] and speed[2] > 0.:
                ratio = (plane-old[2])/(now[2]-old[2])
                self.finish[:2] = old[:2]+ratio*(now[:2]-old[:2])-self.origin[:2]
                self.crossing_vz = float(speed[2])
                self.route, self.crossing = 5, True
                self._end('crossing')
                return 0.
            if speed[2] <= 0.:
                self.route = 6
                self._end('apex')
                return 0.
        if now[2] < .51 or now[2] > 2.1 or abs(now[0]) > .66 or not -1.65 < now[1] < -.33:
            self.route = 7
            self._end('out of play')
        return 0.

    def stats(self):
        px, py, vx = self.finish
        side = (-1. if px < 0 else 1.) if self.catchable else self.side
        center, half_width = (.057, .043) if self.catchable else (.080, .055)
        dx, dy = (side*px-center)/half_width, py/.070
        gate = self.legal and self.crossing and abs(dx) <= 1 and abs(dy) <= 1 and side*vx >= .045
        if self.catchable:
            gate = gate and .16 <= self.crossing_vz <= 1.45
        powered = gate and self.surface_speed > .055
        aim = np.exp(-.35*(dx*dx+dy*dy))
        lift = np.clip(self.maximum/self.gate_height, 0., 1.)
        toward = np.clip((side*vx+.05)/.22, 0., 1.)
        power = np.clip(self.surface_speed/.22, 0., 1.)
        speed_quality = 1.
        if self.catchable and self.crossing:
            speed_quality = np.clip(1.-abs(self.crossing_vz-.7)/1.30, 0., 1.)
        shaped = .10+.33*lift+.45*aim*lift*speed_quality+.24*toward*lift*speed_quality+.27*power+gate+.5*powered
        score = float(shaped) if self.legal else 0.
        return np.array((score, gate, powered, self.decisions, self._steps, self.encounters,
                         self.route, self.maximum, vx, self.surface_speed, self.legal, px, py), dtype=float)


class StereoState:
    """Provide public metric flight tracking through two surveyed stationary cameras."""

    def __init__(self, arena, obs):
        self.arena = arena
        if not hasattr(arena, 'eyes'):
            arena.eyes = []
            arena._model.vis.global_.fovy = 31.
            for yaw, tilt in ((10., -26.), (100., -25.)):
                eye = mujoco.MjvCamera()
                eye.type = mujoco.mjtCamera.mjCAMERA_FREE
                eye.lookat[:] = (0., -.93, 1.03)
                eye.distance, eye.azimuth, eye.elevation = 1.58, yaw, tilt
                arena.eyes.append(eye)
        self.eyes = stationary(arena)
        self.paddle = load_public(FUSED_OUTPUT).decoder
        self.home = self.paddle.body(obs['joints'])[12:15].copy()
        self.samples = []
        self.velocity = np.zeros(3)
        self.last = self.home + self.paddle.default
        self.frame = -1

    def observe(self, obs):
        self.frame += 1
        seen = [_ellipse(frame) for frame in self.arena.fixed_images()]
        ball = np.full(3, np.nan)
        if all(item is not None for item in seen):
            centres = np.asarray(seen)[None, :, :2]
            ball = triangulate(self.eyes, centres, tolerance=.028)[0][0]
        if np.isfinite(ball).all():
            self.samples.append((self.frame, ball))
            self.samples = self.samples[-5:]
            timing = (np.array([i for i, _ in self.samples]) - self.frame)/60.
            time = timing - timing.mean()
            if len(time) >= 2:
                points = np.asarray([p for _, p in self.samples])
                # Use a free-flight curvature correction in the optical
                # differentiator. Keeper decisions end at the impact.
                points[:, 2] += .5 * 9.81 * timing**2
                self.velocity = np.clip((time @ points)/(time @ time), [-3.,-3.,-5.], [3.,3.,5.])
            self.last = ball
        else:
            self.last = self.last + self.velocity/60.
        blade = self.paddle.body(obs['joints'])
        return np.r_[self.last-self.home, self.velocity, blade[12:15]-self.home, blade[15:17]]


def rollout(arena, native, seed, weights, *, mode='public', replay=None):
    public = mode == 'public'
    obs = arena.reset(seed=seed, render=public)
    if mode == 'stereo':
        observer = StereoState(arena, obs)
    else:
        observer = load_public(FUSED_OUTPUT) if public else CleanState(arena) if mode == 'clean' else None
    weights = np.ascontiguousarray(weights, dtype=float)
    issued, output = np.zeros(5), np.zeros(5)
    tape = []
    while True:
        open_ = not arena.touching and not arena.sent
        if open_:
            arena.decisions += 1
        if not open_:
            output.fill(0.)
        elif observer is not None:
            state = np.ascontiguousarray(observer.observe(obs), dtype=float)
            native.command(state.ctypes.data, weights.ctypes.data, issued.ctypes.data, output.ctypes.data)
        if replay is not None:
            output = np.asarray(replay[len(tape)])
        tape.append(output.tolist())
        obs, _, done = arena.step(output, render=public and open_)
        if replay is not None and done != (len(tape) == len(replay)):
            raise ValueError('volley replay changed the terminal frame')
        if done:
            break
    return dict(seed=seed, metrics=dict(zip(STATS, arena.stats().tolist(), strict=True)), actions=tape)


def verified(arena, native, seed, weights, *, mode='public'):
    run = rollout(arena, native, seed, weights, mode=mode)
    raw = rollout(arena, native, seed, weights, mode='replay', replay=run['actions'])
    if run != raw:
        raise ValueError('volley replay changed the referee evidence')
    return run


def parity(arena, native, seed, weights):
    """Independent privileged diagnostic of the two referees, outside policy fitting."""
    model = arena._model
    handle = native.open(model._address)
    try:
        arena.reset(seed=seed, render=False)
        state = np.empty(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, arena._data, state, STATE)
        physics = np.array([arena._drag_force_scale, arena._drop.damping])
        output = np.empty(len(STATS))
        native.volley(handle, weights.ctypes.data, 1, state.ctypes.data,
                       physics.ctypes.data, 1, CONTROL_LIMIT, output.ctypes.data)
        answer = rollout(arena, native, seed, weights, mode='clean')
        np.testing.assert_allclose(output, list(answer['metrics'].values()), atol=3.e-9, rtol=3.e-8)
        return answer['metrics']
    finally:
        native.close(handle)


def loads(catchable):
    tag = 'tt24_volley_source-two-receivers' if catchable else 'tt24_volley_source'
    target = ROOT / (tag + ('.npz' if catchable else '-frozen.npz'))
    receipt = ROOT / (tag + '.json.gz')
    heads = []
    for checkpoint, report in ((DONOR, DONOR_REPORT), (target, receipt)):
        doc = _read(report)
        if doc['frozen_sha256' if checkpoint == DONOR else 'frozen_weights_sha256'] != _hash(checkpoint):
            raise ValueError('source weight lineage changed')
        with np.load(checkpoint, allow_pickle=False) as archive:
            heads.append(np.ascontiguousarray(archive['weights'], dtype=float))
    return *heads, target


def register(path, reference, seed, catchable, stereo):
    rng = np.random.default_rng(seed)
    seeds = iter(rng.choice(2**31-2, 12+COHORTS*(PHASES+EVAL_SIZE)+RETENTION_SIZE, replace=False).tolist())
    feasibility = [next(seeds) for _ in range(12)]
    streams = [dict(training=[next(seeds) for _ in range(PHASES)], held=[next(seeds) for _ in range(EVAL_SIZE)])
               for _ in range(COHORTS)]
    primary = ('first legal upward flight entering either bounded-speed receiver'
               if catchable else 'first legal reversed-direction upward passage through the mirrored stationary partner aperture')
    base = dict(schema='tt24-donor-volley-plan-1', seed=seed, preflight=feasibility, streams=streams,
                retention=[next(seeds) for _ in range(RETENTION_SIZE)], source_hash=_hash(DONOR),
                volley_reference_hash=_hash(reference), observer_hash=_hash(FUSED_OUTPUT),
                phase_radii=list(RADII), physical_attempt_budget=2*PHASES, new_demonstrations=0,
                preflight_gate='Ready reference gets at least 4 of 12 wrist routes, exceeds stationary by at least 3',
                held_primary=primary,
                arena='stock SO-101 with sequence ending after one impact plus first airborne delivery',
                candidate_support='same four randomly sampled full-head directions for all initializations per stream')
    if stereo:
        base['task_observer'] = 'paired calibrated fixed RGB and public encoder paddle fit'
    if path.exists() and _read(path) != base:
        raise ValueError('source or qualification schedule changed')
    if not path.exists():
        _write(path, base)
    return base


def summary(rows):
    result = np.array([[item['metrics'][key] for key in STATS] for item in rows])
    return dict(attempts=len(rows), routes=int(result[:, 1].sum()), powered=int(result[:, 2].sum()),
                mean_reward=float(result[:, 0].mean()), legal_departures=int(result[:, 10].sum()),
                decisions=int(result[:, 3].sum()), physical_frames=int(result[:, 4].sum()))


def preflight(arena, native, plan, plan_path, output, repeated, prior, observation):
    seeds = plan['preflight']
    modes = [('volley_reference', prior, observation), ('stationary', np.zeros(SHAPE), 'hold')]
    rows = {name: [verified(arena, native, seed, w, mode=mode) for seed in seeds] for name, w, mode in modes}
    check = [parity(arena, native, seeds[i], w) for i, w in enumerate((prior, repeated, np.zeros(SHAPE)))]
    outcome = {name: summary(x) for name, x in rows.items()}
    good = outcome['volley_reference']['routes']
    idle = outcome['stationary']['routes']
    passed = good >= 4 and good-idle >= 3
    _write(output, dict(plan_sha256=_hash(plan_path), native_parity=check, summary=outcome, rows=rows, passed=passed))
    print('wrist first-volley feasibility', outcome, 'passed', passed, flush=True)
    return passed


def scramble_weights(w):
    rng = np.random.default_rng(17260711)
    return w[:, rng.permutation(w.shape[1])] * rng.choice((-1., 1.), w.shape[1])


def improve(arena, native, initial, directions, seeds, observation):
    weight = initial.copy()
    exposure = []
    for seed, direction, radius in zip(seeds, directions, RADII, strict=True):
        positive = verified(arena, native, seed, weight+radius*direction, mode=observation)
        negative = verified(arena, native, seed, weight-radius*direction, mode=observation)
        delta = positive['metrics']['reward']-negative['metrics']['reward']
        weight += .86 * radius * np.tanh(delta/.55) * direction
        exposure.append(dict(positive=positive, negative=negative, paired_difference=float(delta)))
    return weight, exposure


def adaptation(arena, native, plan, repeated, prior, observation):
    blank, scrambled = np.zeros(SHAPE), scramble_weights(repeated)
    outcomes = []
    final = []
    for i, group in enumerate(plan['streams']):
        rng = np.random.default_rng(plan['seed']+5627+17*i)
        raw = rng.normal(size=(PHASES, *SHAPE))
        directions = raw / np.linalg.norm(raw, axis=(1,2), keepdims=True)
        values = {}
        options = dict(bounce_init=repeated, zero_init=blank, scrambled_init=scrambled)
        for name, w in options.items():
            final_w, practice = improve(arena, native, w, directions, group['training'], observation)
            values[name] = dict(practice=practice, final_weights=final_w.tolist())
            print(i, name, 'practice', [round(r['paired_difference'], 3) for r in practice], flush=True)
        final.append(np.array(values['bounce_init']['final_weights']))
        frozen = dict(frozen_bounce=repeated, frozen_scrambled=scrambled, stationary=blank, ready_volley=prior)
        for name, w in frozen.items():
            values[name] = dict(practice=[], final_weights=w.tolist())
        for name, info in values.items():
            w = np.array(info['final_weights'])
            mode = 'hold' if name == 'stationary' else observation
            rows = [verified(arena, native, seed, w, mode=mode) for seed in group['held']]
            info.update(held=rows, summary=summary(rows))
            print(i, name, info['summary'], flush=True)
        outcomes.append(dict(stream=i, directions=directions.tolist(), arms=values))
    return outcomes, final


def back_to_bounce(native, seeds, reference, updates):
    arena = BounceArena(view='wrist', width=192, height=144)
    values = {}
    try:
        for name, w in [('recalled_bounce', reference), *[(f'volley_updated_{i}', v) for i,v in enumerate(updates)]]:
            rows = []
            for seed in seeds:
                recorded = _live(native, arena, w, seed, 'standard', 'public')
                replay = _live(native, arena, w, seed, 'standard', 'replay', tape=recorded['actions'])
                if recorded != replay:
                    raise ValueError('retention tape changed')
                rows.append(dict(seed=seed, **recorded))
            runs = [r['metrics']['longest_consecutive_rebounds'] for r in rows]
            values[name] = dict(mean_streak=float(np.mean(runs)), maximum=max(runs), streaks=runs, rows=rows)
            print('retention',name,runs,flush=True)
    finally:
        arena.close()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument('--two-receivers', action='store_true')
    profile.add_argument('--stereo', action='store_true')
    args = parser.parse_args()
    two = args.two_receivers
    base = 'donor-volley-stereo' if args.stereo else 'donor-volley-two-receivers' if two else 'donor-volley'
    observation = 'stereo' if args.stereo else 'public'
    plan_path, feasibility, results = (ROOT/f'{base}{suffix}'
                                       for suffix in ('-plan.json', '-preflight.json.gz', '.json.gz'))
    repeated, prior, reference = loads(two)
    native = VolleyNative(two, two)
    arena = PartnerCourt(two)
    try:
        seed = SEED+659273 if args.stereo else SEED+8153 if two else SEED
        plan = register(plan_path, reference, seed, two, args.stereo)
        if feasibility.exists():
            early = _read(feasibility)
            if early['plan_sha256'] != _hash(plan_path):
                raise ValueError('preflight and schedule differ')
            passed = early['passed']
        else:
            passed = preflight(arena, native, plan, plan_path, feasibility, repeated, prior, observation)
        if args.preflight_only or not passed:
            return
        outcomes, final = adaptation(arena, native, plan, repeated, prior, observation)
        for i, w in enumerate(final):
            _checkpoint(ROOT/f'{base}-adapted-{i}.npz', w)
        _write(results, dict(schema='tt24-donor-volley-1', plan_sha256=_hash(plan_path), feasibility_sha256=_hash(feasibility),
                            source_head_sha256_after=_hash(DONOR), physical_attempt_budget=2*PHASES, streams=outcomes))
        retention = back_to_bounce(native, plan['retention'], repeated, final)
        report = _read(results)
        report['retention'] = retention
        _write(results, report)
    finally:
        arena.close()
        native.workspace.cleanup()


if __name__ == '__main__':
    main()
