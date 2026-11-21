"""Propose serves in a separate factory model with independent launch samples."""
from dataclasses import dataclass

from mujoco import rollout
import numpy as np
from scipy.stats import qmc

from .tt4_arena import BALL_INERTIA, BALL_MASS, MAGNUS_COEFFICIENT, SPIN_DECAY, mujoco
from .tt19_factory import FactoryModel, HOME, RATE
from .tt19_model_rollout import DrySimulator, ORIGIN


LOOK = 51
SHOTS = 192
DESIGN_SEED = 175840113
WORLD_SEED = 395633829


@dataclass(frozen=True)
class Toss:
    center: np.ndarray
    velocity: np.ndarray

    def state(self):
        return dict(center=self.center.tolist(), velocity=self.velocity.tolist())


def tosses():
    """Generate an independent, wider toss prior around the public endline."""
    rng = np.random.default_rng(WORLD_SEED)
    yield Toss(np.array([0., -.900, .344]), np.array([0., 0., 2.84]))
    for x in (-.070, -.040, -.011, .021, .048, .072, .002):
        y, z, up = rng.uniform((-.929, .316, 2.67), (-.863, .383, 3.035))
        vx, vy = rng.uniform((-.057, -.022), (.057, .022))
        yield Toss(np.array([x, y, z]), np.array([vx, vy, up]))


def proposals(count, *, geometry=None):
    """Convert public tool path goals into new two-stage requested motions."""
    geometry = FactoryModel() if geometry is None else geometry
    sequence = qmc.Sobol(10, scramble=True, seed=DESIGN_SEED).random_base2(
        int(np.ceil(np.log2(count))))[:count]
    for u in sequence:
        height = .148 + .215 * u[0]
        depth = -.900 + .045 * (u[1] - .5)
        pitch = -.08 + .64 * u[2]
        normal = np.array([0., np.cos(pitch), np.sin(pitch)])
        impact = np.array([0., depth, height]) - .026 * normal
        # A rapid sweep lasts a few control frames. Starting ahead of the
        # expected contact accommodates motor and command settling.
        direction = np.array([0., 1., -.67 + .98 * u[3]])
        backup = .035 + .118 * u[4]
        finish = .036 + .125 * u[5]
        pre = geometry.solve(impact - backup * direction, normal, HOME)
        post = geometry.solve(impact + finish * direction, normal, pre)
        start = (2.84 + np.sqrt(2.84**2 + 2 * 9.81 * (.344 - height))) * (30 / 9.81)
        duration = int(2.5 + 4.9 * u[6])
        onset = start - (backup / (backup + finish)) * duration - .1 - 4.8 * u[7]
        base = (pre - HOME) / RATE[:5]
        delta = (post - pre) / RATE[:5]
        base[3] += (u[8] - .5) * 1.8
        delta[3] += (u[9] - .5) * 2.2
        _, _, jacobian, _ = geometry.pose(pre)
        lateral = float(np.clip(1. / max(.035, jacobian[0, 0] * RATE[0]), 10., 24.))
        yield np.r_[np.clip(base, -5.6, 5.6), np.clip(delta, -6.2, 6.2),
                    np.clip(onset, 5., 21.), duration, lateral]


def actions(programs, initial_x, vx):
    """Mirror the public stroke program using the independent toss trajectory."""
    programs = np.asarray(programs)
    onset = np.ceil(programs[:, 10]).astype(int)
    duration = np.rint(programs[:, 11]).astype(int)
    movement = np.zeros((len(programs), 5))
    tapes = np.zeros((len(programs), LOOK, 5))
    for tick in range(4, LOOK):
        ramp = np.clip((tick - onset + 1) / duration, 0., 1.)
        back = np.clip(1. - (tick - onset - duration - 3) / duration, 0., 1.)
        goal = programs[:, :5] + (ramp * back)[:, None] * programs[:, 5:10]
        last = min(tick - tick % 2, 10)
        goal[:, 0] += (initial_x + vx * last / 30.) * programs[:, 12]
        tape = np.clip(goal - movement, -.85, .85)
        tapes[:, tick] = tape
        movement += tape
    return tapes


def _evaluate_paths(positions, velocities):
    result = []
    previous, current = positions[:, :-1], positions[:, 1:]
    vin, vout = velocities[:, :-1], velocities[:, 1:]
    push = (vout[..., 1] - vin[..., 1] > .18) & (vout[..., 1] > .24)
    descending = (vin[..., 2] < -.25) & (previous[..., 1] < -.80) & (previous[..., 2] > .028)
    leap = (vin[..., 2] < -.04) & (vout[..., 2] > .10) & (previous[..., 2] < .033)
    leap &= (abs(previous[..., 0]) < .433) & (abs(previous[..., 1]) < .809)
    for index, events in enumerate(push & descending):
        contacts = np.flatnonzero(events)
        if not len(contacts):
            result.append([False, False, 0.])
            continue
        strike = int(contacts[0])
        bounces = np.flatnonzero(leap[index, strike + 1:]) + strike + 1
        crossings = np.flatnonzero((previous[index, strike:, 1] < 0.) & (current[index, strike:, 1] >= 0.)) + strike
        clears = bool(len(crossings) and previous[index, crossings[0], 2] > .163)
        own = bool(len(bounces) and previous[index, bounces[0], 1] < -.012)
        legal = bool(own and len(bounces) >= 2 and clears and previous[index, bounces[1], 1] > .015)
        score = .25 + .25 * own + .24 * clears + 1.8 * legal
        if legal:
            x, y = previous[index, bounces[1], :2]
            score += .34 * np.clip(1. - abs(y - .58), 0., 1.) + .25 * np.clip(abs(x) / .38, 0., 1.)
        result.append([True, legal, float(score)])
    return result


class ServeSimulator(DrySimulator):
    """Roll out the public nominal plant with the independently generated toss."""

    def score(self, programs, toss):
        tape = actions(programs, toss.center[0], toss.velocity[0])
        batch = len(tape)
        actual = np.broadcast_to(self.base, (batch, len(self.base))).copy()
        actual[:, self.ball_p:self.ball_p + 3] = toss.center + ORIGIN
        actual[:, self.ball_p + 3:self.ball_p + 7] = (1., 0., 0., 0.)
        actual[:, self.ball_v:self.ball_v + 3] = toss.velocity
        actual[:, self.ball_v + 3:self.ball_v + 6] = 0.
        controls = np.zeros((batch, self.timestep, self.n + 6 * self.model.nbody))
        micros = np.arange(1, self.timestep + 1)[None, :, None]
        drive = np.broadcast_to(self.home, (batch, self.n)).copy()
        positions, velocities = [actual[:, None, self.ball_p:self.ball_p + 3] - ORIGIN], [
            actual[:, None, self.ball_v:self.ball_v + 3]]
        half = self.timestep // 2
        for tick in range(LOOK):
            target = drive[:, None, self.axes] + micros * (tape[:, tick] * self.increments)[:, None, :]
            target = np.clip(target, self.low, self.high)
            controls[:, :, :self.n] = self.home
            controls[:, :, self.axes] = target
            drive[:, self.axes] = target[:, -1]
            for start in (0, half):
                quaternion = actual[:, self.ball_p + 3:self.ball_p + 7]
                local = actual[:, self.ball_v + 3:self.ball_v + 6]
                axis = np.cross(quaternion[:, 1:], local)
                spin = local + 2 * (quaternion[:, :1] * axis + np.cross(quaternion[:, 1:], axis))
                velocity = actual[:, self.ball_v:self.ball_v + 3]
                force = -BALL_MASS * self.drag * np.linalg.norm(velocity, axis=1, keepdims=True) * velocity
                force += BALL_MASS * MAGNUS_COEFFICIENT * np.cross(spin, velocity)
                controls[:, start:start + half, self.force_id:self.force_id + 3] = force[:, None, :]
                controls[:, start:start + half, self.force_id + 3:self.force_id + 6] = (-BALL_INERTIA * SPIN_DECAY * spin)[:, None, :]
                states, _ = rollout.rollout(self.model, self.data, actual,
                                            np.ascontiguousarray(controls[:, start:start + half]),
                                            control_spec=self.controls)
                actual = np.ascontiguousarray(states[:, -1])
                positions.append(states[..., self.ball_p:self.ball_p + 3] - ORIGIN)
                velocities.append(states[..., self.ball_v:self.ball_v + 3])
        return _evaluate_paths(np.concatenate(positions, axis=1), np.concatenate(velocities, axis=1))


def _score(simulator, programs, sample):
    return [r for left in range(0, len(programs), 24) for r in simulator.score(programs[left:left+24], sample)]


def propose(simulator, count=SHOTS, *, geometry=None):
    scenarios = list(tosses())
    programs = np.asarray(list(proposals(count, geometry=geometry)))
    nominal = np.asarray(_score(simulator, programs, scenarios[0]))
    seed = np.argsort(-nominal[:, 2], kind='stable')[:12]
    print('factory central toss', len(programs), int(nominal[:, 0].sum()), int(nominal[:, 1].sum()), flush=True)
    rnd = np.random.default_rng(DESIGN_SEED + 19)
    scale = np.r_[np.full(5, .30), np.full(5, .55), 1.4, .64, 1.3]
    around = (programs[seed[:6], None, :] + rnd.normal(size=(len(seed[:6]), 6, 13)) * scale).reshape(-1, 13)
    around[:, 10] = np.clip(around[:, 10], 4., 22.)
    around[:, 11] = np.clip(np.round(around[:, 11]), 2., 8.)
    variants = np.vstack((programs[seed], around))
    checks = np.array([_score(simulator, variants, sample) for sample in scenarios[::2]])
    best = np.argsort(-checks[:, :, 2].mean(axis=0), kind='stable')[:12]
    remaining = np.array([_score(simulator, variants[best], sample) for sample in scenarios[1::2]])
    performance = np.concatenate((checks[:, best], remaining), axis=0)
    order = np.argsort(-performance[:, :, 2].mean(axis=0), kind='stable')
    # Keep four genuinely distinct candidates so that twelve target trials
    # can compare alternatives instead of testing one local perturbation.
    chosen = []
    for index in order:
        if all(np.linalg.norm(variants[best[index]] - variants[best[prior]]) >= .73 for prior in chosen):
            chosen.append(int(index))
        if len(chosen) == 4:
            break
    chosen.extend(int(index) for index in order if int(index) not in chosen and len(chosen) < 4)
    return dict(programs=variants[best[chosen]].tolist(), outcomes=performance[:, chosen].tolist(),
                scenarios=[sample.state() for sample in scenarios[::2] + scenarios[1::2]],
                central_results=nominal.tolist(), geometry_proposals=len(programs), local_proposals=len(around),
                simulation_tapes=len(programs) + len(variants) * 4 + len(best) * 4, ranked_candidates=best[chosen].tolist())
