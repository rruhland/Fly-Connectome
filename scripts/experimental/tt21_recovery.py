"""Learn when to launch the next stroke after an optically observed exchange."""

import numpy as np

from .tt19_flight import optical_start
from .tt19_model_rollout import RolloutPlayer
from .tt19_shoot import COURT


# Map-table y threshold, extra settling ticks, and minimum optical readiness.
PROGRAMS = ((-.24, 5, None), (.15, 0, .050), (-.40, 2, .035))
NEAR_WINDOW = (PROGRAMS[0][0], -.15)
# Nominally a catalogue stroke can take about ten frames after the decision.
PREPARATION = 10


def time_to_reach(history, law):
    start = optical_start(history, law.gravity)
    if start is None or start[1][1] >= -.033:
        return None
    trajectory, _ = law.trajectory(*start, 18, bounds=COURT)
    reach = np.flatnonzero(trajectory[:, 1] < -.91)
    return int(reach[0]) if len(reach) else 19


def upward_run(history, start):
    """Find the next observed forward arc that starts by our racket."""
    segment = [(t, p) for t, p in history if t >= start and p is not None]
    for (a, p), (b, q), (c, r) in zip(segment, segment[1:], segment[2:]):
        if c-a <= 5 and -.93 < p[1] < -.65 and (q[1]-p[1])/(b-a) > .027 and (r[1]-q[1])/(c-b) > .027:
            return c
    return None


def visual_merit(history, launch, law):
    """Estimate net passage and landing from our own outgoing ball observations."""
    if launch is None:
        return 0.
    after = [(t, np.asarray(p)) for t, p in history if p is not None and t >= launch]
    arc = upward_run(after, launch)
    if arc is None:
        return 0.
    credit = .08
    for index, (t, p) in enumerate(after):
        if t < arc or p[1] < -.64:
            continue
        estimate = optical_start(after[:index+1], law.gravity)
        if estimate is None or estimate[1][1] < .023:
            break
        trajectory, _ = law.trajectory(*estimate, 25, bounds=(0., 0.))
        net = np.flatnonzero(trajectory[:, 1] >= 0.)
        floor = np.flatnonzero(trajectory[:, 2] < .020)
        if not len(net) or not len(floor) or floor[0] <= net[0]:
            continue
        landing = trajectory[floor[0]]
        if trajectory[net[0], 2] >= .155 and abs(landing[0]) < COURT[0] and .035 < landing[1] < COURT[1]:
            credit = .44
        elif trajectory[net[0], 2] >= .155:
            credit = max(credit, .15)
    return credit


class TimingMemory:
    """Update a small stroke-timing bandit from chronological optical play."""

    def __init__(self, state=None):
        self.count = np.zeros(len(PROGRAMS), dtype=int)
        self.value = np.zeros(len(PROGRAMS))
        if state is not None:
            self.count[:] = state['count']
            self.value[:] = state['sum']

    def record(self, arm, reward, merit):
        self.count[arm] += 1
        self.value[arm] += .30 * float(reward) + merit

    def choose(self, explore=False):
        utility = (self.value + np.array((.08, 0., 0.))) / (1 + self.count)
        if explore:
            utility += .70 * np.sqrt(np.log(2 + self.count.sum()) / (1 + self.count))
        return int(np.argmax(utility))

    def state(self):
        return dict(count=self.count.tolist(), sum=self.value.tolist(), pick=self.choose())


class ContextTimingMemory:
    """Pool sparse timing returns while retaining fast and slow approach clocks."""

    def __init__(self):
        self.count = np.zeros((2, len(PROGRAMS)), dtype=int)
        self.value = np.zeros_like(self.count, dtype=float)

    def record(self, eta, arm, reward, merit):
        index = int(eta > PREPARATION)
        self.count[index, arm] += 1
        self.value[index, arm] += .30 * float(reward) + merit

    def choose(self, eta):
        if eta is None:
            return 0
        index = int(eta > PREPARATION)
        prior = self.value.sum(axis=0) / np.maximum(1, self.count.sum(axis=0))
        pooled = (self.value[index] + 2. * prior) / (self.count[index] + 2.)
        pooled[0] += .02
        pooled[(self.count[index] < 2) & (np.arange(len(PROGRAMS)) != 0)] = -np.inf
        return int(np.argmax(pooled))

    def state(self):
        return dict(count=self.count.tolist(), sum=self.value.tolist(), fast=self.choose(PREPARATION),
                    slow=self.choose(PREPARATION+1), cutoff=PREPARATION)


class RecoveryRollout(RolloutPlayer):
    """Preserve the factory opening; choose a timing policy for resumed play."""

    def __init__(self, original):
        # The same public-CAD catalog, dry plant, and landing ranker are used by
        # both controls; only the window in which planning begins may change.
        self.__dict__.update(original.__dict__)
        self.arm = 0
        self.context = None

    def command(self, tick, xyz, server, *, estimated_velocity=None):
        context = self.context
        if self.arm == 0 or context is None or not context.outgoing:
            return super().command(tick, xyz, server, estimated_velocity=estimated_velocity)
        self.history.append((tick, xyz))
        if self.plan is not None:
            start, index, until = self.plan
            if tick-start < until:
                action = self.tapes[index, tick-start]
                self.movement += action
                return np.r_[action, 0.]
            self.plan = None
            self.ends = tick
        if np.max(abs(self.movement)) > .002:
            action = -np.clip(self.movement, -1., 1.)
            self.movement += action
            return np.r_[action, 0.]
        if xyz is not None and xyz[1] > .10:
            self.cleared = True
        limit, settle, tolerance = PROGRAMS[self.arm]
        if (self.plans and not self.cleared) or tick - self.ends < settle:
            return np.zeros(6)
        estimate = ((xyz, estimated_velocity) if estimated_velocity is not None and xyz is not None
                    else optical_start(self.history, self.law.gravity))
        if estimate is None or xyz is None:
            return np.zeros(6)
        p, velocity = estimate
        if velocity[1] >= -.034 or not -.81 < p[1] < limit:
            return np.zeros(6)
        if context.blade_error is not None and context.blade_error > tolerance and p[1] > -.30:
            return np.zeros(6)
        if context.last_attempt is not None and tick-context.last_attempt < 2:
            return np.zeros(6)
        context.last_attempt = tick
        detail = self.choose(estimate)
        if detail is None:
            return np.zeros(6)
        if not detail['legal'] and min(detail['robustness']) > 2.8 and p[1] > -.47:
            return np.zeros(6)
        index, duration = detail['index'], detail['duration']
        self.plans.append(dict(tick=tick, **detail))
        self.plan = (tick, index, duration)
        self.cleared = False
        action = self.tapes[index, 0]
        self.movement += action
        return np.r_[action, 0.]


class NearSideRecovery(RecoveryRollout):
    """Offer one earlier attempt just before normal receiving would begin."""

    def command(self, tick, xyz, server, *, estimated_velocity=None):
        if self.arm == 1 and xyz is not None and NEAR_WINDOW[0] < xyz[1] <= NEAR_WINDOW[1]:
            return super().command(tick, xyz, server, estimated_velocity=estimated_velocity)
        return RolloutPlayer.command(self, tick, xyz, server, estimated_velocity=estimated_velocity)


class RallyContext:
    """Identify a new exchange and readiness without contact or joint feedback."""

    def __init__(self, role):
        self.role = role
        self.history = []
        self.home = []
        self.outgoing = False
        self.outgoing_tick = None
        self.approach = None
        self.eta = None
        self.blade_error = None
        self.last_attempt = None
        self.launch = None

    def observe(self, tick, ball, blade, has_opening, law):
        self.history.append((tick, ball))
        if blade is not None and tick < 6:
            self.home.append(np.asarray(blade))
        self.blade_error = None if not self.home or blade is None else float(
            np.linalg.norm(np.asarray(blade) - np.median(self.home, axis=0)))
        if not self.outgoing and has_opening and ball is not None and ball[1] > .24:
            # On a receive, an earlier inbound crossing is discounted by the
            # first learner program, which starts only on the near half.
            minimum_tick = 18 if self.role == 1 else 22
            if tick >= minimum_tick:
                self.outgoing = True
                self.outgoing_tick = tick
        if self.outgoing and self.approach is None and ball is not None and ball[1] < .16:
            self.eta = time_to_reach(self.history, law)
            if self.eta is not None:
                self.approach = tick
