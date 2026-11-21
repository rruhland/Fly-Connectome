"""Transfer a recurrent shot repertoire to a stereo/command-only SO-101 receiver."""

from collections import deque

import numpy as np

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt19_flight import optical_start
from .tt19_model_rollout import observed_ball
from .tt20_curriculum import OUTPUT, configuration
from .tt20_natural import NaturalViewTransport
from .tt21_body_state import CommandTwin
from .tt22_cycle import CyclePlanner, World


class OpticalCalibration:
    """Estimate low-amplitude speed biases from the next incoming RGB sightings."""

    def __init__(self, law, *, learn=False, samples=()):
        self.law = law
        self.learn = learn
        self.samples = deque(samples, maxlen=80)
        self.pending = {}

    def reset(self):
        self.pending.clear()

    def estimate(self, tick, start):
        if start is None:
            return None
        position, velocity = start
        if self.learn:
            prediction = self.pending.pop(tick, None)
            # No bounce/contact semantics or hidden measurements enter the fit.
            if prediction is not None and velocity[1] < -.03 and position[2] > .095:
                residual = (position - prediction) / 3.
                if np.max(abs(residual)) < .018:
                    self.samples.append(residual.tolist())
            if velocity[1] < -.03 and position[2] > .14 and position[1] > -.54:
                trajectory, _ = self.law.trajectory(position, velocity, 3, bounds=(.424, .80))
                if min(trajectory[:, 2]) > .087:
                    self.pending[tick+3] = trajectory[-1]
        if len(self.samples) < 8:
            return start
        bias = np.clip(np.median(self.samples, axis=0), -.006, .006)
        return position, velocity + bias * min(1., len(self.samples) / 25.)

    def state(self):
        return list(self.samples)


class TransferredPlanner(CyclePlanner):
    def __init__(self, world, styles):
        super().__init__(world)
        self.styles = styles

    def _options(self, side, frames, *, wide=False):
        goals, notes = super()._options(side, frames, wide=wide)
        if not self.styles or wide:
            return goals, notes
        keep = [i for i, row in enumerate(notes) if (row['pitch'], row['pace'], row['lead']) in self.styles]
        return [goals[i] for i in keep], [notes[i] for i in keep]


class OpticalReceiver:
    """Build a new dry solve from only stereo RGB, the clock and issued rate commands."""

    def __init__(self, lenses, *, styles=(), learn=False, samples=()):
        _, _, law, _, _ = configuration(OUTPUT)
        self.law = law
        # The live arena/camera pose/ball/encoders are not held by this object.
        self.world = World()
        self.planner = TransferredPlanner(self.world, styles)
        self.twin = CommandTwin(self.world.arena)
        self.lenses = lenses
        self.calibration = OpticalCalibration(law, learn=learn, samples=samples)

    def reset(self):
        self.twin.reset()
        self.calibration.reset()
        self.tracker = NaturalViewTransport('fixed', lens_fov=self.lenses, flight=False).tracker()
        self.history = []
        self.tape = None
        self.start = None
        self.stage = 'incoming'
        self.attempts = 0
        self.replanned = False
        self.last_attempt = -5
        self.decisions = []

    def _shadow(self, tick, estimate):
        arena = self.world.arena
        position, speed = estimate
        mujoco.mj_copyData(arena.data, arena.model, self.twin.data)
        ball, velocity = arena._ball_joint, arena._ball_dof
        arena.data.qpos[ball:ball+3] = position + [0., 0., TABLE_HEIGHT]
        arena.data.qpos[ball+3:ball+7] = [1.,0.,0.,0.]
        arena.data.qvel[velocity:velocity+3] = speed * 30.
        arena.data.qvel[velocity+3:velocity+6] = observed_ball(self.history, estimate, self.law).spin
        arena.steps = tick
        arena._last_hitter = -1
        arena._serving = arena._waiting_for_strike = arena._server_bounce = False
        arena._received_bounce = position[1] < -.32 and any(
            p is not None and p[1] < -.23 and p[2] < .081 for _, p in self.history[-7:])
        arena.terminated = arena.truncated = False
        arena.legal_serve = True
        arena._previous_touches = {}
        arena.events = []
        mujoco.mj_forward(arena.model, arena.data)

    def _estimate(self, tick):
        return self.calibration.estimate(tick, optical_start(self.history, self.law.gravity))

    def _refinement_phase(self, tick, xyz):
        if not self.replanned and tick-self.start >= 5 and -.50 < xyz[1] < -.18:
            self.replanned = True
            return 'refine'
        return None

    def command(self, tick, rgb):
        self.tracker.update(rgb, tick=tick)
        ball = self.tracker.ball
        point = np.asarray(ball.position) if ball is not None and ball.observed else None
        self.history.append((tick, point))
        start = self._estimate(tick)
        if start is not None:
            xyz, speed = start
            if self.tape is not None and speed[1] > .024 and xyz[1] > -.72:
                self.tape = None
                self.stage = 'outgoing'
            if self.stage == 'outgoing' and speed[1] > .023 and xyz[1] > .19:
                self.stage = 'crossed'
            if self.stage == 'crossed' and speed[1] < -.035:
                self.stage = 'incoming'
                self.attempts = 0
                self.replanned = False
                self.last_attempt = tick - 5
            if (self.stage == 'incoming' and self.tape is None and self.attempts < 4
                    and tick - self.last_attempt >= 4 and speed[1] < -.039 and xyz[1] < .77):
                self._shadow(tick, start)
                self.tape, trial = self.planner.choose(1, wide=self.attempts>0)
                self.decisions.append(dict(**trial, phase='launch', donor_subset=bool(self.planner.styles and self.attempts==0),
                                           optical=xyz.tolist(), optical_velocity_m_s=(30*speed).tolist()))
                self.start = self.last_attempt = tick
                self.attempts += 1
            phase = self._refinement_phase(tick, xyz) if self.tape is not None and speed[1] < -.039 else None
            if phase is not None:
                self._shadow(tick, start)
                extension, detail = self.planner.choose(1, wide=False)
                self.decisions.append(dict(**detail, phase=phase, donor_subset=bool(self.planner.styles),
                                           optical=xyz.tolist(), optical_velocity_m_s=(30*speed).tolist()))
                if extension is not None:
                    self.tape, self.start = extension, tick
        near = np.zeros(6)
        if self.tape is not None and tick-self.start < len(self.tape):
            near[:5] = self.tape[tick-self.start]
        self.twin.advance(near)
        return near

    def close(self):
        self.planner.close()
        self.world.close()


def teacher_styles(rows):
    styles = set()
    for row in rows:
        for index, decision in enumerate(row['decisions']):
            if decision['side'] != 1 or 'swing' not in decision:
                continue
            until = next((n['tick'] for n in row['decisions'][index+1:] if n['side']==1), np.inf)
            if any(e['kind']=='legal_return' and e.get('side')==1 and decision['tick'] < e['step'] < until
                   for e in row['events']):
                swing = decision['swing']
                styles.add((swing['pitch'], swing['pace'], swing['lead']))
    return sorted(styles)


def rally_optical(world, opponent, receiver, seed, *, verbose=False):
    world.reset(seed)
    receiver.reset()
    arena = world.arena
    actions, decisions = [], []
    sender, tape, begin, attempts, last_attempt = None, None, 0, 0, -6
    for tick in range(arena.max_steps):
        # Optical recipient never sees these privileged far-only values.
        far = np.zeros(4)
        if sender != arena._last_hitter:
            sender = arena._last_hitter
            tape, attempts, last_attempt = None, 0, -6
        if sender == 1:
            if (tape is None and arena.velocity[1] > .4 and tick-last_attempt >= 5 and attempts < 3
                    and arena._racket_ids[1] not in arena._previous_touches):
                tape, record = opponent.choose(-1, wide=attempts>0)
                decisions.append(record)
                begin, last_attempt, attempts = tick, tick, attempts+1
            if tape is not None and tick-begin < len(tape):
                far[:] = tape[tick-begin]
        sizes = len(receiver.decisions)
        near = receiver.command(tick, arena.render_cameras())
        if verbose and len(receiver.decisions) > sizes:
            note = receiver.decisions[-1]
            print('optical', seed, tick, note['candidate_count'], note['solver_legal'], 'chosen' in note,
                  arena.legal_returns, arena.rival_returns, flush=True)
        actions.append([near.tolist(), far.tolist()])
        _, score, ended, capped, _ = world.step(near, far)
        if ended or capped:
            break
    events = arena.events.copy()
    legal = [event for event in events if event['kind']=='legal_return']
    if any(a['side']==b['side'] for a,b in zip(legal, legal[1:])):
        raise ValueError('nonalternating return sequence')
    return dict(seed=seed, actions=actions, decisions=receiver.decisions, far_decisions=decisions,
                result=arena.point_result(), reward=score, events=events,
                rgb_sha256=receiver.tracker.digest.hexdigest(), legal_alternating_returns=len(legal),
                online_samples=receiver.calibration.state())
