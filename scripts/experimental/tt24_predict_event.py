"""Learn a continuous corrective motor inverse from visible rebound transitions."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_motor import optical_sightings
from .tt24_predict import BLOCK, PATH as BODY_DATA, arrays, seeds


PATH = Path('artifacts/tt24/predict-event-data.json.gz')
RESULT = Path('artifacts/tt24/predict-event-result.json.gz')
INTEGRATED_RESULT = Path('artifacts/tt24/predict-event-integrated.json.gz')
PRACTICE_SEED = 85341
HELD_SEED = 85342
INTEGRATED_HELD_SEED = 85343


def ridge(design, target, lam):
    penalty = np.eye(design.shape[1]) * lam
    penalty[0, 0] *= .03
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


class EncoderInverse:
    """Invert measured joint changes conditioned on the last issued command."""

    def __init__(self, logged):
        state, action, following, _, _, elapsed = arrays(logged)
        x, y, u = state[elapsed == BLOCK], following[elapsed == BLOCK], action[elapsed == BLOCK]
        design = np.c_[np.ones(len(x)), x[:, 7:22], u]
        target = np.c_[y[:, 7:12] - x[:, 7:12], y[:, 12:17]]
        self.fit = ridge(design, target, .003)
        self.weight = np.r_[np.ones(5), np.full(5, .16)]
        effect = self.fit[-5:] * self.weight
        self.inverse = effect.T @ np.linalg.inv(effect @ effect.T + np.eye(5) * .012)

    def reset(self):
        pass

    def command(self, posture, speed, previous, goal):
        effect = np.r_[1., posture * 5., speed / 3., previous] @ self.fit[:-5]
        wanted = np.r_[3. * (goal - posture), np.zeros(5)]
        return np.clip(((wanted - effect) * self.weight) @ self.inverse, -.16, .16)


class IntegratedInverse:
    """Fit steady encoder changes against accumulated rate commands."""

    def __init__(self, logged):
        features, outcomes = [], []
        for episode in logged:
            state, action, following, _, _, elapsed = arrays([episode])
            history = np.r_[np.zeros((1, 5)), action[:-1]]
            design = np.c_[np.ones(len(state)), np.cumsum(action, axis=0), action, history]
            features.extend(design[elapsed == BLOCK])
            outcomes.extend(following[elapsed == BLOCK, 7:12] / 5.)
        self.gain = ridge(np.asarray(features), np.asarray(outcomes), .008)[1:6]
        self.inverse = self.gain.T @ np.linalg.inv(self.gain @ self.gain.T + np.eye(5) * .0007)

    def reset(self):
        self.drive = np.zeros(5)

    def command(self, posture, speed, previous, goal):
        remaining = goal - self.drive
        action = np.clip(.63 * remaining @ self.inverse, -.15, .15)
        self.drive += action @ self.gain
        return action


def appearance(obs):
    ball, _ = optical_sightings(obs.get('rgb'))
    return None if ball is None else np.r_[ball.center * (5., 4.), 2. * np.log(ball.radius / .08)]


def play(arena, launch, body, *, goal=None, controller=None, split='practice', hold=False):
    body.reset()
    obs = arena.reset(seed=launch, split=split)
    start = obs['joints']
    setting = np.zeros(5) if goal is None else goal
    action = np.zeros(5)
    trace = []
    visual = appearance(obs)
    done = False
    step = 0
    while not done:
        if controller is not None:
            setting = controller.update(visual, obs['joints'], obs['joint_velocity'], step)
        if step % BLOCK == 0 and not hold:
            action = body.command(obs['joints'] - start, obs['joint_velocity'], action, setting)
        before = obs
        obs, reward, done = arena.step(action)
        trace.append([None if visual is None else visual.tolist(), before['joints'].tolist(),
                      before['joint_velocity'].tolist(), action.tolist(), int(reward)])
        visual = appearance(obs)
        step += 1
    return trace


def collect(arena, body, output):
    rng = np.random.default_rng(PRACTICE_SEED - 1)
    episodes = []
    for i, launch in enumerate(seeds(PRACTICE_SEED, 16)):
        goal = np.clip(rng.normal(0., .017, 5), -.034, .034)
        episode = play(arena, launch, body, goal=goal)
        print('event practice', i + 1, launch, sum(row[-1] for row in episode), len(episode), flush=True)
        episodes.append(dict(launch=launch, trace=episode))
    with gzip.open(output, 'wt') as stream:
        json.dump(dict(schema='tt24-predict-event-1', practice=episodes), stream, separators=(',', ':'))


def load(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def observations(episode):
    trace = episode['trace']
    x = np.asarray([np.full(3, np.nan) if t[0] is None else t[0] for t in trace])
    q = np.asarray([t[1] for t in trace])
    reward = np.asarray([t[4] for t in trace])
    return x, q, reward


def gravity_axis(episodes):
    accelerations = []
    for episode in episodes:
        x, q, _ = observations(episode)
        for i in range(5, len(x) - 5):
            recent = x[i - 3:i + 4]
            if not np.isfinite(recent).all() or np.max(np.ptp(q[i - 3:i + 4], axis=0)) > .055:
                continue
            acceleration = (recent[0] + recent[-1] - 2. * recent[3]) / 9.
            if .002 < acceleration[1] < .050:
                accelerations.append(acceleration / np.linalg.norm(acceleration))
    if not accelerations:
        raise ValueError('no visible approach segments with limited measured joint motion')
    vertical = np.median(accelerations, axis=0)
    vertical /= np.linalg.norm(vertical)
    _, _, basis = np.linalg.svd(vertical[None])
    return np.column_stack((basis[1:].T, vertical))


def slope(x):
    frame = np.arange(len(x)) - (len(x) - 1) / 2.
    return frame @ x / (frame @ frame)


def event_features(episodes, basis):
    rows = []
    for episode in episodes:
        x, q, reward = observations(episode)
        x = x @ basis
        latest = -1
        first = True
        for paid in np.flatnonzero(reward):
            window = np.arange(max(7, latest + 5, paid - 19), paid - 4)
            valid = window[np.isfinite(x[window, 2])]
            if not len(valid):
                continue
            turn = valid[np.argmax(x[valid, 2])]
            before, after = x[turn - 7:turn - 1], x[turn + 2:turn + 8]
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                continue
            incoming, outgoing = slope(before), slope(after)
            if incoming[2] < .027 or outgoing[2] > -.024:
                continue
            # Use an extrapolated image position at the turn, away from limb
            # occlusion or the camera boundary during the contact itself.
            contact = np.mean(before, axis=0) + 4.5 * incoming
            impulse = (outgoing[:2] - incoming[:2]) / incoming[2]
            latest = turn
            rows.append(dict(contact=contact, posture=q[turn], impulse=impulse,
                             incoming=incoming, outgoing=outgoing, first=first))
            first = False
    return rows


class EventModel:
    """Identify how joint posture changes the apparent direction of a rebound."""

    def __init__(self, episodes):
        self.basis = gravity_axis(episodes)
        rows = event_features(episodes, self.basis)
        if not rows:
            raise ValueError('no visual encounters could be associated with observed rewards')
        self.rows = rows
        contact = np.asarray([row['contact'] for row in rows])
        posture = np.asarray([row['posture'] for row in rows])
        self.pivot = posture.mean(axis=0)
        first_contacts = contact[[row['first'] for row in rows]]
        self.center = np.median(first_contacts, axis=0)
        self.visual_scale = np.maximum(contact.std(axis=0), .15)
        self.pose_scale = .07
        self.fit = ridge(self.features(contact, posture), np.asarray([row['impulse'] for row in rows]), 1.4)
        self.no_pose = ridge(self.features(contact, posture)[:, :4],
                             np.asarray([row['impulse'] for row in rows]), 1.4)
        motor = self.fit[4:]
        self.inverse = motor.T @ np.linalg.inv(motor @ motor.T + np.eye(5) * .42)
        self.y_neutral = np.median(first_contacts[:, 2])

    def features(self, contact, posture):
        return np.c_[np.ones(len(contact)), (contact - self.center) / self.visual_scale,
                     (posture - self.pivot) / self.pose_scale]

    def desired(self, location, incoming, posture):
        model = self.features(location[None], posture[None])[0]
        drift = model @ self.fit
        desired_velocity = -.060 * (location[:2] - self.center[:2])
        desired_impulse = np.clip((desired_velocity - incoming[:2]) / max(incoming[2], .05), -1.9, 1.9)
        return posture + ((desired_impulse - drift) @ self.inverse) * self.pose_scale


class BodyCorrector:
    """Update the inverse from live approach measurements at each encounter."""

    def __init__(self, model, feedback=True):
        self.model = model
        self.feedback = feedback
        self.trajectory = []
        self.setting = np.zeros(5)
        self.origin = None
        self.changed = -30

    def update(self, visual, posture, speed, step):
        if self.origin is None:
            self.origin = posture.copy()
            neutral = self.model.center.copy()
            self.setting = np.clip(self.model.desired(neutral, np.array((0., 0., .10)), posture) - posture,
                                   -.025, .025)
        if visual is None:
            self.trajectory.clear()
            return self.setting
        reading = visual @ self.model.basis
        self.trajectory.append(reading)
        self.trajectory = self.trajectory[-7:]
        if not self.feedback or len(self.trajectory) < 7 or step - self.changed < 15:
            return self.setting
        incoming = slope(np.asarray(self.trajectory))
        if incoming[2] <= .035 or np.linalg.norm(speed) > 1.9:
            return self.setting
        time_left = (self.model.y_neutral - reading[2]) / incoming[2]
        if time_left < 6. or time_left > 24.:
            return self.setting
        contact = reading + incoming * min(10., time_left)
        correction = self.model.desired(contact, incoming, posture) - self.origin
        self.setting = np.clip(.30 * self.setting + .70 * correction, -.027, .027)
        self.changed = step
        return self.setting


def validate(episodes):
    rows = []
    for start in range(4):
        test = [v for i, v in enumerate(episodes) if i % 4 == start]
        model = EventModel([v for i, v in enumerate(episodes) if i % 4 != start])
        witnesses = event_features(test, model.basis)
        x = model.features(np.asarray([r['contact'] for r in witnesses]), np.asarray([r['posture'] for r in witnesses]))
        y = np.asarray([r['impulse'] for r in witnesses])
        rows.append(dict(n=len(y), joint_mse=np.mean((x @ model.fit - y)**2, axis=0).tolist(),
                         omit_mse=np.mean((x[:, :4] @ model.no_pose - y)**2, axis=0).tolist()))
    return rows


def qualify(arena, body, data, output, held_seed):
    model = EventModel(data)
    tests = validate(data)
    print('contact prediction validation', tests, 'n', len(model.rows), 'pose fit', model.fit[4:], flush=True)
    held = {}
    for name in ('hold', 'neutral', 'fixed', 'feedback'):
        runs = []
        for i, launch in enumerate(seeds(held_seed, 12)):
            controller = BodyCorrector(model, feedback=name == 'feedback') if name in ('fixed', 'feedback') else None
            trace = play(arena, launch, body, controller=controller, split='held', hold=name == 'hold')
            outcome = arena.judge_metrics() | dict(trace=trace)
            runs.append(outcome)
            print('event held', name, i + 1, launch, outcome['rebounds'], outcome['steps'], flush=True)
        held[name] = runs
    with gzip.open(output, 'wt') as stream:
        json.dump(dict(schema='tt24-predict-event-1', validation=tests, optical_event_count=len(model.rows),
                       fit=model.fit.tolist(), basis=model.basis.tolist(), held=held), stream, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('collect', 'qualify', 'integrated'))
    parser.add_argument('--data', type=Path, default=PATH)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    encoder_data = load(BODY_DATA)['practice']
    integrated = args.task == 'integrated'
    body = IntegratedInverse(encoder_data) if integrated else EncoderInverse(encoder_data)
    arena = BounceArena(max_steps=360)
    try:
        if args.task == 'collect':
            collect(arena, body, args.data)
        else:
            output = args.output or (INTEGRATED_RESULT if integrated else RESULT)
            seed = INTEGRATED_HELD_SEED if integrated else HELD_SEED
            qualify(arena, body, load(args.data)['practice'], output, seed)
    finally:
        arena.close()


if __name__ == '__main__':
    main()
