"""Native-image Pong practice and paired retention for the shared return learner."""

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong


ACTIONS = (-1, 0, 1)
CAMERA_SIZE = 128
CAMERA_TICKS = 4
NATIVE_HZ = 120
CAMERA_SECONDS = CAMERA_TICKS / NATIVE_HZ
PRACTICE_OPPONENT_SPEED = .35
RALLY_TICKS = 1680
MINIMUM_RETENTION_CASES = 32
MINIMUM_LATER_OPPORTUNITIES = 20
RETENTION_METRICS = {
    'first_return_fraction': (.75, .10),
    'later_return_fraction': (.65, .15),
    'earned_point_fraction': (.15, .10),
}


@dataclass(frozen=True)
class PongCase:
    serve: int
    presentation: int
    exploration: int


def cases_from_seeds(seeds):
    """Use separate, reproducible nuisance and policy randomness for each serve."""
    cases = []
    for seed in seeds:
        first, second = np.random.SeedSequence([783421, int(seed)]).generate_state(2)
        cases.append(PongCase(int(seed), int(first), int(second)))
    return cases


class _Presentation:
    """Apply the same palette to every rendered component and permute action IDs."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.rotation = int(rng.integers(4))
        self.reflect = bool(rng.integers(2))
        self.order = rng.permutation(ACTIONS)
        dark = rng.integers(1, 40, 3, dtype=np.uint8)
        light = rng.integers(175, 250, 3, dtype=np.uint8)
        self.palette = np.stack((dark, light))

    def frame(self, native):
        if self.reflect:
            native = native[:, ::-1]
        native = np.rot90(native, self.rotation)
        image = np.empty((*native.shape, 3), dtype=np.uint8)
        image[...] = self.palette[0]
        image[native] = self.palette[1]
        return image

    def drive(self, action):
        return int(self.order[ACTIONS.index(action)])


class _RandomActions:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.remaining = 0
        self.action = None
        self.samples = []

    def observe(self, image, preceding_action, reward):
        from scripts.experimental.tt3_visual import Sample

        self.samples.append(Sample(image, preceding_action, reward))

    def choose(self):
        if self.remaining == 0:
            alternatives = [action for action in ACTIONS if action != self.action]
            self.action = int(self.rng.choice(alternatives))
            self.remaining = int(self.rng.integers(5, 13))
        self.remaining -= 1
        return self.action


def _native_rallies(cases, agents, *, max_ticks, opponent_speed, camera_ticks=CAMERA_TICKS, game_factory=Pong):
    """Run native physics. The agents only receive the camera, action ID and score."""
    if not cases or not len(cases) == len(agents):
        raise ValueError('one agent per native Pong case is required')
    if max_ticks < 1 or camera_ticks < 1:
        raise ValueError('maximum native ticks and camera stride must be positive')
    game = game_factory([case.serve for case in cases], Physics(opponent_speed=opponent_speed))
    presentations = [_Presentation(case.presentation) for case in cases]
    active = torch.ones(len(cases), dtype=torch.bool)
    drive = torch.zeros(len(cases))
    actions = [None] * len(cases)
    action_traces = [bytearray() for _ in cases]
    contacts = [[] for _ in cases]
    rows = [None] * len(cases)
    frames = game.render(CAMERA_SIZE, CAMERA_SIZE).numpy()
    for index, agent in enumerate(agents):
        agent.observe(presentations[index].frame(frames[index]), None, 0.)

    for tick in range(max_ticks):
        if tick % camera_ticks == 0:
            for index in active.nonzero().flatten().tolist():
                action = agents[index].choose()
                if type(action) is not int or action not in ACTIONS:
                    raise ValueError('Pong action must be one of the signed numeric IDs')
                actions[index] = action
                action_traces[index].append(ACTIONS.index(action))
                drive[index] = presentations[index].drive(action)

        # Native float32 crossing check used by the existing public return scorer.
        # Audit data are kept outside every policy's observation and reward path.
        x = game.ball[:, 0]
        vx = game.ball_velocity[:, 0]
        plane = game.config.paddle_x + game.config.ball_radius
        incoming = (x >= plane) & (x + vx * game.config.dt <= plane) & (vx < 0) & active
        outcome = game.step(drive, auto_reset=False)
        for index in incoming.nonzero().flatten().tolist():
            contacts[index].append(dict(tick=tick + 1, returned=bool(outcome.hits[index])))
        finished = active & ((outcome.scores != 0) | (tick == max_ticks - 1))
        observe = active if (tick + 1) % camera_ticks == 0 else finished
        if observe.any():
            frames = game.render(CAMERA_SIZE, CAMERA_SIZE).numpy()
            for index in observe.nonzero().flatten().tolist():
                score = int(outcome.scores[index])
                agents[index].observe(presentations[index].frame(frames[index]), actions[index], float(score))
                if finished[index]:
                    rows[index] = dict(case=asdict(cases[index]), score=score, ticks=tick + 1,
                                       camera_actions=len(action_traces[index]), contacts=contacts[index],
                                       next_serve=int(game.rng[index]),
                                       action_sha256=hashlib.sha256(action_traces[index]).hexdigest())
        active &= ~finished
        if not active.any():
            break
    return rows


def source_episodes(seeds, *, max_ticks=RALLY_TICKS, opponent_speed=PRACTICE_OPPONENT_SPEED,
                    batch_size=8):
    """Collect raw pretraining sequences from random held actions and natural score."""
    if batch_size < 1:
        raise ValueError('source collection batch size must be positive')
    cases = cases_from_seeds(seeds)
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        agents = [_RandomActions(case.exploration) for case in batch]
        _native_rallies(batch, agents, max_ticks=max_ticks, opponent_speed=opponent_speed)
        yield from (agent.samples for agent in agents)


class _VisualActionEffects:
    """Infer a discrete drive map and two-stage lag from visible probe responses."""

    minimum_uses = 6
    minimum_observations = 14
    dwell_frames = 4

    def __init__(self, rng):
        first, second, third = rng.permutation(ACTIONS).tolist()
        self.sequence = (first, second, third, second, first, third)
        lags = [(early, late) for early in (.045, .095, .17) for late in (.13, .22, .36)]
        self.alpha = (1 - np.exp(-CAMERA_SECONDS / np.asarray(lags)))[:, :, None]
        self.trace = np.zeros((len(lags), 2, len(ACTIONS)))
        self.integral = np.zeros((len(lags), len(ACTIONS)))
        self.uses = np.zeros(len(ACTIONS), dtype=int)
        self.history = {}
        self.frames = 0
        self.axis = None
        self.effects = None
        self.ready = None
        self.identity = None

    def observe(self, binding, preceding_action):
        if self.axis is not None:
            return
        if preceding_action is not None:
            index = ACTIONS.index(preceding_action)
            self.uses[index] += 1
            previous = self.trace[:, 0].copy()
            self.trace[:, 0] *= 1. - self.alpha[:, 0]
            self.trace[:, 0, index] += self.alpha[:, 0, 0]
            self.trace[:, 1] += self.alpha[:, 1] * (
                (previous + self.trace[:, 0]) / 2. - self.trace[:, 1])
            self.integral += self.trace[:, 1] * CAMERA_SECONDS
            self.frames += 1
        for track in binding.tracks:
            if track.observed:
                self.history.setdefault(track.id, []).append((self.frames, self.integral.copy(), track.position))
        if self.uses.min() < self.minimum_uses:
            return
        candidates = []
        for track in binding.tracks:
            fitted = self._fit(self.history.get(track.id, ()))
            if fitted is not None:
                strength, axis, effects, ready = fitted
                agrees = binding.controlled is not None and binding.controlled.id == track.id
                if agrees or strength >= .77:
                    candidates.append((strength + .08 * agrees, track.id, axis, effects, ready))
        if not candidates:
            return
        _, self.identity, self.axis, self.effects, self.ready = max(candidates, key=lambda candidate: candidate[0])

    def _fit(self, observations):
        if len(observations) < self.minimum_observations or observations[0][0] != 0:
            return None
        t, x, y = (np.asarray(values) for values in zip(*observations, strict=True))
        if np.linalg.norm(np.ptp(y, axis=0)) < .026:
            return None
        centered = y - y.mean(axis=0)
        features = x[..., [1, 0, 0]] - x[..., [2, 2, 1]]
        features -= features.mean(axis=0)
        support = np.square(features).sum(axis=0)
        effects = np.einsum('tki,td->kid', features, centered) / (support[..., None] + 1e-8)
        residual = np.square(centered[:, None, None, :] - features[..., None] * effects).sum(axis=(0, 3))
        lag, neutral = np.unravel_index(np.argmin(residual), residual.shape)
        movement = effects[lag, neutral]
        strength = np.linalg.norm(movement)
        spread = np.square(centered).sum()
        best = residual[lag, neutral]
        runner_up = min(residual[:, index].min() for index in range(len(ACTIONS)) if index != neutral)
        if strength < .18 or best > .16 * spread or runner_up < 1.25 * best + .000025:
            return None
        time = t - t.mean()
        trend = time[:, None] * ((time @ centered) / (time @ time))
        null = np.square(centered - trend).sum()
        confidence = 1 - best / max(null, 1e-8)
        if confidence < .40:
            return None
        axis = movement / strength
        first = (1, 0, 0)[neutral]
        second = (2, 2, 1)[neutral]
        if axis[np.argmax(abs(axis))] < 0:
            axis = -axis
            first, second = second, first
        return confidence, axis, (ACTIONS[second], ACTIONS[neutral], ACTIONS[first]), y[0]

    def probe(self):
        return int(self.sequence[(self.frames // self.dwell_frames) % len(self.sequence)])

    def point_actor(self, tracks):
        if self.axis is None:
            return None
        current = next((track for track in tracks if track.id == self.identity), None)
        if current is not None:
            return current
        if self.identity is None:
            initial = min(tracks, key=lambda track: np.linalg.norm(track.position - self.ready), default=None)
            if initial is not None and np.linalg.norm(initial.position - self.ready) < .055:
                self.identity = initial.id
                return initial
        return None

    def action(self, error):
        return self.effects[0] if error < -.011 else self.effects[2] if error > .011 else self.effects[1]


class _PongReturnAgent:
    """Bind image roles, apply a shared relation and execute its projected target."""

    def __init__(self, role_model, memory, seed, *, option=None, motor_factory=_VisualActionEffects):
        from scripts.experimental.tt3_learning import relation_goal
        from scripts.experimental.tt3_visual import RoleBinder

        rng = np.random.default_rng(seed)
        self.binder = RoleBinder(role_model)
        self.motor_factory = motor_factory
        self.motor = self.motor_factory(rng)
        self.requested = option
        self.memory = memory
        self.rng = rng
        self.goal = getattr(memory, 'goal', relation_goal)
        self.choose_for_response = getattr(memory, 'choose_for_response', None)
        self.goal_for_response = getattr(memory, 'goal_for_response', None)
        self.begin_point()

    def begin_point(self):
        self.binder.reset()
        if self.motor.axis is None and self.motor.frames:
            self.motor = self.motor_factory(self.rng)
        self.motor.identity = None
        self.option = self.memory.choose(self.rng, explore=0.) if self.requested is None else self.requested
        self.binding = None
        self.credited = False
        self.actor = None
        self.actor_history = deque(maxlen=5)
        self.mover_id = None
        self.mover_history = deque(maxlen=6)
        self._last_goal = None
        self._response_evaluated = False

    def observe(self, image, preceding_action, reward):
        self.binding = self.binder.update(image, preceding_action)
        self.motor.observe(self.binding, preceding_action)
        actor = self.motor.point_actor(self.binding.tracks)
        mover = self.binding.relevant
        if actor is not None and (mover is None or mover.id == actor.id):
            mover = None
        if actor is None or self.actor is None or actor.id != self.actor.id:
            self.actor_history.clear()
        self.actor = actor
        self._record(mover)

    def _record(self, mover):
        if self.actor is not None:
            self.actor_history.append(np.asarray(self.actor.position))
        if mover is None or mover.id != self.mover_id:
            self.mover_history.clear()
            self.mover_id = mover.id if mover is not None else None
        if mover is not None:
            point = np.asarray(mover.position)
            if len(self.mover_history) >= 3:
                step = point - self.mover_history[-1]
                prior = self.mover_history[-1] - self.mover_history[0]
                if np.any((abs(prior) > .016) & (step * prior < -.00013)):
                    previous = self.mover_history[-1]
                    self.mover_history.clear()
                    self.mover_history.append(previous)
            self.mover_history.append(point)

    def choose(self):
        if self.motor.axis is None:
            return self.motor.probe()
        if not self._response_evaluated and self.requested is None and self.choose_for_response is not None:
            self.option = self.choose_for_response(self.motor.axis[:, None], self.rng)
            self._response_evaluated = True
        if self.actor is None:
            return self.motor.effects[1]
        position = np.asarray(self.actor.position)
        velocity = np.zeros_like(position)
        if len(self.actor_history) > 1:
            velocity = (self.actor_history[-1] - self.actor_history[0]) / (
                len(self.actor_history) - 1)
        goal = self._goal(position)
        error = float((goal - position - 6. * velocity) @ self.motor.axis)
        return self.motor.action(error)

    def _goal(self, position):
        axis = self.motor.axis
        if len(self.mover_history) < 2:
            return self._last_goal if self._last_goal is not None else self.motor.ready
        normal = np.array([-axis[1], axis[0]])
        target = self.mover_history[-1]
        velocity = (target - self.mover_history[0]) / (len(self.mover_history) - 1)
        # Distances and velocities use observed image dimensions and cameras.
        # There is no physics-state or native side information in this projection.
        arguments = (self.option, position, target, velocity, self.motor.ready)
        options = dict(normal=normal, motion_bounds=(np.full(2, -.5), np.full(2, .5)),
                       response_axis=axis, offset=.065)
        if self.goal_for_response is None:
            goal = self.goal(*arguments, **options)
        else:
            goal = self.goal_for_response(axis[:, None], *arguments, **options)
        if goal is None:
            return self.motor.ready
        self.credited = True
        self._last_goal = goal
        return goal


def practice_pong(cases, role_model, memory, *, max_ticks=RALLY_TICKS,
                  opponent_speed=PRACTICE_OPPONENT_SPEED, batch_size=24, assignment_seed=704625,
                  options=None, points_per_match=1):
    """Balance exploratory hypotheses before choosing from native score estimates."""
    from scripts.experimental.tt3_learning import RELATIONS

    if options is None:
        rng = np.random.default_rng(assignment_seed)
        indices = np.tile(np.arange(len(RELATIONS)), (len(cases) + len(RELATIONS) - 1) // len(RELATIONS))
        options = rng.permutation(indices[:len(cases)]).tolist()
    if len(options) != len(cases):
        raise ValueError('one requested option per Pong practice case is required')
    return _play_pong(cases, role_model, memory, options, max_ticks, opponent_speed, batch_size,
                      points_per_match=points_per_match, update_memory=True)


def evaluate_pong(cases, role_model, memory, *, max_ticks=RALLY_TICKS,
                  opponent_speed=PRACTICE_OPPONENT_SPEED, batch_size=24, points_per_match=1,
                  motor_factory=_VisualActionEffects):
    """Evaluate learned numerical state read-only on paired native starting cases."""
    return _play_pong(cases, role_model, memory, [None] * len(cases), max_ticks, opponent_speed,
                      batch_size, points_per_match=points_per_match, update_memory=False, motor_factory=motor_factory)


def _play_pong(cases, role_model, memory, options, max_ticks, opponent_speed, batch_size,
               *, points_per_match, update_memory, motor_factory=_VisualActionEffects):
    if not cases or min(batch_size, points_per_match) < 1:
        raise ValueError('positive Pong case, point and batch counts are required')
    before = state_sha256(memory.state())
    rows = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        agents = [_PongReturnAgent(role_model, memory, case.exploration, option=option, motor_factory=motor_factory)
                  for case, option in zip(batch, options[start:start + batch_size], strict=True)]
        point_cases = batch
        matches = [[] for _ in batch]
        for point in range(points_per_match):
            played = _native_rallies(point_cases, agents, max_ticks=max_ticks, opponent_speed=opponent_speed)
            for row, case, agent, match in zip(played, batch, agents, matches, strict=True):
                row.update(match=asdict(case), point=point + 1,
                           relation=int(agent.option), visual_effects=agent.motor.effects,
                           visual_probe_frames=agent.motor.frames, relation_executed=agent.credited)
                if update_memory and agent.credited:
                    contextual_update = getattr(memory, 'observe_for_response', None)
                    if contextual_update is None:
                        memory.observe(agent.option, row['score'])
                    else:
                        contextual_update(agent.motor.axis[:, None], agent.option, row['score'])
                match.append(row)
            if point + 1 < points_per_match:
                point_cases = [PongCase(row['next_serve'], case.presentation, case.exploration)
                               for row, case in zip(played, batch, strict=True)]
                for agent in agents:
                    agent.begin_point()
        rows.extend(row for match in matches for row in match)
    if not update_memory and state_sha256(memory.state()) != before:
        raise ValueError('Pong evaluation changed the shared ReturnMemory')
    return rows


def state_sha256(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _outcomes(row):
    contacts = row['contacts']
    first = contacts[:1]
    later = contacts[1:]
    first_return = sum(contact['returned'] for contact in first)
    later_returns = sum(contact['returned'] for contact in later)
    return (first_return, len(first), later_returns, len(later),
            int(row['score'] > 0 and first_return > 0), 1)


def _fractions(outcomes, axis):
    totals = outcomes.sum(axis=axis).astype(float)
    denominators = totals[..., 1::2]
    return np.divide(totals[..., ::2], denominators, out=np.full(denominators.shape, np.nan),
                     where=denominators > 0)


def summarize_pong(rows):
    if not rows:
        raise ValueError('a Pong summary requires at least one rally')
    outcomes = np.array([_outcomes(row) for row in rows])
    first_returns, first_opportunities, later_returns, later_opportunities, earned_points, starts = outcomes.sum(axis=0)
    fractions = {key: float(value) if np.isfinite(value) else None for key, value in
                 zip(RETENTION_METRICS, _fractions(outcomes, 0), strict=True)}
    opening = [row for row in rows if row.get('point', 1) == 1]
    return dict(starts=int(starts), matches=len(opening), first_opportunities=int(first_opportunities),
                first_returns=int(first_returns), later_opportunities=int(later_opportunities),
                later_returns=int(later_returns), earned_points=int(earned_points),
                rallies_with_later_return=int((outcomes[:, 2] > 0).sum()),
                unreturned_positive_scores=sum(row['score'] > 0 and not row['contacts'] for row in rows),
                losses=sum(row['score'] < 0 for row in rows), timeouts=sum(row['score'] == 0 for row in rows),
                action_map_bound=sum(row['visual_effects'] is not None for row in rows),
                relation_executed=sum(row['relation_executed'] for row in rows), **fractions)


def paired_pong(reference, challenger, *, draws=8192, seed=838564335):
    """Compare two runs on identical starts, including every missed first approach."""
    cases = [(row.get('match', row['case']), row.get('point', 1), row['case']) for row in reference]
    if not cases or cases != [(row.get('match', row['case']), row.get('point', 1), row['case'])
                              for row in challenger]:
        raise ValueError('paired Pong requires identical ordered native starts and remaps')
    first_ticks = lambda row: row['contacts'][0]['tick'] if row['contacts'] else None
    if any(first_ticks(a) != first_ticks(b) for a, b in zip(reference, challenger, strict=True)):
        raise ValueError('first incoming trajectory changed before possible physical contact')
    match_ids = [state_sha256(match) for match, _, _ in cases]
    matches = list(dict.fromkeys(match_ids))
    index_by_match = {match: index for index, match in enumerate(matches)}
    index = [index_by_match[match] for match in match_ids]
    outcomes = []
    for arm in (reference, challenger):
        totals = np.zeros((len(matches), 6), dtype=np.int64)
        np.add.at(totals, index, [_outcomes(row) for row in arm])
        outcomes.append(totals)
    indices = np.random.default_rng(seed).integers(len(matches), size=(draws, len(matches)))
    center = _fractions(outcomes[1], 0) - _fractions(outcomes[0], 0)
    differences = _fractions(outcomes[1][indices], 1) - _fractions(outcomes[0][indices], 1)
    estimates = {}
    for index, (key, threshold) in enumerate(RETENTION_METRICS.items()):
        metric = differences[:, index]
        valid = metric[np.isfinite(metric)]
        interval = np.quantile(valid, [.025, .975]).tolist() if len(valid) else None
        estimates[key] = dict(challenger_minus_reference=(float(center[index]) if np.isfinite(center[index]) else None),
                              serve_bootstrap_interval95=interval, available_draws=len(valid),
                              minimum_reference=threshold[0], maximum_decline=threshold[1])
    return dict(cases=cases, match_clusters=len(matches), bootstrap_draws=draws, metrics=estimates)


def assess_retention(before, after):
    """Require a capable pre-table player and paired evidence for retention."""
    pre = summarize_pong(before)
    post = summarize_pong(after)
    paired = paired_pong(before, after)
    baseline = dict(minimum_matches=paired['match_clusters'] >= MINIMUM_RETENTION_CASES,
                    later_opportunities=pre['later_opportunities'] >= MINIMUM_LATER_OPPORTUNITIES)
    noninferiority = dict(later_opportunities=post['later_opportunities'] >= MINIMUM_LATER_OPPORTUNITIES)
    for key, (minimum, margin) in RETENTION_METRICS.items():
        baseline[key] = pre[key] is not None and pre[key] >= minimum
        confidence = paired['metrics'][key]['serve_bootstrap_interval95']
        noninferiority[key] = (confidence is not None and confidence[0] >= -margin and
                               paired['metrics'][key]['available_draws'] == paired['bootstrap_draws'])
    opening_before = [row for row in before if row.get('point', 1) == 1]
    opening_after = [row for row in after if row.get('point', 1) == 1]
    return dict(before=pre, after=post, paired=paired, opening_before=summarize_pong(opening_before),
                opening_after=summarize_pong(opening_after), opening_paired=paired_pong(opening_before, opening_after),
                capable_baseline=baseline,
                noninferiority=noninferiority, established=all(baseline.values()) and all(noninferiority.values()))
