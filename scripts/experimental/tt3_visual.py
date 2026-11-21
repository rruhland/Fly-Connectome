"""Learn visual roles from unannotated pixels, requested actions and scalar returns."""
from collections import deque
from dataclasses import dataclass
import os

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment, nnls


@dataclass(frozen=True)
class Sample:
    frames: object
    preceding_action: object = None
    reward: float = 0.


@dataclass(frozen=True)
class Track:
    id: int
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    observed: bool = True
    age: int = 1


@dataclass(frozen=True)
class Binding:
    controlled: Track | None
    relevant: Track | None
    tracks: tuple[Track, ...] = ()
    controlled_evidence: float = 0.
    relevant_evidence: float = 0.


@dataclass(frozen=True)
class _Detection:
    position: np.ndarray
    color: int
    area: int


_NATIVE_COMPONENTS = os.environ.get('TT3_COMPONENT_LIBRARY')


def _components(frame):
    frame = np.asarray(frame)
    if _NATIVE_COMPONENTS and frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8:
        try:
            from .tt3_pixels import regions
        except ImportError:
            from tt3_pixels import regions
        records = regions(frame, _NATIVE_COMPONENTS)
        height, width = frame.shape[:2]
        centers = (records[:, :2] - np.array((width, height))) / (2 * width)
        return [_Detection(center, int(color), int(area))
                for center, (_, _, color, area) in zip(centers, records)]
    return _reference_components(frame)


def _reference_components(frame):
    """Group the pixels of each locally constant appearance into hypotheses."""
    frame = np.asarray(frame)
    if frame.ndim == 2:
        pixels = frame
    elif frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8:
        pixels = frame[..., 0].astype(np.uint32) << 16
        pixels |= frame[..., 1].astype(np.uint32) << 8
        pixels |= frame[..., 2]
    else:
        raise ValueError('expected a grayscale image or uint8 RGB image')
    height, width = pixels.shape
    colors, counts = np.unique(pixels, return_counts=True)
    detections = []
    center = np.array((width, height)) / 2
    for color, count in zip(colors, counts):
        if count < 2 or count > .45 * pixels.size:
            continue
        labels, _ = ndimage.label(pixels == color)
        for index, box in enumerate(ndimage.find_objects(labels), start=1):
            y, x = box
            if ((y.start == 0 and y.stop == height) or (x.start == 0 and x.stop == width)):
                continue
            area = np.count_nonzero(labels[box] == index)
            if area < 2 or area > .10 * pixels.size:
                continue
            centroid = np.array(((x.start + x.stop) / 2, (y.start + y.stop) / 2))
            detections.append(_Detection((centroid - center) / width, int(color), area))
    return detections


class _Hypothesis:
    def __init__(self, index, detection, tick):
        self.id = index
        self.color = detection.color
        self.area = float(detection.area)
        self.history = deque(maxlen=5)
        self.age = 0
        self.observe(detection, tick)

    def observe(self, detection, tick):
        self.history.append((tick, detection.position))
        self.age += 1
        self.area = .8 * self.area + .2 * detection.area
        self.current_area = detection.area
        self.tick, self.position = self.history[-1]
        first_tick, first_position = self.history[0]
        self.velocity = (self.position - first_position) / max(1, tick - first_tick)

    def prediction(self, tick):
        return self.position + (tick - self.tick) * self.velocity

    def track(self, tick):
        return Track(self.id, tuple(self.prediction(tick)), tuple(self.velocity),
                     self.tick == tick, self.age)


class _CameraTracker:
    def __init__(self):
        self.hypotheses = {}
        self.tick = -1
        self.count = 0

    def update(self, frame):
        self.tick += 1
        detections = _components(frame)
        active = list(self.hypotheses.values())
        used = set()
        if active and detections:
            old = np.array([item.prediction(self.tick) for item in active])
            new = np.array([item.position for item in detections])
            distance = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=-1)
            appearance = np.array([item.color for item in active])[:, None] != np.array(
                [item.color for item in detections])[None, :]
            old_size = np.array([item.area for item in active])[:, None]
            new_size = np.array([item.area for item in detections])[None, :]
            size = np.abs(np.log(new_size / old_size))
            cost = distance + .004 * size + .08 * appearance
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if distance[row, column] <= .045 and cost[row, column] <= .052:
                    active[row].observe(detections[column], self.tick)
                    used.add(column)
        for index, detection in enumerate(detections):
            if index not in used:
                self.hypotheses[self.count] = _Hypothesis(self.count, detection, self.tick)
                self.count += 1
        self.hypotheses = {key: value for key, value in self.hypotheses.items()
                           if self.tick - value.tick <= 5}
        return tuple(self.hypotheses.values())


class VisualTracker:
    """Track all local components in one image or two registered orthographic images.

    For a camera pair, both horizontal image axes register the same physical dimension.
    Position is expressed in image widths about the center of each relevant image axis.
    Identity and short-term appearance are episodic and never part of the fitted model.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._top = _CameraTracker()
        self._side = _CameraTracker()
        self._pairs = {}
        self.support = {}

    def update(self, frames):
        if not isinstance(frames, (tuple, list)):
            tracks = self._top.update(frames)
            self.support = {item.id: (item.current_area, item.area) for item in tracks}
            return tuple(item.track(self._top.tick) for item in tracks)
        if len(frames) != 2:
            raise ValueError('one image or exactly two registered images are required')
        first, second = frames
        top = self._top.update(first)
        side = self._side.update(second)
        tick = self._top.tick
        self.support = {item.id: (item.current_area, item.area) for item in top}
        if not top or not side:
            return ()
        top_x = np.array([item.prediction(tick)[0] for item in top])
        side_x = np.array([item.prediction(tick)[0] for item in side])
        top_v = np.array([item.velocity[0] for item in top])
        side_v = np.array([item.velocity[0] for item in side])
        horizontal = np.abs(top_x[:, None] - side_x[None, :])
        appearance = np.array([item.color for item in top])[:, None] != np.array(
            [item.color for item in side])[None, :]
        continuity = np.array([self._pairs.get(item.id, -1) for item in top])[:, None] != np.array(
            [item.id for item in side])[None, :]
        cost = horizontal + 2 * np.abs(top_v[:, None] - side_v[None, :])
        cost += .055 * appearance + .008 * continuity
        rows, columns = linear_sum_assignment(cost)
        pairs = {}
        tracks = []
        for row, column in zip(rows, columns):
            if horizontal[row, column] > .043 or cost[row, column] > .073:
                continue
            first_item, second_item = top[row], side[column]
            a, b = first_item.track(tick), second_item.track(tick)
            pairs[a.id] = b.id
            tracks.append(Track(a.id, ((a.position[0] + b.position[0]) / 2,
                                       a.position[1], b.position[1]),
                                ((a.velocity[0] + b.velocity[0]) / 2,
                                 a.velocity[1], b.velocity[1]),
                                a.observed and b.observed, min(a.age, b.age)))
        self._pairs = pairs
        return tuple(tracks)


@dataclass(frozen=True)
class _ActionEvidence:
    features: np.ndarray
    effect: np.ndarray


class _SceneHistory:
    def __init__(self):
        self.tracker = VisualTracker()
        self.tick = -1
        self.actions = deque(maxlen=72)
        self.paths = {}

    def update(self, frames, action):
        self.tick += 1
        value = None if action is None else np.asarray(action, dtype=float).reshape(-1)
        if value is not None and value.size == 1 and value[0] in (-1, 0, 1):
            value = np.eye(3)[int(value[0]) + 1]
        self.actions.append((self.tick, value))
        tracks = self.tracker.update(frames)
        for track in tracks:
            if track.observed:
                history = self.paths.setdefault(track.id, deque(maxlen=64))
                history.append((self.tick, np.asarray(track.position)))
        if self.tick % 24 == 0:
            self.paths = {index: path for index, path in self.paths.items()
                          if path[-1][0] >= self.tick - 64}
        return tracks

    def effect(self, track):
        if track.id not in self.paths or len(self.paths[track.id]) < 9:
            return None
        actions = [value for _, value in self.actions if value is not None]
        if len(actions) < 8:
            return None
        if not np.any(np.std(actions, axis=0) > 1e-5):
            return None
        start = self.actions[0][0]
        values = np.stack([value if value is not None else np.zeros_like(actions[0])
                           for _, value in self.actions])
        times, positions = zip(*self.paths[track.id])
        times = np.asarray(times)
        positions = np.array(positions)
        # Estimate a response above smooth camera drift at several temporal scales.
        # Reverse command order to measure association remaining without causality.
        cumulative = np.concatenate((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
        current, comparison, effects = [], [], []
        for step, width in ((1, 1), (2, 3), (4, 8), (6, 12)):
            local = times[step:] - start
            delta_time = times[step:] - times[:-step]
            valid = (local >= 1) & (local < len(values)) & (delta_time <= step + 3)
            if valid.sum() < 6:
                continue
            local = local[valid]
            left = np.maximum(0, local + 1 - width)
            history = (cumulative[local + 1] - cumulative[left]) / (local + 1 - left)[:, None]
            t = np.linspace(-1, 1, len(history))
            drift, _ = np.linalg.qr(np.column_stack((np.ones_like(t), t)))
            speed = ((positions[step:] - positions[:-step]) / delta_time[:, None])[valid]
            speed -= drift @ (drift.T @ speed)
            variance = float(np.sum(speed * speed))
            if variance <= 1e-10:
                continue
            response = history - drift @ (drift.T @ history)
            fitted, effect = _predictability(response, speed, variance)
            reverse, _ = _predictability(response[::-1], speed, variance)
            current.append(fitted)
            comparison.append(reverse)
            effects.append(effect)
        if not effects:
            return None
        strength = np.array([np.linalg.norm(effect) for effect in effects])
        best = int(np.argmax(np.asarray(current) * strength))
        gain = float(np.linalg.norm(effects[best]))
        features = np.array((current[best], max(0., current[best] - comparison[best]), gain))
        return _ActionEvidence(features, effects[best])

    def motion(self, track, controlled=None, effect=None, tick=None):
        path = self.paths.get(track.id)
        if path is None:
            return np.zeros(5)
        recent = [(time, position) for time, position in path if tick is None or time <= tick][-14:]
        if len(recent) < 5:
            return np.zeros(5)
        times, positions = zip(*recent)
        positions = np.asarray(positions)
        segment = np.diff(positions, axis=0)
        travel = float(np.linalg.norm(segment, axis=1).sum())
        move = (positions[-1] - positions[0]) / (times[-1] - times[0])
        speed = float(np.linalg.norm(move))
        straight = np.clip(speed * (times[-1] - times[0]) / max(travel, 1e-8), 0, 1)
        independent = move
        distance = positions[-1]
        if controlled is not None:
            distance = distance - np.asarray(controlled.position)
        if effect is not None and np.linalg.norm(effect) > 1e-7:
            _, singular, vectors = np.linalg.svd(effect, full_matrices=False)
            span = vectors[singular >= singular[0] * .25]
            independent = move - (move @ span.T) @ span
            distance = distance - (distance @ span.T) @ span
        across = float(np.linalg.norm(independent))
        progress = abs(float(independent @ distance)) / max(float(np.linalg.norm(distance)), 1e-6)
        return np.array((speed, speed * straight, across, progress,
                         across * across / max(speed, 1e-8)))


def _predictability(requests, motion, variance):
    centered = requests - requests.mean(axis=0)
    gram = centered.T @ centered + np.eye(centered.shape[1]) * 1.5
    effect = np.linalg.solve(gram, centered.T @ motion)
    residual = motion - centered @ effect
    explained = 1 - np.sum(residual * residual) / variance
    return float(np.clip(explained, 0, 1)), effect


def _scaled(rows):
    rows = np.asarray(rows)
    # Only a common scale within a scene transfers between frame rates and magnifications.
    return rows / np.maximum(rows.max(axis=0), 1e-8)


def _control_rows(rows):
    scaled = _scaled(rows)
    scaled[:, :2] *= scaled[:, 2:]
    return scaled


def _fit_ranking(differences, nonnegative=False, prior=None, prior_count=0):
    rows = np.asarray(differences, dtype=float)
    identity = np.eye(rows.shape[1])
    penalty = 2. + .25 * prior_count
    center = np.zeros(rows.shape[1]) if prior is None else np.asarray(prior)
    if nonnegative:
        features = np.vstack((rows, np.sqrt(penalty) * identity))
        targets = np.concatenate((np.ones(len(rows)), np.sqrt(penalty) * center))
        coefficients, _ = nnls(features, targets)
    else:
        coefficients = np.linalg.solve(rows.T @ rows + penalty * identity,
                                       rows.sum(axis=0) + penalty * center)
    return tuple(map(float, coefficients))


class _RoleEvidence:
    def __init__(self):
        self.control = []
        self.relevance = []
        self.null_control = []

    def observe(self, scene, tracks, history, image_shape, reward=0.):
        candidates, evidence = _effects(scene, tracks)
        actor = None
        actor_effect = None
        if candidates:
            rows = np.array([item.features for item in evidence])
            normalized = _control_rows(rows)
            contrast = rows[:, 0] * rows[:, 1] * normalized[:, 2]
            winner = int(np.argmax(contrast))
            rivals = np.arange(len(candidates)) != winner
            if contrast[winner] > .02 and contrast[winner] > 1.8 * max(contrast[rivals], default=0.):
                negatives = normalized[rivals] if len(candidates) > 1 else np.zeros((1, rows.shape[1]))
                self.control.extend(normalized[winner] - negatives)
                self.null_control.extend(rows[rivals, 0])
            actor, actor_effect = candidates[winner], evidence[winner].effect
        if reward:
            self._observe_relevance(scene, tracks, history, image_shape, actor, actor_effect)

    def _observe_relevance(self, scene, tracks, history, image_shape, actor, actor_effect):
        witness = _reward_witness(history, image_shape)
        if witness is None:
            return
        if actor is not None and actor.id == witness.id:
            actor = actor_effect = None
        candidates = [track for track in tracks
                      if track.age >= 5 and (actor is None or track.id != actor.id)]
        if all(track.id != witness.id for track in candidates):
            candidates.append(witness)
        if len(candidates) < 2:
            return
        winner = next(index for index, track in enumerate(candidates) if track.id == witness.id)
        # Frame boundaries witness the event and are omitted from the role features.
        for tick, snapshot, _ in list(history)[-5::2]:
            current = {track.id: track for track in snapshot}
            if witness.id not in current:
                continue
            historical_actor = None if actor is None else current.get(actor.id)
            rows = _scaled([scene.motion(current.get(track.id, track), historical_actor,
                                         actor_effect, tick) for track in candidates])
            self.relevance.extend(rows[winner] - np.delete(rows, winner, axis=0))

    def model(self, prior=None):
        prior = RoleModel.untrained() if prior is None else prior
        control = prior.control_weights
        relevance = prior.relevance_weights
        floor = prior.control_floor
        if self.control:
            control = _fit_ranking(self.control, nonnegative=True, prior=control,
                                   prior_count=prior.control_examples)
        if self.relevance:
            relevance = _fit_ranking(self.relevance, prior=relevance,
                                     prior_count=prior.reward_examples)
        if self.null_control:
            floor = float(np.clip(np.quantile(self.null_control, .80), .09, .33))
        return RoleModel(control, relevance, floor, prior.source_control_examples,
                         prior.source_reward_examples,
                         prior.receiving_control_examples + len(self.control),
                         prior.receiving_reward_examples + len(self.relevance))


@dataclass(frozen=True)
class RoleModel:
    """Portable numeric comparisons fitted without object annotations."""

    control_weights: tuple[float, ...]
    relevance_weights: tuple[float, ...]
    control_floor: float
    source_control_examples: int
    source_reward_examples: int
    receiving_control_examples: int = 0
    receiving_reward_examples: int = 0

    def __post_init__(self):
        for name, width in (('control_weights', 3), ('relevance_weights', 5)):
            weights = tuple(map(float, getattr(self, name)))
            if len(weights) != width or not np.isfinite(weights).all():
                raise ValueError(f'{width} finite {name} required')
            object.__setattr__(self, name, weights)

    @property
    def control_examples(self):
        return self.source_control_examples + self.receiving_control_examples

    @property
    def reward_examples(self):
        return self.source_reward_examples + self.receiving_reward_examples

    @classmethod
    def untrained(cls):
        return cls((0., 0., 0.), (0., 0., 0., 0., 0.), .22, 0, 0)

    @classmethod
    def fit(cls, episodes):
        bank = _RoleEvidence()
        for episode in episodes:
            scene = _SceneHistory()
            history = deque(maxlen=9)
            for sample in episode:
                tracks = scene.update(sample.frames, sample.preceding_action)
                history.append((scene.tick, tracks, scene.tracker.support))
                if scene.tick > 15 and (scene.tick % 16 == 0 or sample.reward):
                    shape = _image_shape(sample.frames)
                    bank.observe(scene, tracks, history, shape, sample.reward)
        if not bank.control or not bank.relevance:
            raise ValueError('role learning needs independently excited motions and visible reward events')
        model = bank.model()
        return cls(model.control_weights, model.relevance_weights, model.control_floor,
                   model.receiving_control_examples, model.receiving_reward_examples)


def _effects(scene, tracks):
    candidates, evidence = [], []
    for track in tracks:
        effect = scene.effect(track)
        if effect is not None:
            candidates.append(track)
            evidence.append(effect)
    return candidates, evidence


def _image_shape(frames):
    image = frames[0] if isinstance(frames, (tuple, list)) else frames
    return image.shape[:2]


def _reward_witness(history, shape):
    if len(history) < 6:
        return None
    terminal, latest, support = history[-1]
    height, width = shape
    outer = np.array((.5, height / (2 * width)))
    recent = {track.id: track for _, tracks, _ in history for track in tracks
              if track.observed and track.age >= 7}
    currently_seen = {track.id for track in latest if track.observed}
    winners = []
    for track in recent.values():
        position, speed = np.asarray(track.position[:2]), np.asarray(track.velocity[:2])
        border = outer - np.abs(position)
        axis = int(np.argmin(border))
        outward = position[axis] * speed[axis] > 0
        lost = track.id not in currently_seen
        area, expected = support.get(track.id, (1, 1))
        sliced = area < .90 * expected
        if outward and border[axis] < max(4 / width, 4 * abs(speed[axis])):
            if lost or sliced or border[axis] < 3 / width:
                winners.append(track)
    if len(winners) == 1 and terminal > 5:
        return winners[0]
    return None


class RoleBinder:
    """Apply the same learned comparisons while discovering scene-local identities."""
    def __init__(self, model, *, learn=False):
        self.model = model
        self._initial = model
        self._bank = _RoleEvidence() if learn else None
        self.reset()

    def reset(self):
        self._scene = _SceneHistory()
        self._control_id = None
        self._relevant_id = None
        self._effect = None
        self._control_score = 0.
        self._task_scores = {}
        self._proposed_control = None
        self._proposal_ticks = 0
        self._recent = deque(maxlen=9)

    def update(self, frames, previous_action=None):
        tracks = self._scene.update(frames, previous_action)
        by_id = {track.id: track for track in tracks}
        if self._bank is not None:
            self._image_shape = _image_shape(frames)
            self._recent.append((self._scene.tick, tracks, self._scene.tracker.support))
            if self._scene.tick >= 8 and self._scene.tick % 8 == 0:
                self._learn(tracks)
        if self._scene.tick % 3 == 2:
            self._compare_control(tracks)
        controlled = by_id.get(self._control_id)
        candidates = [track for track in tracks if track.age >= 5 and track.id != self._control_id]
        relevant, evidence = self._compare_relevance(candidates, controlled)
        return Binding(controlled, relevant, tracks, self._control_score, evidence)

    def observe_reward(self, reward):
        """Assimilate a scalar event when this binder is configured to learn."""
        if reward and self._bank is not None and self._recent:
            self._learn(self._recent[-1][1], reward)

    def _learn(self, tracks, reward=0.):
        initial_counts = len(self._bank.control), len(self._bank.relevance)
        self._bank.observe(self._scene, tracks, self._recent, self._image_shape, reward)
        if initial_counts != (len(self._bank.control), len(self._bank.relevance)):
            self.model = self._bank.model(self._initial)

    def _compare_control(self, tracks):
        candidates, evidence = _effects(self._scene, tracks)
        if not candidates:
            return
        rows = np.array([item.features for item in evidence])
        scores = _control_rows(rows) @ self.model.control_weights
        winner = int(np.argmax(scores))
        candidate, item = candidates[winner], evidence[winner]
        runner_up = max((score for index, score in enumerate(scores) if index != winner), default=0.)
        margin = float(scores[winner] - runner_up)
        threshold = self.model.control_floor
        if item.features[0] < threshold or item.features[1] < .022 or margin <= .035:
            self._proposed_control = None
            self._proposal_ticks = 0
            return
        if candidate.id == self._proposed_control:
            self._proposal_ticks += 1
        else:
            self._proposed_control = candidate.id
            self._proposal_ticks = 1
        control_visible = any(track.id == self._control_id for track in tracks)
        changing = candidate.id != self._control_id and control_visible
        if changing and (self._proposal_ticks < 3 or margin < .13):
            return
        if candidate.id == self._control_id and self._effect is not None and self._effect.shape == item.effect.shape:
            self._effect = .65 * self._effect + .35 * item.effect
        else:
            self._effect = item.effect
        self._control_id = candidate.id
        self._control_score = margin

    def _compare_relevance(self, candidates, controlled):
        if not candidates:
            return None, 0.
        rows = np.array([self._scene.motion(track, controlled, self._effect) for track in candidates])
        scores = _scaled(rows) @ self.model.relevance_weights
        scores[rows[:, 0] < .0004] = 0.
        smooth = np.array([.70 * self._task_scores.get(track.id, 0.) + .30 * score
                           for track, score in zip(candidates, scores)])
        self._task_scores = {track.id: score for track, score in zip(candidates, smooth)}
        winner = int(np.argmax(smooth))
        previous = next((i for i, track in enumerate(candidates) if track.id == self._relevant_id), None)
        if previous is not None and smooth[previous] >= smooth[winner] - .09:
            winner = previous
        runner_up = max((score for index, score in enumerate(smooth) if index != winner), default=0.)
        if smooth[winner] <= .05:
            return None, 0.
        self._relevant_id = candidates[winner].id
        return candidates[winner], float(smooth[winner] - runner_up)
