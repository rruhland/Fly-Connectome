"""Bind visible arcade roles from the camera and pass them to the Pong controller."""

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import ndimage
import torch

from fly_connectome.vision import VisualStateEncoder, load_default
from pong_relational_persistent_binding import PUBLIC_FIELDS


IMAGE_SIDE = 64
PIXEL = 1 / IMAGE_SIDE
CAMERA_SECONDS = .02


@dataclass(frozen=True)
class RoleBinding:
    """Map a camera rail to canonical left-paddle Pong coordinates."""

    axis: int
    axis_sign: int
    across_sign: int
    station: float

    def __post_init__(self):
        if (self.axis not in (0, 1) or self.across_sign not in (-1, 1)
                or self.axis_sign != self.across_sign * (-1 if self.axis else 1)
                or not .25 <= abs(self.station) <= .5
                or self.across_sign * self.station >= 0):
            raise ValueError('a proper quarter-turn onto a left visual rail is required')

    @classmethod
    def from_station(cls, axis, station):
        across_sign = 1 if station < 0 else -1
        return cls(axis, across_sign * (-1 if axis else 1), across_sign, station)

    def transform(self, point):
        """Transform a centered camera coordinate or displacement in [row, column] order."""
        return (self.axis_sign * point[self.axis], self.across_sign * point[1 - self.axis])

    def image(self, frame):
        aligned = frame.T if self.axis else frame
        if self.axis_sign < 0:
            aligned = aligned.flip(0)
        if self.across_sign < 0:
            aligned = aligned.flip(1)
        return aligned.contiguous()


NATIVE_PONG_BINDING = RoleBinding.from_station(0, -.453125)


@dataclass(frozen=True)
class Component:
    index: int
    position: tuple
    span: tuple
    box: tuple

    @property
    def rail(self):
        axis = int(self.span[1] > self.span[0])
        length, thickness = self.span[axis], self.span[1 - axis]
        station = self.position[1 - axis]
        if length >= 7 and thickness <= 4 and length >= 2.5 * thickness and abs(station) >= .25:
            # Preserve the station through a one-pixel bump when silhouettes touch.
            return axis, round(station / (2 * PIXEL)) * (2 * PIXEL)
        return None


def components(frame):
    """Segment the current public silhouette into four-connected components."""
    labels, _ = ndimage.label(frame.numpy())
    groups = []
    for index, box in enumerate(ndimage.find_objects(labels), 1):
        position = tuple((edge.start + edge.stop) * PIXEL / 2 - .5 for edge in box)
        span = tuple(edge.stop - edge.start for edge in box)
        groups.append(Component(index, position, span, tuple((edge.start, edge.stop) for edge in box)))
    return labels, groups


class RoleAdapter:
    """Find a rail that responds to opposite requests and a compact moving object.

    The isolated silhouettes are passed through the approved camera tracker. A frozen
    adapter keeps its supplied rail, or remains unbound if it has no prior rail.
    """

    def __init__(self, mode='online', binding=None):
        if mode not in ('online', 'frozen'):
            raise ValueError('role mode must be online or frozen')
        if binding is not None and not isinstance(binding, RoleBinding):
            raise ValueError('a RoleBinding is required')
        self.mode = mode
        self.binding = binding
        self.vision = load_default()
        self.encoder = VisualStateEncoder(IMAGE_SIDE, IMAGE_SIDE, CAMERA_SECONDS)
        self.reset()

    def reset(self):
        """Clear scene-local histories while retaining the current rail prior."""
        self._reset_camera()
        self._reset_probes()
        self.frame = None
        self.seconds = 0.
        self.binding_seconds = None
        self.probe_count = 0
        self._initial_rail_missing = 0
        self._rail_seen = False
        self._compact_boxes = set()
        self._ball = None
        self._ball_seen = -math.inf
        self._opponent_rails = {}
        self.opponent_visible = False

    def _reset_camera(self):
        self.vision.reset_state()
        self.encoder.reset_state()
        self.previous = torch.zeros(IMAGE_SIDE, IMAGE_SIDE, dtype=torch.bool)
        self.canonical_entities = []
        self.pair_ids = None
        self._role_ids = [None, None]

    def _reset_probes(self):
        self._rail_history = {}
        self._requests = deque(maxlen=80)

    @property
    def ready(self):
        return self.pair_ids is not None

    def state(self):
        return None if self.binding is None else asdict(self.binding)

    def probe_action(self):
        if self.mode == 'frozen' or self.binding is not None:
            return 1
        # An initial pulse from rest, followed by longer opposite pulses, makes
        # the visible reversal distinguishable from other moving objects.
        if self.probe_count < 6:
            return 0
        return (2, 0, 1, 0, 2, 1)[(self.probe_count - 6) // 14 % 6]

    def _bind(self, key, component):
        axis, _ = key
        self.binding = RoleBinding.from_station(axis, component.position[1 - axis])
        self.binding_seconds = self.seconds
        return component

    def _discover_rail(self, rails):
        for key, candidate in rails.items():
            axis, _ = key
            history = self._rail_history.setdefault(key, deque(maxlen=80))
            history.append((self.seconds, candidate.position[axis]))
        if self.mode != 'online' or self.binding is not None or len(self._requests) < 5:
            return None

        if len(rails) == 1:
            # With a single rail, visible displacement after energizing the
            # initial request identifies it without waiting for a second mover.
            key, component = next(iter(rails.items()))
            history = self._rail_history[key]
            if (len(history) >= 6 and any(action != 1 for _, action in self._requests)
                    and abs(history[-1][1] - history[0][1]) >= PIXEL / 2):
                return self._bind(key, component)

        # Compare the visible displacements before and after a real command
        # reversal. Either sign is admissible, including a remapped motor.
        switches = [time for (_, previous), (time, action)
                    in zip(self._requests, list(self._requests)[1:]) if action != previous]
        winners = []
        for key, candidate in rails.items():
            ranges = self._rail_history[key]
            for time in reversed(switches):
                before = [p for stamp, p in ranges if time - .30 <= stamp < time]
                after = [p for stamp, p in ranges if stamp >= time]
                if len(before) < 5 or len(after) < 5:
                    continue
                initial_motion = before[-1] - before[0]
                direction = math.copysign(1., initial_motion)
                reversed_motion = max(direction * (p - after[-1]) for p in after)
                if abs(initial_motion) >= PIXEL / 2 and reversed_motion >= PIXEL / 2:
                    winners.append((abs(initial_motion) + reversed_motion, key, candidate))
                    break
        if not winners:
            return None
        _, key, component = max(winners, key=lambda item: item[0])
        return self._bind(key, component)

    def _ball_component(self, groups):
        compact = [c for c in groups if max(c.span) <= 3]
        current = {component.box for component in compact}
        novel = [component for component in compact if component.box not in self._compact_boxes]
        self._compact_boxes = current
        if self._ball is not None:
            distance = lambda c: math.dist(c.position, self._ball)
            nearby = [c for c in compact if distance(c) <= max(2 * PIXEL, .9 * (self.seconds - self._ball_seen))]
            moved = [c for c in nearby if c in novel]
            if moved or nearby:
                component = min(moved or nearby, key=distance)
                if moved:
                    self._ball_seen = self.seconds
                if self.seconds - self._ball_seen <= .16:
                    self._ball = component.position
                    return component
            if self.seconds - self._ball_seen <= .16:
                return None
            self._ball = None
        if len(novel) == 1 and self.seconds > 0:
            component = novel[0]
            self._ball = component.position
            self._ball_seen = self.seconds
            return component
        return None

    def _track(self, labels, own, ball, opponent):
        pieces = tuple(component.index for component in (own, ball, opponent) if component is not None)
        mask = torch.from_numpy(np.isin(labels, pieces))
        frame = self.binding.image(mask)
        events = torch.stack((self.previous & ~frame, ~self.previous & frame)).float()
        self.previous = frame
        tracked = self.encoder.encode_entities(*self.vision.track_entities(events, frame.float()))
        present = {entity['id']: entity for entity in tracked}
        used = set()
        for index, component in enumerate((own, ball)):
            if component is not None:
                target = self.binding.transform(component.position)
                matched = [(math.dist(entity['normalized_position'], target), identity)
                           for identity, entity in present.items() if entity['observed'] and identity not in used]
                if matched:
                    distance, identity = min(matched)
                    if distance < 1.5 * PIXEL:
                        self._role_ids[index] = identity
            if self._role_ids[index] not in present or self._role_ids[index] in used:
                self._role_ids[index] = None
            elif self._role_ids[index] is not None:
                used.add(self._role_ids[index])
        if opponent is not None:
            target = self.binding.transform(opponent.position)
            used.update(identity for identity, entity in present.items() if identity not in used
                        and entity['observed'] and math.dist(entity['normalized_position'], target) < PIXEL)
        self.pair_ids = tuple(self._role_ids) if None not in self._role_ids else None
        self.canonical_entities = []
        for identity, entity in present.items():
            if identity not in used:
                continue
            row = {field: entity[field] for field in PUBLIC_FIELDS}
            for field in ('normalized_position', 'observed_velocity'):
                row[field] = row[field].tolist()
            self.canonical_entities.append(row)
        return self.canonical_entities

    def observe(self, frame, elapsed_seconds=0., action=None):
        if (not isinstance(frame, torch.Tensor) or frame.dtype != torch.bool
                or tuple(frame.shape) != (IMAGE_SIDE, IMAGE_SIDE) or frame.device.type != 'cpu'):
            raise ValueError('a binary 64×64 CPU camera frame is required')
        if self.frame is None:
            if action is not None or elapsed_seconds:
                raise ValueError('initial camera precedes requests and elapsed time')
        elif action not in (0, 1, 2) or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('the preceding requested action and public elapsed time are required')
        self.frame = frame
        self.seconds += elapsed_seconds
        if self.mode == 'online' and self.binding is None and action is not None:
            self._requests.append((self.seconds, action))
            self.probe_count += 1
        labels, groups = components(frame)
        rails = {component.rail: component for component in groups if component.rail is not None}
        for key, component in rails.items():
            along = component.position[key[0]]
            low, high = self._opponent_rails.get(key, (along, along))
            self._opponent_rails[key] = min(low, along), max(high, along)
        ball = self._ball_component(groups)
        own = None
        if self.binding is not None:
            axis, station = self.binding.axis, self.binding.station
            known = [c for (along, _), c in rails.items() if along == axis
                     and abs(c.position[1 - axis] - station) <= 1.5 * PIXEL]
            if known:
                own = min(known, key=lambda c: abs(c.position[1 - axis] - station))
                self._rail_seen = True
            elif not self._rail_seen and self.mode == 'online':
                self._initial_rail_missing += 1
                if self._initial_rail_missing >= 3:
                    self.binding = None
                    self._reset_camera()
        if self.binding is None:
            own = self._discover_rail(rails)
            if own is None:
                return []
            self._rail_seen = True
        axis, station = self.binding.axis, self.binding.station
        opposite = [c for key, c in rails.items() if key[0] == axis
                    and c.position[1 - axis] * station < 0
                    and abs(c.position[1 - axis]) >= .33
                    and (self._opponent_rails[key][1] - self._opponent_rails[key][0] >= PIXEL)]
        opponent = max(opposite, key=lambda c: c.span[axis], default=None)
        self.opponent_visible = opponent is not None
        return self._track(labels, own, ball, opponent)


class RoleController:
    """Connect a life-local Pong controller to public camera transitions."""

    def __init__(self, controller, detector):
        self.controller = controller
        self.detector = detector
        self.timing = getattr(controller, 'timing', controller)
        self.request = None
        self.engaged = False
        self.total_reward = 0.

    def reset(self, frame):
        self.detector.reset()
        self.detector.observe(frame)
        self.request = None
        self.engaged = False
        self.total_reward = 0.

    def choose(self, baseline_action=1):
        if self.request is not None:
            raise ValueError('observe the actual camera transition before choosing again')
        if not self.engaged and not self.detector.ready:
            self.request = self.detector.probe_action()
            return self.request, dict(action=self.request, probing=self.detector.binding is None
                                      and self.detector.mode == 'online')
        self.engaged = True
        self.timing.inherited.bind_public_pair(self.detector.pair_ids)
        self.request, decision = self.controller.choose(self.detector.canonical_entities, baseline_action)
        return self.request, decision

    def observe(self, transition):
        if self.request is None:
            raise ValueError('choose the actual request before observing the transition')
        entities = self.detector.observe(transition.frame, transition.elapsed_seconds, self.request)
        self.request = None
        self.total_reward += transition.reward
        if self.engaged:
            done = transition.terminated or transition.truncated
            score = int(self.total_reward > 0) - int(self.total_reward < 0) if done else 0
            self.controller.observe(public_entities=entities, own_hits=0, reward=score,
                                    terminated=transition.terminated, truncated=transition.truncated,
                                    elapsed_seconds=transition.elapsed_seconds)
