"""Use visible scene structure to attach reward learning to a shared paddle servo."""

from dataclasses import asdict, dataclass

import numpy as np

from arcade_transfer_session import TransferSession
from breakout_brick_placement import BrickPlacement, BrickShotPolicy, canonical_camera, visible_targets
from pong_role_transfer import RoleBinding, components


@dataclass(frozen=True)
class SceneEvidence:
    appearance: str
    target_rectangles: int
    opposing_rails: bool


def scene_evidence(frame):
    """Find a board of small rectangles or two opposed long rails in public pixels."""
    _, groups = components(frame)
    rails = {component.rail: component for component in groups if component.rail is not None}
    opposing_rails = False
    target_rectangles = 0
    for axis, station in rails:
        opposing_rails |= any(
            other_axis == axis and station * other_station < 0 and abs(station - other_station) >= .65
            for other_axis, other_station in rails
        )
        binding = RoleBinding.from_station(axis, station)
        targets = visible_targets(binding.image(frame), binding.across_sign * station)
        if len(targets):
            _, row_sizes = np.unique(targets[:, 1], return_counts=True)
            if np.count_nonzero(row_sizes >= 2) >= 2:
                target_rectangles = max(target_rectangles, len(targets))
    if target_rectangles and not opposing_rails:
        appearance = 'target_array'
    elif opposing_rails and not target_rectangles:
        appearance = 'opposing_rails'
    else:
        appearance = 'unresolved'
    return SceneEvidence(appearance, target_rectangles, opposing_rails)


class _MotorPath:
    """Use the same timing controller without giving this flight to Pong placement."""

    def __init__(self, original):
        self.timing = original.timing
        self.shot_memory = original.shot_memory
        self.timing.contact_offset = 0.

    def choose(self, entities, baseline_action):
        return self.timing.choose(entities, baseline_action)

    def observe(self, **transition):
        self.timing.observe(**transition)


class _SceneLife:
    def __init__(self, life, session):
        self.life = life
        self.session = session
        self.evidence = None
        self.placement = None
        self.recalled_role = False
        self.frame = None

    @property
    def detector(self):
        return self.life.detector

    @property
    def timing(self):
        return self.life.controller.timing

    def reset(self, frame):
        self.evidence = scene_evidence(frame)
        appearance = self.evidence.appearance
        self.frame = frame
        prior = self.session.roles.get(appearance)
        if self.session.recall_roles and prior is not None:
            _, groups = components(frame)
            if any(component.rail is not None and component.rail[0] == prior.axis
                   and abs(component.position[1 - prior.axis] - prior.station) <= 1.5 / 64
                   for component in groups):
                self.detector.binding = prior
                self.recalled_role = True
        if appearance != 'opposing_rails':
            self.life.bridge.controller = _MotorPath(self.life.controller)
        if appearance == 'target_array':
            seed = int(self.session.brick_rng.integers(2**31))
            self.placement = BrickPlacement(self.timing, self.session.brick_policy, seed)
        self.life.reset(frame)

    def choose(self, baseline_action=1):
        if self.placement is not None:
            self.placement.camera(*canonical_camera(self.frame, self.detector))
        return self.life.choose(baseline_action)

    def observe(self, transition):
        self.life.observe(transition)
        self.frame = transition.frame
        if self.placement is not None:
            # Targets use positive per-removal increments. A terminal miss may
            # be zero or negative independently of the already earned score.
            self.placement.observe_score(max(0., transition.reward), terminated=transition.terminated,
                                         truncated=transition.truncated)


class AlternatingSession:
    """Carry one online motor model and two sensor-selected placement memories."""

    def __init__(self, policy_seed, *, checkpoint=None, brick_state=None, recall_roles=False):
        core_rng, brick_rng = np.random.SeedSequence(policy_seed).spawn(2)
        self.core = TransferSession('full', core_rng, checkpoint)
        self.brick_rng = np.random.default_rng(brick_rng)
        self.brick_policy = (BrickShotPolicy() if brick_state is None
                             else BrickShotPolicy.from_state(brick_state))
        self.recall_roles = bool(recall_roles)
        self.roles = {}
        self.life = None

    def begin_life(self):
        self.life = _SceneLife(self.core.begin_life(), self)
        return self.life

    def end_life(self):
        life = self.life
        self.core.end_life()
        if (self.recall_roles and life.evidence.appearance != 'unresolved'
                and life.detector.binding is not None and life.detector.binding_seconds is not None):
            self.roles[life.evidence.appearance] = life.detector.binding
        self.life = None

    def state(self):
        return dict(core=self.core.state(), brick_policy=self.brick_policy.to_state(),
                    brick_rng=self.brick_rng.bit_generator.state, recall_roles=self.recall_roles,
                    roles={scene: asdict(binding) for scene, binding in self.roles.items()})

    @classmethod
    def from_state(cls, state):
        session = cls(0, brick_state=state['brick_policy'], recall_roles=state['recall_roles'])
        session.core = TransferSession.from_state(state['core'])
        session.brick_rng.bit_generator.state = state['brick_rng']
        session.roles = {scene: RoleBinding(**binding) for scene, binding in state['roles'].items()}
        return session
