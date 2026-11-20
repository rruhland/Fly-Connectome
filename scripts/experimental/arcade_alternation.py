"""Use visible scene structure to attach reward learning to a shared paddle servo."""

import copy
from dataclasses import asdict, dataclass

import numpy as np

from arcade_transfer_session import TransferSession
from breakout_brick_placement import BrickPlacement, BrickShotPolicy, canonical_camera, visible_targets
from breakout_visual_rebound import SCHEMA as REBOUND_SCHEMA, VisualReboundPlacement, VisualReboundPolicy
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
    def __init__(self, session):
        self.life = None
        self.session = session
        self.evidence = None
        self.placement = None
        self.recalled_role = False
        self.recalled_timing = False
        self.frame = None
        self.start_timing = None
        self.paddle_extent = None
        self.requests = set()

    @property
    def detector(self):
        return self.life.detector

    @property
    def timing(self):
        return self.life.controller.timing

    def reset(self, frame):
        if self.life is not None:
            raise ValueError('finish the current life before resetting')
        self.evidence = scene_evidence(frame)
        appearance = self.evidence.appearance
        self.frame = frame
        prior = self.session.roles.get(appearance)
        if self.session.recall_roles and prior is not None:
            _, groups = components(frame)
            if any(component.rail is not None and component.rail[0] == prior.axis
                   and abs(component.position[1 - prior.axis] - prior.station) <= 1.5 / 64
                   for component in groups):
                self.session.core.binding = prior
                self.recalled_role = True
        timing_prior = self.session.timings.get(appearance)
        if (self.session.recall_timing and self.recalled_role and timing_prior is not None
                and appearance != self.session.previous_appearance):
            self.session.core.timing = copy.deepcopy(timing_prior)
            self.recalled_timing = True
        self.start_timing = copy.deepcopy(self.session.core.timing)
        self.life = self.session.core.begin_life()
        if appearance != 'opposing_rails':
            self.life.bridge.controller = _MotorPath(self.life.controller)
        if appearance == 'target_array':
            seed = int(self.session.brick_rng.integers(2**31))
            policy = self.session.brick_policy
            placement = VisualReboundPlacement if isinstance(policy, VisualReboundPolicy) else BrickPlacement
            self.placement = placement(self.timing, policy, seed)
        self.life.reset(frame)

    def choose(self, baseline_action=1):
        if isinstance(self.placement, VisualReboundPlacement):
            self.placement.camera(self.frame, self.detector)
        elif self.placement is not None:
            self.placement.camera(*canonical_camera(self.frame, self.detector))
        action, decision = self.life.choose(baseline_action)
        if self.session.recall_timing:
            self.requests.add(action)
        return action, decision

    def observe(self, transition):
        self.life.observe(transition)
        self.frame = transition.frame
        score = max(0., transition.reward)
        if isinstance(self.placement, VisualReboundPlacement):
            self.placement.observe(self.detector.seconds, self.detector.canonical_entities,
                                   self.detector.pair_ids, score, terminated=transition.terminated,
                                   truncated=transition.truncated)
        elif self.placement is not None:
            # Targets use positive per-removal increments. A terminal miss may
            # be zero or negative independently of the already earned score.
            self.placement.observe_score(score, terminated=transition.terminated,
                                         truncated=transition.truncated)
        if self.session.recall_timing and self.detector.pair_ids is not None:
            own_id = self.detector.pair_ids[0]
            own = next((row for row in self.detector.canonical_entities if row['id'] == own_id and row['observed']), None)
            if own is not None:
                position = own['normalized_position'][0]
                low, high = self.paddle_extent or (position, position)
                self.paddle_extent = min(low, position), max(high, position)

    def timing_supported(self):
        belief = self.timing.belief
        return (belief.states is not None and belief.probabilities[belief.confirmed] >= .85
                and self.paddle_extent is not None and self.paddle_extent[1] - self.paddle_extent[0] >= 2 / 64
                and len(self.requests) >= 2)


class AlternatingSession:
    """Carry one online motor model and two sensor-selected placement memories."""

    def __init__(self, policy_seed, *, checkpoint=None, brick_state=None, recall_roles=False,
                 recall_timing=False, causal_pursuit=False):
        core_rng, brick_rng = np.random.SeedSequence(policy_seed).spawn(2)
        relation = {'causal_pursuit': True} if causal_pursuit else {}
        self.core = TransferSession('full', core_rng, checkpoint, **relation)
        self.brick_rng = np.random.default_rng(brick_rng)
        if brick_state is None:
            self.brick_policy = BrickShotPolicy()
        elif brick_state['schema'] == REBOUND_SCHEMA:
            self.brick_policy = VisualReboundPolicy.from_state(brick_state)
        else:
            self.brick_policy = BrickShotPolicy.from_state(brick_state)
        self.recall_timing = bool(recall_timing)
        self.recall_roles = bool(recall_roles) or self.recall_timing
        self.roles = {}
        self.timings = {}
        self.previous_appearance = None
        self.life = None

    def begin_life(self):
        if self.life is not None:
            raise ValueError('finish the current life before resetting')
        self.life = _SceneLife(self)
        return self.life

    def end_life(self):
        life = self.life
        self.core.end_life()
        appearance = life.evidence.appearance
        if (self.recall_roles and appearance != 'unresolved'
                and life.detector.binding is not None and life.detector.binding_seconds is not None):
            self.roles[appearance] = life.detector.binding
        if (self.recall_timing and appearance in self.roles
                and life.detector.binding == self.roles[appearance] and life.timing_supported()):
            self.timings[appearance] = copy.deepcopy(self.core.timing)
        self.previous_appearance = appearance
        self.life = None

    def state(self):
        return dict(core=self.core.state(), brick_policy=self.brick_policy.to_state(),
                    brick_rng=self.brick_rng.bit_generator.state, recall_roles=self.recall_roles,
                    recall_timing=self.recall_timing, previous_appearance=self.previous_appearance,
                    roles={scene: asdict(binding) for scene, binding in self.roles.items()},
                    timings=copy.deepcopy(self.timings))

    @classmethod
    def from_state(cls, state):
        session = cls(0, brick_state=state['brick_policy'], recall_roles=state['recall_roles'],
                      recall_timing=state['recall_timing'])
        session.core = TransferSession.from_state(state['core'])
        session.brick_rng.bit_generator.state = state['brick_rng']
        session.roles = {scene: RoleBinding(**binding) for scene, binding in state['roles'].items()}
        session.timings = copy.deepcopy(state['timings'])
        session.previous_appearance = state['previous_appearance']
        return session
