"""Carry the same sensorimotor learner through unlabeled paddle-game lives."""
import copy
import json
from pathlib import Path

import numpy as np

from pong_carried_shot_value import CarriedShotPlacement, ShotMemory
from pong_rapid_motor_recovery import ORDERS
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_score_placement import ScorePlacement
from pong_timing_controller import SCHEMA as TIMING_SCHEMA, TimedMotorController
from pong_visual_button_effects import VisualButtonEffects

CHECKPOINT = Path(__file__).resolve().parents[2] / 'artifacts/m1b/2026-11-19-pong-after-mixed-course.json'
ARMS = ('full', 'motor_arrival', 'frozen', 'frozen_orientation', 'fresh', 'fresh_visual', 'passive', 'center')


class _Passive:
    def reset(self, frame):
        pass

    def choose(self, baseline_action=1):
        return 1, {}

    def observe(self, transition):
        pass


class _Life:
    def __init__(self, bridge):
        self.bridge = bridge
        self.seconds = 0.
        self.earned = 0.
        self.ended = False

    def reset(self, frame):
        self.bridge.reset(frame)

    def choose(self, baseline_action=1):
        return self.bridge.choose(baseline_action)

    def observe(self, transition):
        self.seconds += transition.elapsed_seconds
        self.earned += max(0., transition.reward)
        self.ended = transition.terminated or transition.truncated
        self.bridge.observe(transition)

    @property
    def controller(self):
        return self.bridge.controller

    @property
    def detector(self):
        return self.bridge.detector


class TransferSession:
    """Warm and reset views of a public visual relation, motor map and shot learner."""

    def __init__(self, arm, policy_seed, checkpoint=None):
        if arm not in ARMS:
            raise ValueError(f'unknown transfer arm {arm!r}')
        self.arm = arm
        self.rng = np.random.default_rng(policy_seed)
        self.life = None
        self.lives = 0
        self.cold = None
        if arm == 'fresh_visual':
            from arcade_cold_visual import ColdVisualSession
            self.cold = ColdVisualSession(policy_seed)
        if arm in ('passive', 'fresh_visual'):
            return

        self.frozen = arm in ('frozen', 'frozen_orientation')
        self.relation_evidence = [[], []]
        self.binding = None
        if arm == 'fresh':
            # This matched reset keeps the numerical model family's built-in
            # response templates. The separate fresh_visual arm fits its own.
            self.polarity = int(self.rng.choice([-1, 1]))
            self.policy = dict(choices=[3, 0, 1, 0, self.polarity], coefficients=[0.] * 5,
                               scales=[1.] * 4, require_first_hit=False)
            self.motor = VisualButtonEffects(policy_seed).state()
            self.timing = dict(schema=TIMING_SCHEMA, delay_commands=0,
                               motor_order=list(ORDERS[int(self.rng.integers(len(ORDERS)))]),
                               motor_probabilities=[1 / len(ORDERS)] * len(ORDERS))
            self.score = ScorePlacement()
            self.shot_memory = ShotMemory()
        else:
            if checkpoint is None or isinstance(checkpoint, (str, Path)):
                checkpoint = json.loads(Path(checkpoint or CHECKPOINT).read_text())
            self.policy = copy.deepcopy(checkpoint['interception'])
            self.polarity = self.policy['choices'][4]
            self.motor = copy.deepcopy(checkpoint['motor'])
            self.timing = copy.deepcopy(checkpoint['timing'])
            keep_scores = arm not in ('motor_arrival', 'center')
            self.score = ScorePlacement(**(checkpoint['score'] if keep_scores else {}),
                                        updates_enabled=not self.frozen and arm != 'center')
            self.shot_memory = ShotMemory(**checkpoint['shot_memory']) if keep_scores else ShotMemory()

    def begin_life(self):
        if self.life is not None:
            raise ValueError('finish the current life before resetting')
        if self.arm == 'passive':
            self.life = _Passive()
        elif self.arm == 'fresh_visual':
            self.life = self.cold.begin_life()
        else:
            from pong_role_transfer import RoleAdapter, RoleBinding, RoleController
            choices = list(self.policy['choices'])
            choices[4] = self.polarity
            relation = RelationalPolicy(*choices, self.policy['coefficients'], self.policy['scales'])
            policy = PersistentRelationalPolicy(relation, self.policy['require_first_hit'])
            motor = VisualButtonEffects.from_state(self.motor)
            motor.updates_enabled = not self.frozen
            timing = TimedMotorController(policy, motor, timing_state=self.timing, timing_updates=not self.frozen)
            core = CarriedShotPlacement(timing, self.score.episode(int(self.rng.integers(2**31))),
                                        self.shot_memory)
            fixed = self.arm == 'frozen_orientation'
            binding = RoleBinding(0, 1, 1, -.45) if fixed else self.binding
            adapter = RoleAdapter(mode='frozen' if fixed else 'online', binding=binding)
            self.life = _Life(RoleController(core, adapter))
        return self.life

    def end_life(self):
        if self.life is None:
            raise ValueError('begin a life before ending it')
        if self.arm == 'fresh_visual':
            self.cold.end_life()
        elif self.arm != 'passive':
            if not self.life.ended:
                raise ValueError('an actual terminal or time limit is required')
            if not self.frozen:
                core = self.life.controller
                self.timing = core.timing.state()
                self.shot_memory = core.shot_memory
                self.binding = self.life.detector.binding
                # A rewardless short life provides evidence against the active
                # relation. Try an untested polarity, then use recent outcomes.
                index = int(self.polarity > 0)
                value = min(1., self.life.seconds / 5.) + min(1., self.life.earned)
                evidence = self.relation_evidence[index]
                evidence.append(value)
                del evidence[:-8]
                if value < .8:
                    other = self.relation_evidence[1 - index]
                    if not other or np.mean(other) >= np.mean(evidence):
                        self.polarity *= -1
        self.life = None
        self.lives += 1

    def state(self):
        if self.arm == 'fresh_visual':
            return self.cold.state()
        if self.arm == 'passive':
            return dict(arm=self.arm, lives=self.lives)
        return dict(arm=self.arm, lives=self.lives, timing=copy.deepcopy(self.timing),
                    motor=copy.deepcopy(self.motor), policy=copy.deepcopy(self.policy),
                    score=self.score.to_state(), shot_memory=self.shot_memory.to_state(),
                    binding=self.binding.state() if self.binding is not None else None,
                    polarity=self.polarity, relation_evidence=copy.deepcopy(self.relation_evidence),
                    rng_state=copy.deepcopy(self.rng.bit_generator.state))
