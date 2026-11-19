"""Public-only calibration and movement-intent tests."""

from itertools import permutations
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from pong_visual_button_effects import PROBE_MEASURE, PROBE_SETTLE, VisualButtonEffects, VisualIntentController
from test_relational_persistent_binding import entity, model


def calibrated(displacements):
    learner = VisualButtonEffects(871)
    learner.start_probes()
    for action in learner.probe_queue.copy():
        learner.record_probe(displacements[action])
    return learner


@pytest.mark.parametrize('displacements', tuple(permutations((.12, .01, -.1))))
def test_order_is_discovered_from_displacements_and_rest_is_used_without_visual_binding(displacements):
    learner = calibrated(displacements)
    assert learner.effects == list(displacements)
    assert learner.completed_calibrations == 1
    restored = VisualButtonEffects.from_state(json.loads(json.dumps(learner.state())))
    frozen = restored.frozen_branch()
    controller = VisualIntentController(model(), frozen)
    action, trace = controller.choose([], baseline_action=1)
    assert action == displacements.index(.01)
    assert trace['effective_requests'] == list(map(displacements.index, (.12, .01, -.1)))
    state = frozen.state()
    controller.observe(own_hits=0, reward=-1., terminated=True, truncated=False,
        elapsed_seconds=.02, public_entities=[])
    assert frozen.state() == state == dict(restored.state(), updates_enabled=False)


def test_repeated_interior_stall_requests_evidence_without_reward_or_phase_data():
    learner = calibrated((.09, 0., -.09))
    controller = VisualIntentController(model(), learner)
    scene = [entity(10, 0., -.3), entity(11, -.25, 0.), entity(12, 0., .3)]
    for _ in range(PROBE_SETTLE + PROBE_MEASURE):
        action, trace = controller.choose(scene, 1)
        assert action == trace['intended'] == 2 and not trace['probing']
        controller.observe(own_hits=0, reward=0., terminated=False, truncated=False,
            elapsed_seconds=.02, public_entities=scene)
    assert learner.contradictions == 1
    assert controller.choose(scene, 1)[1]['probing']


def test_stall_far_from_initial_position_does_not_claim_motor_change():
    learner = calibrated((.09, 0., -.09))
    controller = VisualIntentController(model(), learner)
    def scene(own):
        return [entity(10, own, -.3), entity(11, -.4, 0.), entity(12, 0., .3)]
    previous = scene(0.)
    for i in range(PROBE_SETTLE + 3 * PROBE_MEASURE):
        action, trace = controller.choose(previous, 1)
        latest = scene(-.25 * min((i + 1) / 14, 1.))
        assert action == 2 and not trace['probing']
        controller.observe(own_hits=0, reward=0., terminated=False, truncated=False,
            elapsed_seconds=.02, public_entities=latest)
        previous = latest
    assert learner.contradictions == 0
