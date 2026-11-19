"""Check conservative handoff, score credit and matched center intent."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from pong_risk_aware_placement import RiskAwarePlacement
from pong_score_placement import ScorePlacement
from pong_timing_controller import ArrivalContext


class TimingStub:
    def __init__(self):
        self.inherited = SimpleNamespace(bound_ids=None)
        self.seconds = 0.
        self.delay_commands = 0
        self.camera_seconds = .02
        self.observations = []

    def observe(self, **transition):
        self.observations.append(transition)


@pytest.mark.parametrize('pursuit,cancelled', ((.30, False), (.58, True)))
def test_terminal_score_is_credited_only_if_placement_was_applied(pursuit, cancelled):
    policy = ScorePlacement()
    policy.observe_score(3, .07, 1.)
    episode = policy.episode(381793)
    timing = TimingStub()
    controller = RiskAwarePlacement(timing, episode)
    controller.pursuit_floor = pursuit

    context = ArrivalContext(.07, .85, 0., -.4, .4, .07)
    offset = controller._offset(context)
    assert bool(offset) is not cancelled
    assert bool(controller.cancelled) is cancelled
    controller.observe(terminated=True, truncated=False, reward=-1.)
    assert episode.done and timing.observations == [dict(terminated=True, truncated=False, reward=-1.)]
    assert policy.visits[3] == (1 if cancelled else 2)


def test_center_forecast_remains_eligible_under_pursuit():
    placement = ScorePlacement(updates_enabled=False).episode(1)
    controller = RiskAwarePlacement(TimingStub(), placement, forecast_center=True)
    controller.pursuit_floor = .58
    context = ArrivalContext(.07, .85, 0., -.4, .4, .07)
    assert controller._offset(context) == pytest.approx(1e-9, abs=1e-14)
    assert not controller.cancelled


@pytest.mark.parametrize('actual_speed,recognizes_pursuit', ((.35, False), (.65, True)))
def test_pursuit_is_measured_from_consistent_public_positions(actual_speed, recognizes_pursuit):
    timing = TimingStub()
    timing.axis = 0
    timing.inherited.bound_ids = (10, 11)
    timing.inherited.history = []
    controller = RiskAwarePlacement(timing, ScorePlacement().episode(1))
    for ordinal in range(9):
        timing.seconds = .02 * ordinal
        normalized = round((-.20 + actual_speed * timing.seconds) * 128) / 128
        controller.entities = [
            dict(id=10, normalized_position=(0., -.45), observed=True),
            dict(id=11, normalized_position=(.2, 0.), observed=True),
            dict(id=12, normalized_position=(normalized, .45), observed=True),
        ]
        controller._observe_pursuit()
    context = ArrivalContext(.07, .85, 0., -.4, .4, .07)
    assert (controller._reason(context) == 'pursuit') is recognizes_pursuit


def test_recent_shot_outcomes_carry_across_delayed_points_and_allow_retry():
    from pong_carried_shot_value import CarriedShotPlacement, ShotMemory

    memory = ShotMemory()
    policy = ScorePlacement()
    policy.observe_score(3, .07, 1.)
    context = ArrivalContext(.07, .85, 0., -.4, .4, .07)
    offsets = []
    for reward in [-1.] + [0.] * 5 + [1.]:
        timing = TimingStub()
        timing.delay_commands = 12
        timing.belief = SimpleNamespace(confirmed=0, probabilities=[1.])
        timing.candidate_dwell = 0
        controller = CarriedShotPlacement(timing, policy.frozen_branch().episode(1), memory)
        offsets.append(controller._offset(context))
        controller.observe(terminated=reward != 0., truncated=reward == 0., reward=reward)
        memory = ShotMemory(**controller.shot_memory.to_state())
    assert offsets[0] != 0 and offsets[-1] == offsets[0]
    assert offsets[1:-1] == [0.] * 5
    assert memory.delayed_win_estimate == pytest.approx(.65)
    assert memory.skipped_points == 0


def test_delay_shot_evidence_does_not_suppress_an_ordinary_scoring_opportunity():
    from pong_carried_shot_value import CarriedShotPlacement, ShotMemory

    memory = ShotMemory(delayed_win_estimate=.05)
    timing = TimingStub()
    policy = ScorePlacement()
    policy.observe_score(3, .07, 1.)
    controller = CarriedShotPlacement(timing, policy.episode(1), memory)
    assert controller._offset(ArrivalContext(.07, .85, 0., -.4, .4, .07)) != 0.
    controller.observe(terminated=True, truncated=False, reward=1.)
    assert controller.shot_memory == memory
