from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from arm_pong_timing import ReleaseTiming, VisualRelease


def _two_releases(learner):
    learner.observe(1, 0)
    assert learner.held == 0
    learner.request(1, 1)
    learner.observe(2, 0)
    learner.request(2, None)
    learner.observe(3, None)
    assert learner.held is None
    learner.observe(5, 1)
    learner.request(5, 0)
    learner.observe(8, 1)
    learner.observe(9, None)
    learner.observe(11, 0)


def test_release_timing_fixed_control_sees_same_events_and_never_anticipates():
    adaptive = ReleaseTiming(.01, learn=True)
    fixed = ReleaseTiming(.01, learn=False)
    for estimator in (adaptive, fixed):
        _two_releases(estimator)
        assert estimator.releases == [VisualRelease(0, 1, 3), VisualRelease(1, 5, 9)]
    assert adaptive.stop_seconds == .03
    assert fixed.stop_seconds == 0.
    adaptive.observe(12, None)
    assert adaptive.stop_seconds == 0.


def test_return_to_held_target_cancels_release_sample():
    learner = ReleaseTiming(.01, learn=True)
    learner.observe(1, 0)
    learner.request(1, None)
    learner.request(2, 0)
    learner.observe(3, None)
    assert learner.releases == []
    assert learner.stop_seconds == 0.
