"""Exercise optical impact segmentation and signed shot correction."""

from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt29_optical_outcomes import OpticalOutcomes, OutgoingFlight, ShotResiduals
from scripts.experimental.tt29_vision import VERTICAL, flight_terms


def test_outgoing_fit_excludes_incoming_samples_and_net_deflection():
    flight = OutgoingFlight(.20, 4.)
    for time in (.12, .15, .18):
        flight.observe(time, np.array([0., -.85 - 3. * (time - .12), 1.]))

    initial = np.array([.10, -1.11, 1.])
    velocity = np.array([.30, 3.5, 1.35])
    for time in np.arange(.21, .56, .03):
        travel, fall = flight_terms(time - .21, flight.drag)
        flight.observe(time, initial + travel * velocity + fall * VERTICAL)

    before_deflection = flight.outcome()
    flight.observe(.57, np.array([.20, -.01, .88]))
    assert flight.done
    assert flight.segment[0][0] == .21
    np.testing.assert_allclose(flight.outcome(), before_deflection)
    assert before_deflection[2] < 0.


def test_negative_optical_error_penalizes_matching_low_shots():
    memory = ShotResiduals()
    base = np.zeros(6)
    features = np.array([base, base, base + 8.])
    planned = np.array([[0., .40, .09], [0., .70, .30], [0., .40, .09]])
    np.testing.assert_array_equal(memory.penalty(features, planned), np.zeros(3))

    for _ in range(12):
        memory.add(base, np.array([0., -.18, -.12]))
    penalty = memory.penalty(features, planned)
    assert penalty[0] > 1.
    assert penalty[1] == 0.
    assert penalty[2] < 1e-10

    restored = ShotResiduals(memory.checkpoint())
    np.testing.assert_array_equal(restored.penalty(features, planned), penalty)


def test_selected_shot_uses_the_rollout_clock():
    rates = np.zeros((1, 3, 6))
    scores = np.zeros((1, 13))
    scores[0, 1] = .25
    arena = SimpleNamespace(
        data=SimpleNamespace(time=1.125), ball=None, velocity=None, spin=None, _received_bounce=False
    )
    planner = SimpleNamespace(
        arena=arena,
        flight=SimpleNamespace(path=lambda *_: np.zeros((1, 7))),
        predict=SimpleNamespace(trajectories=lambda *_: (rates, scores)),
    )
    feedback = OpticalOutcomes(planner, ShotResiduals(), learn=False)
    commands, _ = feedback.trajectories(None, None, None)
    arena.data.time = 2.
    feedback.command(commands[0])
    assert feedback.pending[0].strike == 1.375
