import numpy as np

from scripts.experimental.tt4_transfer import (
    ArmAdapter,
    CalibrationSample,
    TransferPrior,
    ZERO,
    calibration_commands,
)
from scripts.experimental.tt4_transfer_trial import _metrics, _seeds, load_sources


def test_probe_fit_uses_action_responses_across_points_and_recovers_changed_wiring():
    ready = np.array((.03, -1.41, .25))
    response = np.array([[.03, .19, -.04, -.06], [-.21, .07, .00, -.01], [.11, -.10, -.29, -.01]])
    trajectories = []
    for seed in range(6):
        commands = np.vstack((ZERO, calibration_commands(seed, 36)))
        rows = []
        coordinates = np.zeros(4)
        effective = np.zeros(4)
        for tick, action in enumerate(commands):
            coordinates += action / 4
            effective = .60 * effective + .40 * coordinates
            mover = np.array((.20 * seed, 1.4 - .05 * tick, .30 + .18 * np.cos(.2 * tick)))
            rows.append(CalibrationSample(action, {2: ready + response @ effective, 4: mover}, {2: 5, 4: 9}))
        trajectories.append(rows)

    arm = ArmAdapter(trajectories)

    np.testing.assert_allclose(arm.ready, ready)
    np.testing.assert_allclose(arm.jacobian(ZERO), response, atol=.045)
    assert arm.rank == 3
    assert arm.hue == 5
    action = arm.command(ZERO, ZERO, ready, ready + np.array((-.08, .04, .06)))
    assert np.max(np.abs(action)) <= 1.
    assert (response @ action) @ np.array((-.08, .04, .06)) > 0.


def test_mini_and_native_share_source_but_enter_with_distinct_timing_evidence():
    native, states, _ = load_sources()
    options = ('legacy_mini', 'pong_only', 'fresh', 'mini_shuffled_timing')
    priors = {name: TransferPrior(name, native, 92, mini=states['legacy_mini']) for name in options}

    observed = {}
    for name, prior in priors.items():
        option, delay = prior.choose()
        observed[name] = prior.delay[0][delay]
        if name in ('pong_only', 'fresh'):
            assert observed[name] == 0.
            prior.observe(option, delay, -1., opportunity=True, outgoing=False)
            new_option, new_delay = prior.choose()
            assert new_delay != delay
            prior.observe(new_option, new_delay, 1., opportunity=True, outgoing=True)
            assert len({delay, new_delay, prior.choose()[1]}) == 3

    assert observed['legacy_mini'] > 0.
    assert observed['mini_shuffled_timing'] == -observed['legacy_mini']


def test_paired_schedule_has_distinct_probe_and_test_seeds():
    first = _seeds(7212, 0, 10, 12)
    other = _seeds(7212, 1, 10, 12)
    assert first == _seeds(7212, 0, 10, 12)
    assert not set(first['calibration']) & set(first['tests'])
    assert not set(first['tests']) & set(other['tests'])


def test_reply_denominator_excludes_contacts_without_legal_exchange():
    samples = [
        (1, 0, 0, 0),
        (1, 1, 0, 1),
        (3, 2, 2, 0),
    ]
    rows = [dict(contacts=c, player_returns=p, rival_returns=r, points_for=w, points_against=1-w,
                 steps=10, frames=11, attributed_rewards=0, predicted_returns=0, approach_frames=1)
            for c, p, r, w in samples]
    scores = _metrics(rows)
    assert scores['with_contact'] == 3
    assert scores['with_return'] == 2
    assert scores['reply_after_return'] == 1
    assert scores['sustained'] == 1
    assert scores['won_after_return'] == 1
