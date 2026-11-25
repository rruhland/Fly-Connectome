from types import SimpleNamespace

import numpy as np

from scripts.experimental import tt59_online_corrections as live


def test_credit_uses_sent_reference_and_delayed_real_reward():
    credited = []
    reference = np.zeros(6)
    anchor = np.zeros((live.base.H, 6))
    feature = np.ones(22)
    x = np.array([0., -.55, .89, .04, -3.9, -2.3])
    player = SimpleNamespace(
        credit_context=(x, feature, anchor, reference, []),
        pending_tick=100,
        learned_here=0,
        mode="adaptive23",
        readout=SimpleNamespace(credit=lambda *a: credited.append(a)),
    )
    for j in range(111, 161):
        live.Online.sensory_credit(player, j, np.full(6, j/1000), j == 160)
    assert player.learned_here == 1 and len(credited) == 1
    received = credited[0][-1]
    assert np.array_equal(received[0], np.full(6, .111))
    assert np.array_equal(received[-1], np.full(6, .125))
    assert player.credit_context is None


def test_adaptive_increment_does_not_refit_frozen_basis():
    model = live.new_memory()
    start = len(model.ids)
    basis = model.basis.copy()
    context = np.zeros(22)
    correction = np.ones((live.base.H, 6))*0.025
    model.credit(np.array([0., -.60, .88, 0., -4.0, -2.]), context, np.zeros_like(correction), correction)
    assert start == 23 and len(model.ids) == start+1 and model.ids[-1] == -1
    assert np.array_equal(basis, model.basis)
    assert model.weights.shape == (24, live.base.RANK)


def test_sealed_online_stream_finishes_and_does_not_credit_frozen_policy():
    from scripts.experimental import tt59_online_report as report
    from scripts.experimental.tt4_transfer_trial import _hash, _read

    receipt = _read(report.REPORT)
    assert receipt["stream"] == _hash(live.OUT)
    assert receipt["complete"] and len(receipt["rows"]) == 8
    result = receipt["scores"]
    assert [result[m]["returns"] for m in live.MODES] == [62, 53, 53]
    assert [result[m]["learned"] for m in live.MODES] == [0, 0, 1]
    tapes = _read(live.OUT)
    for row, recorded in zip(receipt["rows"], tapes["rows"], strict=True):
        assert recorded["arms"]["adaptive23"]["actions"] == recorded["arms"]["frozen23"]["actions"]
        assert row["arms"]["adaptive23"]["first_diff_from_frozen"] is None
        for arm in row["arms"].values():
            assert arm["complete_point"]
            if "replay" in arm:
                assert arm["audited_safe"]
