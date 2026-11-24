"""Checks for causal, local outcome updates in the direct TT37 learner."""

import numpy as np

from scripts.experimental.tt37_online import ExchangeMemory, Stroke, sensory_context


def test_consequence_updates_only_nearby_readout():
    local = np.zeros(12)
    unseen = local.copy()
    unseen[0] = 2.
    rng = np.random.default_rng(8)
    bank = ExchangeMemory()
    assert bank.choose(local, rng, 0.)[0] == 0
    error = np.full(5, .15)
    bank.keep(dict(context=local, choice=5, reward=1., miss_dq_rad=error))
    chosen, hint, _, _ = bank.choose(local, rng, 0.)
    assert chosen == 5
    np.testing.assert_allclose(hint, error)
    assert bank.choose(unseen, rng, 0.)[0] == 0


class Body:
    def local(self, q):
        return np.array([.10, -1.22, .97]), np.eye(3, 5), .1


def test_context_reads_present_body_and_ball():
    o = np.array([.16, -.85, 1.10, .08, -3.4, .7])
    q = np.zeros(6)
    assert len(sensory_context(o, q, Body())) == 12
    np.testing.assert_allclose(sensory_context(o, q, Body())[:4], [.16, .06, .37, .13])


def test_an_observed_miss_can_supply_signed_bearing_without_legal_credit():
    body, bank = Body(), ExchangeMemory()
    rng = np.random.default_rng(9)
    stroke = Stroke(body, bank, np.zeros(5), rng, 0., True)
    o = np.array([.14, -.6, 1.05, .08, -3.4, .7])
    q = np.zeros(6)
    stroke.start(10, o, q, None)
    assert not bank.records
    stroke.joints[18] = q[:5]
    blade = body.local(q)[0]
    camera = [(16, blade + [.10, .04, .04], blade), (18, blade + [.10, -.04, .04], blade)]
    stroke.finish(21, camera, False)
    assert len(bank.records) == 1 and not bank.records[0]["legal"]
    assert bank.records[0]["miss_dq_rad"][0] > 0.
