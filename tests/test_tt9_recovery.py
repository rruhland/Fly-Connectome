from types import SimpleNamespace

from scripts.experimental.tt9_recovery import RecoveryPlayer
from scripts.experimental.tt9_recovery_audit import return_opportunities


def test_reply_needs_a_recognized_toss_and_observed_depth_reversal():
    player = RecoveryPlayer.__new__(RecoveryPlayer)
    player.scene = SimpleNamespace(tick=0, tracker=SimpleNamespace(ball=None))
    player.serve_plan = None
    player.reply = player.far_sighting = None

    def observe(tick, depth, *, seen=True):
        player.scene.tick = tick
        player.scene.tracker.ball = SimpleNamespace(observed=seen, position=(0., depth, .2))
        player._observe_reply()

    observe(30, 1.3)
    observe(31, .9)
    assert player.far_sighting is None and player.reply is None

    player.serve_plan = (17., (0., -1.55))
    observe(20, .8)
    assert player.far_sighting is None
    observe(38, .95)
    observe(40, 1.23)
    observe(41, 1.19)
    observe(42, 1., seen=False)
    assert player.reply is None
    observe(43, 1.)
    assert player.reply == 43


def test_audit_distinguishes_reaching_and_legally_returning_a_service_reply():
    rally = {'server': 1, 'evaluator': {'legal_serve': True}, 'events': [
        {'kind': 'contact', 'side': 1}, {'kind': 'legal_serve', 'side': 1},
        {'kind': 'contact', 'side': -1}, {'kind': 'legal_return', 'side': -1},
        {'kind': 'contact', 'side': 1}, {'kind': 'point'},
    ]}
    assert list(return_opportunities(rally)) == [('after_own_serve', True, False)]
    rally['events'].insert(-1, {'kind': 'legal_return', 'side': 1})
    assert list(return_opportunities(rally)) == [('after_own_serve', True, True)]
