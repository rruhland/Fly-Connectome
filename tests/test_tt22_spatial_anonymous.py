from scripts.experimental.tt22_spatial_anonymous import ending


def test_same_control_near_contact_can_follow_a_legal_rival_bounce():
    data = dict(seed=5, reward=-1, outcome={'reason': 'own_half', 'rival_returns': 1}, receiving=[],
                events=[dict(kind='legal_serve', side=-1, step=20),
                        dict(kind='legal_return', side=-1, step=50),
                        dict(kind='contact', side=1, step=50)])
    audit = ending(data)
    assert audit['legal_opening']
    assert audit['last_legal_rival_reply'] == 50
    assert audit['near_after_reply']
