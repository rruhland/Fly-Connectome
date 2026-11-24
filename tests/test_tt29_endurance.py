import pytest

from scripts.experimental.tt29_endurance import audit


def event(kind, time, **kw):
    return dict(kind=kind, time=time, **kw)


@pytest.mark.parametrize("opening_side", (-1, 1))
def test_count_only_landed_physical_hits_and_exclude_opening(opening_side):
    events = [event("legal_return", 0.1, side=opening_side)]
    for i, side in enumerate((-opening_side, opening_side)):
        begin = i + 0.4
        events.extend(
            [
                event(
                    "contact",
                    begin,
                    side=side,
                    incoming=[0, -4 * side, 0],
                    location=[0, -side, 1],
                ),
                event(
                    "separation",
                    begin + 0.01,
                    collider="learner_racket" if side == 1 else "rival_racket",
                    velocity=[0, 5 * side, 0],
                    spin=[0, 0, 0],
                ),
                event(
                    "legal_return",
                    begin + 0.4,
                    side=side,
                    location=[0, 0.8 * side, 0.76],
                ),
            ]
        )
    report = audit(events, opening_side=opening_side)
    assert report["returns"] == 2
    assert report["learner"] == report["opponent"] == 1
    assert report["outgoing_mps"]["median"] == 5
    with pytest.raises(ValueError):
        audit(events[:1] * 2, opening_side=opening_side)


def test_fault_after_a_double_touch_does_not_erase_earlier_shots():
    hit = event("contact", 0.3, side=1, incoming=[0, -3, 0], location=[0, -1.0, 1])
    events = [
        event("legal_return", 0.1, side=-1),
        hit,
        dict(hit, time=0.4),
        event("point", 0.4, reason="double_hit"),
    ]
    assert audit(events)["returns"] == 0
    with pytest.raises(ValueError):
        audit(events + [event("legal_return", 0.8, side=1, location=[0, 1, 0.76])])
