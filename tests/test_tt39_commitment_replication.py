import pytest

from scripts.experimental.tt29_endurance import audit
from scripts.experimental.tt39_commitment_replication import archived_counts, verify_assignment


def test_historical_summary_keeps_the_assigned_no_offer_in_the_denominator():
    base = [{"score": {"learner": 0, "returns": 0}, "placements": {"delivered": []}},
            {"score": {"learner": 2, "returns": 5}, "placements": {"delivered": [{}, {}]}}]
    result = archived_counts({"rows": base})
    assert result == {"points": 2, "own": 2, "first": 1, "later": 1,
                      "longest_own": 2, "longest_alternating": 5, "no_offer": 1}


def test_rescore_checks_committed_assignment_even_without_an_offer():
    card = {"seed": 7}
    events = [{"time": 2.0, "kind": "point"}]
    row = {"placements": {"card": card, "delivered": []}, "opponent_seed": 7,
           "events": events, "score": audit(events, opening_side=1)}
    verify_assignment(card, row)
    with pytest.raises(ValueError, match="another assignment"):
        verify_assignment({"seed": 8}, row)
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_assignment(card, dict(row, score=dict(row["score"], learner=1)))
    with pytest.raises(ValueError, match="before point termination"):
        verify_assignment(card, dict(row, events=[]))
