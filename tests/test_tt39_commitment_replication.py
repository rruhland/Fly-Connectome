from scripts.experimental.tt39_commitment_replication import archived_counts


def test_historical_summary_keeps_the_assigned_no_offer_in_the_denominator():
    base = [{"score": {"learner": 0, "returns": 0}, "placements": {"delivered": []}},
            {"score": {"learner": 2, "returns": 5}, "placements": {"delivered": [{}, {}]}}]
    result = archived_counts({"rows": base})
    assert result == {"points": 2, "own": 2, "first": 1, "later": 1,
                      "longest_own": 2, "longest_alternating": 5, "no_offer": 1}
