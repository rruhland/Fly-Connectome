from scripts.experimental.tt39_prefix_memory import PRIMARY, compare, episode, roster
from scripts.experimental.tt4_transfer_trial import _read


def test_roster_uses_all_eligible_distinct_prefixes_at_the_deepest_prespecified_boundary():
    tape = _read(PRIMARY)
    expected = [(374369, 2), (374386, 2), (374403, 5), (374420, 10)]
    cases = list(roster(tape))
    assert [(c["reply_seed"], boundary) for _, c, _, boundary, _ in cases] == expected
    assert len({digest for *_, digest in cases}) == len(cases)
    duplicate = dict(tape, cards=tape["cards"] + [tape["cards"][-1]],
                     rows=tape["rows"] + [tape["rows"][-1]])
    assert [case[-1] for case in roster(duplicate)] == [case[-1] for case in cases]


def test_first_next_outcome_is_not_mistaken_for_any_later_return():
    tape = _read(PRIMARY)
    r = tape["rows"][-1]["events"]
    first, second = episode(r, 2), episode(r, 3)
    assert first["outcome"] == second["outcome"] == "legal"
    assert first["legal"]["time"] < second["offer"]["time"]
    assert first["terminal"] is None
    assert episode(r, 10)["outcome"] != "legal"


def test_null_offers_do_not_count_as_a_receiving_miss_or_tie():
    base = dict(outcome="no_legal_offer", offer=None, contact=None, far_contact=None)
    assert compare(base, base)["result"] == "both_no_offer"
    legal = dict(base, outcome="legal", offer={"time": 1}, contact={"time": 2})
    assert compare(base, legal)["result"] == "one_no_offer"
    assert compare(legal, legal)["result"] == "tie_legal"
    assert not compare(base, base)["same_nonnull_offer"]
