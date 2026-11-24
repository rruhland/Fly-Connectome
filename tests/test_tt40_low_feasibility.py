"""Freeze the low development slice and keep evaluator metadata out of example rows."""

from scripts.experimental.tt4_transfer_trial import _read
from scripts.experimental.tt39_opportunity_map import measure
from scripts.experimental.tt40_low_feasibility import SOURCE, roster


def test_low_diagnostic_uses_later_distinct_practice_misses():
    tape = _read(SOURCE)
    chosen = roster(tape)
    assert [(r['id'], r['leg']) for r in chosen] == [(8302,3),(4322,5),(14843,1),(13052,5),(8519,11)]
    assert len({r['id'] for r in chosen}) == 5
    low = measure(SOURCE)['bands']['below_0.99']
    assert (low['offered'],low['touched'],low['legal']) == (15,14,2)
    for r in chosen:
        old = tape['rows'][r['index']]['placements']['delivered'][r['leg']]
        assert r['leg']>0 and r['height']<.99 and not old['near_landed']
