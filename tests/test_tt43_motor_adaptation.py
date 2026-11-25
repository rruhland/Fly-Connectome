from collections import deque
from pathlib import Path

import numpy as np

from scripts.experimental import tt42_diverse_replay as diverse
from scripts.experimental import tt42_incremental_school as school
from scripts.experimental import tt43_motor_effect as effect
from scripts.experimental import tt43_split_memory as memory


def row(s, u, tag=("live", 1, 1)):
    return np.full(16, s), np.full((21, 6), u), tag


def test_consolidation_preserves_distinct_commands_at_similar_positions():
    known = deque([row(0.0, 0.0), row(0.1, 0.1)])
    ordinary, different = row(0.1, 0.1), row(0.1, 1.0)
    assert memory.distinctive_index([ordinary, different], known) == 1
    bank = object.__new__(memory.SplitMemory)
    bank.online = known
    bank.start()
    bank.earned = [[ordinary, different]]
    bank.online.extend(bank.earned[0])
    assert len(bank.online) == 4
    assert bank.consolidate() == {"rewarded_segments": 2, "consolidated": 1}
    assert len(bank.online) == 3
    assert np.array_equal(bank.online[-1][1], different[1])


def test_redundancy_eviction_preserves_earlier_tutoring_and_capacity():
    protected = row(1.0, 1.0, ("carried", "practice", 1, 1))
    bank = diverse.Repertoire([protected, row(0.1, 0.1), row(0.1, 0.1)], maxlen=3)
    bank.append(row(0.8, 0.3))
    assert bank[0] is protected
    assert len(bank) == 3
    assert np.all(bank[-1][0] == 0.8)


def test_full_school_draw_and_one_curriculum_stage_per_point():
    c = {"id": 2, "seed": 30, "reply_seed": 31, "index": 1}
    partner = school.Partner(True)
    first, _ = partner.goal(c, 0, "random", None)
    assert school.NARROW[0, 0] <= first[0] <= school.NARROW[0, 1]
    partner.stage = 3
    assert partner.goal(c, 0, "random", None)[0] == first
    other = c | {"id": 3}
    assert partner.goal(other, 1, "random", None) == school.ORIGINAL(
        other, 1, "random", None
    )


def test_actuator_effect_targets_are_measured_only_on_rewarded_training_motion():
    bank = Path("artifacts/tt37/long-rgb.npz")
    contexts, velocities = effect.optical_pairs(bank)
    assert len(contexts) == len(velocities) > 100
    assert np.isfinite(velocities).all()
    assert (contexts[:, 4] < 0.0).all() and (velocities[:, 1] > 0.0).all()
    assert np.all(effect.control_basis()[:, :, -1] == 0.0)
    pad = np.zeros((2, 19))
    pad[:, 1:4], pad[:, 7:10] = [1.0, 2.0, 0.0], [0.0, 1.0, 0.0]
    feedback = np.zeros((2, 9))
    feedback[1, 1] = 1.0
    delta, hit = effect.residual(pad, feedback, np.ones(3))
    assert hit.tolist() == [False, True]
    assert np.array_equal(delta[0, 3:], np.zeros(3))
    assert delta[1, 1] == 0.0


def test_delayed_return_uses_only_scored_deliveries_and_real_point_end():
    from scripts.experimental.tt44_delayed_score import labels

    f = list(range(7))
    search = {"tick": 10, "target_errors": f, "contact_delay_s": 0.1}
    later = {"tick": 110, "target_errors": f[::-1], "contact_delay_s": 0.1}

    def score(tick, side):
        return {"kind": "legal_return", "step": tick, "side": side}

    events = [score(5, -1), score(56, 1), score(96, -1), score(156, 1), score(195, -1)]
    doc = {"events": events.copy(), "searches": [search, later]}
    assert list(labels(doc)) == [(f, 0.0)]
    doc["events"].append({"kind": "point", "step": 220})
    assert list(labels(doc)) == [(f, 0.0), (f[::-1], 1.0)]
    doc["events"] = events[:2] + [{"kind": "point", "step": 99}]
    assert list(labels(doc)) == []


def test_preserve_credit_multiset_in_misaligned_control():
    from scripts.experimental.tt43_credit_control import rotate

    initial = row(9.0, 9.0)
    bank = deque([initial, row(0.1, 0.1), row(0.2, 0.2), row(0.3, 0.3)], maxlen=4)
    before = list(bank)
    rotate(bank, 3)
    assert bank[0] is initial
    assert all(
        new[0] is old[0] and new[2] == old[2]
        for new, old in zip(bank, before, strict=True)
    )
    assert all(bank[i][1] is not before[i][1] for i in range(1, 4))
    assert {id(x[1]) for x in bank} == {id(x[1]) for x in before}
