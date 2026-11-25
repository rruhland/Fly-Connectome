import numpy as np

from scripts.experimental import tt53_responsive_school as school
from scripts.experimental import tt54_earned_low_flights as flights
from scripts.experimental import tt54_local_motor_model as motor


def test_training_distribution_preserves_adaptive_draw_and_broad_support():
    card = dict(index=3, id=4, seed=14, reply_seed=517, profile="reach")
    teacher = school.School(True, 90, [])
    choices = [teacher.draw(card, i, "random", None)[0] for i in range(150)]
    assert max(np.unique([school.cell_of(c) for c in choices])) == len(school.CELLS) - 1
    assert np.all(teacher.probabilities() >= 0.3 / len(school.CELLS))
    assert np.all(np.array(choices) >= school.BOUNDS[:, 0])
    assert np.all(np.array(choices) <= school.BOUNDS[:, 1])
    teacher.record[0, 0] += 100
    assert choices[3] == teacher.draw(card, 3, "random", None)[0]
    ordinary = school.School(False, 90, [])
    assert ordinary.draw(card, 5, "random", None) == school.DRAW(
        card, 5, "random", None
    )


def test_local_motor_map_uses_other_reward_events_and_preserves_donor_at_origin():
    rng = np.random.default_rng(271)
    rows = [(rng.normal(size=16), rng.normal(size=(21, 6)), i) for i in range(36)]
    lib = motor.Library(rows)
    for mode in ("absolute", "relative", "local"):
        assert np.allclose(lib.generalize(rows[2][0], 2, mode), rows[2][1])
    same_event = motor.Library([(x, g, 1) for x, g, _ in rows])
    assert np.array_equal(same_event.generalize(rows[1][0], 2, "local"), rows[2][1])


def test_earned_height_uses_descending_inbound_path_and_observed_bounce():
    a = dict(tick=7, p=[0.12, -1.0, 0.99], bounced=True)
    b = dict(tick=12, p=[0.15, -1.33, 1.006], bounced=True)
    assert 0.97 < flights.optical_height([a, b]) < 1.02
    assert flights.optical_height([b, a | {"bounced": False}]) is None
