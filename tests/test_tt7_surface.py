from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt5_player import GRAVITY
from scripts.experimental.tt7_surface import SurfaceBelief, SurfacePlayer, optical_bounce
from scripts.experimental.tt7_surface_trial import prepare_material


@pytest.mark.parametrize('coefficient', (.64, .94))
def test_optical_bounce_recovers_separate_arcs_with_biased_height(coefficient):
    # The camera misses the impact frame and overestimates apparent height.
    impact = 10.2
    times = np.array([6, 7, 9, 11, 12, 13])
    delta = times - impact
    velocity = np.where(delta < 0., -.095, .095 * coefficient)
    heights = .049 + velocity * delta - .5 * GRAVITY * delta**2
    rows = list(zip(times, np.column_stack([.13 + .01 * delta, -.76 - .09 * delta, heights])))
    assert optical_bounce(rows[:-1]) is None
    error, ratio, turn = optical_bounce(rows)
    assert error < .003
    assert ratio == pytest.approx(coefficient, abs=.047)
    assert turn == pytest.approx(impact, abs=.10)
    # A departing ball, a reversal outside the table, and a high flight
    # reversal carry no evidence of an approaching bounce on the tabletop.
    for delta_position, reflect_depth in [([0., 0., 0.], -1.), ([0., -1., 0.], 1.), ([0., 0., .22], 1.)]:
        altered = [(t, p * [1., reflect_depth, 1.] + delta_position) for t, p in rows]
        assert optical_bounce(altered) is None


def test_surface_belief_adapts_to_recent_observations():
    belief = SurfaceBelief()
    assert belief.predict(.86) == .86
    belief.values.extend([.9, .90, .65, .64, .64])
    assert .64 < belief.predict(.86) < .73
    belief.values.extend([.93, .95, .94])
    assert .87 < belief.predict(.86) < .95


def test_low_rebound_advances_only_a_low_predicted_intercept():
    player = SurfacePlayer.__new__(SurfacePlayer)
    player._surface_ratio = lambda: .64
    player.adaptive = True
    plane = -1.39
    low = [0., plane, .10]
    assert player._reachable_plane(plane, low) == pytest.approx(-1.20)
    assert player._reachable_plane(plane, [0., plane, .28]) == plane
    player.adaptive = False
    assert player._reachable_plane(plane, low) == plane


def test_material_damping_preserves_visual_assets_and_selects_disturbance():
    from scripts.experimental.tt6_body_arena import TT6Arena

    arena = TT6Arena(seed=44, body='gantry', render_mode='projected')
    try:
        table = arena.model.geom('table')
        original = SimpleNamespace(position=table.pos.copy(), size=table.size.copy(), color=table.rgba.copy(),
                                   stiffness=table.solref[0])
        disturbance = prepare_material(arena, 'low-delay')
        assert disturbance.delays == ((0, 2), (62, 0))
        assert table.solref[1] == -90.
        prepare_material(arena, 'lively')
        assert table.solref[1] == -12.
        np.testing.assert_array_equal(table.pos, original.position)
        np.testing.assert_array_equal(table.size, original.size)
        np.testing.assert_array_equal(table.rgba, original.color)
        assert table.solref[0] == original.stiffness
    finally:
        arena.close()
