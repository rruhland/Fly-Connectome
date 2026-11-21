import numpy as np
import pytest

from scripts.experimental.tt6_axes import TiltedAxisArena, pulse_receipt
from scripts.experimental.tt6_body_arena import TT6Wiring
from scripts.experimental.tt6_body_probe import probe_commands
from scripts.experimental.tt6_render import cache_hall_stereo


@pytest.mark.parametrize('layout', ['held-a', 'held-b'])
def test_physically_tilted_axes_move_tip_differently_with_the_same_rates(layout):
    row = pulse_receipt(layout, 11779391)
    assert row['action_size'] == 5
    assert len(row['axes']) == 4
    for result in row['axes']:
        assert 12. < result['rotation_degrees'] < 20.
        assert result['tip_difference_meters'] > .015
        assert np.count_nonzero(result['public_command']) == 1


@pytest.mark.parametrize('layout', ['held-a', 'held-b'])
def test_tilted_axes_match_the_original_camera_in_exact_reflections_mode(layout):
    options = dict(layout=layout, venue='hall', width=192, height=144, baseline=.28, view_seed=62549384)
    native = TiltedAxisArena(15308926, **options)
    exact = TiltedAxisArena(15308926, **options)
    try:
        cache_hall_stereo(exact, reflections=True)
        first, second = [TT6Wiring(arena, 87216831) for arena in (native, exact)]
        np.testing.assert_array_equal(first.reset(seed=78010937), second.reset(seed=78010937))
        for command in probe_commands(8246131, 8, exact.action_size):
            np.testing.assert_array_equal(first.step(command)[0], second.step(command)[0])
        np.testing.assert_array_equal(native.data.qpos, exact.data.qpos)
    finally:
        native.close()
        exact.close()
