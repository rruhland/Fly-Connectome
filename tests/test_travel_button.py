from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from travel_button_control import CameraPipe, practice_motor
from travel_button_matches import CONDITIONS, PILOT_SEED, run_match, schedule
from travel_button_panel import MAX_TIP_Z, PitchArm, TravelButtonPanel, decode_travel


def test_overlap_does_not_close_switch_and_lifting_releases_at_fixed_xy():
    panel = TravelButtonPanel(17, wiring_seed=3)
    panel._arm = PitchArm(panel.switches[0].center)

    for _ in range(8):
        assert panel.step((0., 0., 0.)) == 0
        assert panel.overlap == 0
        assert panel.depth[0] == 0.
    untouched = decode_travel(panel.frame())
    assert untouched.height == pytest.approx(.009, abs=.0005)
    assert untouched.depth == (0., 0.)

    for _ in range(16):
        panel.step((0., 0., -1.))
    assert panel.active_button == 0
    assert panel.output == panel.wiring[0]
    switch = panel.switches[0]
    assert panel.depth[0] == pytest.approx(switch.travel)
    assert decode_travel(panel.frame()).depth[0] == pytest.approx(switch.travel, abs=.0005)
    press = panel.transitions[0]
    assert press['closed'] and press['depth_m'] >= switch.closes_at
    assert press['tip_z_m'] < 0.

    for _ in range(16):
        panel.step((0., 0., 1.))
    release = panel.transitions[1]
    assert not release['closed'] and release['depth_m'] <= switch.opens_at
    assert release['overlap'] and panel.overlap == 0
    assert panel.depth_only_releases == 1
    assert panel.active_button is None and panel.output == 0


def test_profile_tracks_the_wrist_at_its_upper_limit():
    panel = TravelButtonPanel(17, wiring_seed=3)
    for _ in range(120):
        panel.step((0., 0., 1.))
    assert panel.height == pytest.approx(MAX_TIP_Z)
    assert decode_travel(panel.frame()).height == pytest.approx(MAX_TIP_Z, abs=.0005)


def test_panel_lip_blocks_lateral_escape_until_fingertip_retracts():
    panel = TravelButtonPanel(17, wiring_seed=3)
    first = panel.switches[0].center
    panel._arm = PitchArm(first)
    for _ in range(16):
        panel.step((0., 0., -1.))
    for _ in range(40):
        panel.step((1., 1., -1.))
        assert panel.overlap == 0
        assert panel.output != 0
    for _ in range(20):
        panel.step((1., 1., 1.))
    assert panel.overlap != 0
    assert panel.output == 0
    assert panel._arm.height > 0.


@pytest.mark.parametrize('lag', [0, 6])
def test_camera_captures_synchronously_and_never_delivers_early(lag):
    pipe = CameraPipe(lag)
    captures = []
    deliveries = []
    for tick in range(20):
        def arm_frame():
            captures.append(tick)
            return np.asarray([tick])

        packet = pipe.sample(tick, tick // 7, arm_frame, lambda: np.asarray([tick]))
        if packet is not None:
            capture, point, arm, game = packet
            assert capture + lag == tick
            assert point == capture // 7
            assert arm[0] == game[0] == capture
            deliveries.append(tick)
    assert captures == list(range(0, 20, 4))
    assert deliveries == list(range(lag, 20, 4))
    assert pipe.age_sum == lag * len(deliveries)


def test_disconnected_delayed_match_keeps_pressing_without_game_drive_and_replays():
    case = schedule(4, PILOT_SEED)[0]
    condition = CONDITIONS['trained_disconnected']
    checkpoint = Path(__file__).resolve().parents[1] / 'artifacts/arm-pong/direct-pong-trained.json'
    practice = practice_motor()
    row = run_match(case, condition, checkpoint, practice, 240, keep_tape=True)
    assert row['presses'] > 0
    assert row['driven_ticks'] == 0
    assert row['binding_events'] == []
    assert row['camera_captures'] == 60
    assert row['camera_deliveries'] == 59
    assert row['received_frame_age_ticks'] == 6 * row['camera_deliveries']
    assert row['tape'][0][0] >= 6
    replay = run_match(case, condition, checkpoint, practice, 240, replay=row['tape'], keep_tape=True)
    assert all(row[key] == value for key, value in replay.items())
