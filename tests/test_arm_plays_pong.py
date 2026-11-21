"""Verify the independently replayable arm-to-native-Pong actuation path."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from arm_plays_pong import MatchCase, VisualButtonBinding, run_match, screen_frame
from button_pong_skill import DesiredPongPolicy
from fly_connectome.pong import Physics, Pong
from pong_button_panel import ButtonPanel, SOURCE_GEAR, VisualButtonMotor, pretrain_motor


def test_cached_binary_camera_matches_native_raster():
    game = Pong([1078106091, 1693910867, 1728122718])
    game.ball[:] = torch.tensor([[.001, .988], [.051, .401], [.945, .1125]])
    game.body.position[:] = torch.tensor([.1, .3765, .9])
    game.opponent[:] = torch.tensor([.11875, .529, .9])
    for size in (64, 192):
        native = game.render(size, size).numpy()
        np.testing.assert_array_equal(np.stack([screen_frame(game, i, size) for i in range(3)]), native)


def test_visual_binding_tracks_delayed_displacement_and_unseen_reversal():
    binding = VisualButtonBinding(0)
    paddle, activation, velocity = .41, 0., 0.
    for tick in range(160):
        native_direction = int(tick < 80) - int(tick >= 80)
        activation += .10 * (native_direction - activation)
        velocity += .035 * (.74 * activation - velocity)
        paddle += velocity / 120
        binding.advance(0)
        if tick % 4 == 0:
            binding.observe(round(paddle * 192) / 192, tick)
    assert [event['down_button'] for event in binding.events] == [0, 1]
    assert binding.events[0]['tick'] < 35
    assert 80 < binding.events[1]['tick'] < 152
    assert binding.resolve(-1) == 0
    assert binding.resolve(0) is None
    binding.reset_trace()
    binding.observe(.5, 161)
    assert binding.resolve(1) == 1


def test_joint_trace_replays_native_contacts_scores_and_boundaries():
    case = MatchCase(12341234, 242424, 373733, 811375)
    memory, _ = pretrain_motor(35345, gear=SOURCE_GEAR)
    reported = []

    def callback(row):
        assert row['arm'].shape == (160, 160, 3)
        assert (row['screen'] is not None) == row['camera_tick']
        reported.append(row['score'])

    result = run_match(case, DesiredPongPolicy(), VisualButtonMotor(8, memory=memory),
                       max_ticks=460, opponent_speed=.35, panel_kwargs={'gear': SOURCE_GEAR},
                       rewire_ticks=(132,), record=True, callback=callback)
    panel = ButtonPanel(case.panel_seed, wiring_seed=case.wiring_seed, gear=SOURCE_GEAR)
    game = Pong([case.pong_seed], config=Physics(opponent_speed=.35))
    scores = []
    hits = 0
    assert result['physical_motion_ticks'] > 0
    assert result['contacts']
    for row in result['trajectory']:
        if row['tick'] == 132:
            panel.rewire()
        drive = panel.step(row['action'])
        assert drive == row['drive'] == row['panel_drive']
        assert panel.active_button == row['active_button']
        outcome = game.step(torch.tensor([-float(drive)]), auto_reset=False)
        score = int(outcome.scores[0])
        hits += int(outcome.hits[0])
        scores.append(score)
        if score:
            game.reset(torch.ones(1, dtype=torch.bool))
    assert reported == scores
    assert hits == result['own_returns']
    assert len(reported) == sum(point['physics_ticks'] for point in result['points'])
    assert sum(map(abs, scores)) == result['completed_points']
    assert sum(result['return_ticks'][i] < result['return_ticks'][i + 1]
               for i in range(len(result['return_ticks']) - 1)) == max(0, hits - 1)


def test_disconnected_panel_makes_physical_presses_without_a_paddle_effect():
    case = MatchCase(1665296180, 242424, 3, 811375)
    memory, _ = pretrain_motor(35345)
    results = [run_match(case, DesiredPongPolicy(), VisualButtonMotor(8, memory=memory),
                         max_ticks=410, opponent_speed=.35, contact_enabled=False, stationary=stationary)
               for stationary in (False, True)]
    disconnected, stationary = results
    assert disconnected['contacts']
    assert not disconnected['binding_events']
    assert disconnected['points'] == stationary['points']
    assert disconnected['drives_sha256'] == stationary['drives_sha256']
    assert disconnected['joint_motion_commands'] > stationary['joint_motion_commands'] == 0
