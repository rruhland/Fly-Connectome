from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from arcade_alternation import AlternatingSession, scene_evidence
from breakout_brick_placement import BrickShotPolicy
from breakout_visual_rebound import VisualReboundPolicy
from pong_role_transfer import RoleBinding


def _board():
    frame = torch.zeros((64, 64), dtype=torch.bool)
    frame[60:62, 26:38] = True
    frame[30:32, 15:17] = True
    for row in (5, 11, 17):
        for col in (2, 18, 42):
            frame[row:row + 3, col:col + 5] = True
    return frame


def test_visible_scene_routes_across_camera_rotations():
    board = _board()
    opposed = torch.zeros_like(board)
    opposed[26:38, 2:4] = opposed[26:38, 60:62] = True
    opposed[20:22, 31:33] = True
    for rotation in range(4):
        targets = scene_evidence(board.rot90(rotation))
        assert targets.appearance == 'target_array'
        assert targets.target_rectangles == 9
        rails = scene_evidence(opposed.rot90(rotation))
        assert rails.appearance == 'opposing_rails'
        assert rails.opposing_rails
    board[2:4, 26:38] = True
    assert scene_evidence(board).appearance == 'unresolved'


@pytest.mark.parametrize('policy, evidence', [(BrickShotPolicy, [0.7] * 24), (VisualReboundPolicy, 0.7)])
def test_session_checkpoint_preserves_both_placement_memories(policy, evidence):
    session = AlternatingSession(27, brick_state=policy().to_state(), recall_roles=True)
    session.brick_policy.observe_score(evidence, 1.)
    session.core.score.observe_score(0, 0., -1.)
    state = session.state()
    assert AlternatingSession.from_state(state).state() == state


def test_matching_appearance_can_retrieve_saved_timing_prior():
    session = AlternatingSession(27, recall_timing=True)
    session.roles['target_array'] = RoleBinding.from_station(1, .453125)
    session.timings['target_array'] = dict(session.core.timing, delay_commands=6)
    session.previous_appearance = 'opposing_rails'
    restarted = AlternatingSession.from_state(session.state())
    life = restarted.begin_life()
    life.reset(_board())
    assert life.recalled_role and life.recalled_timing
    assert life.timing.delay_commands == 6
    assert life.timing.timing_updates
