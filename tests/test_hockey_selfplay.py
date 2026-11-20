"""Check reward-only self-play selection and parity of the frozen comparator."""

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from hockey_head_to_head import point
from hockey_selfplay import TACTICS, Tactics, play, schedule


def test_memory_uses_score_and_keeps_source_on_empty_state():
    chosen = TACTICS.index('early_edge')
    memory = Tactics()
    assert memory.select(2, random.Random(1)) == TACTICS.index('carried')
    for score in (1, 0, 1, 1, -1):
        memory.update(2, chosen, score)
    assert Tactics(memory.state()).select(2, random.Random(1)) == chosen


def test_retained_carried_tactic_replays_the_existing_symmetric_point():
    # Substitute any valid hockey rewards without coupling this test to generated artifacts.
    from air_hockey_learning import GoalMemory
    from arcade_wall_learn import WallMemory
    state = {'wall': WallMemory(default=.9685).to_state(), 'goal': GoalMemory().to_state()}
    case = {'seed': 763801209, 'side': 0, 'rival': 'center'}
    original = point(case['seed'], state, ('trained', 'fixed_center'))
    same, _ = play((case, state, Tactics().state(), 0., 'adaptive'))
    assert same['tactic'] == 0
    assert same['actions_sha256'] == original['actions_sha256']
    assert same['reward'] == original['score']


def test_held_out_seeds_are_disjoint_and_swap_camera_sides():
    cases = schedule(24, 6)
    assert not set(row['seed'] for row in cases['practice']) & set(row['seed'] for row in cases['heldout'])
    assert len({(row['seed'], row['rival'], row['side']) for row in cases['heldout']}) == 6 * 4
