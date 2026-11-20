from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from hockey_direction_transfer import VisualPolarity, play_hockey, play_pong, schedule


def test_identifies_inverse_pong_button_from_native_pixel_motion():
    memory = VisualPolarity()
    result = play_pong(1600250801, -1, memory, -1)
    assert memory.polarity == -1
    assert result['hits'] > 0
    assert result['measurement']['request'] == -1
    assert result['measurement']['measured'] > result['measurement']['anchor']
    assert 0 < result['probes'] < result['frames']


def test_hockey_blind_probe_learns_both_axes_and_swapped_prior_recovers():
    case = {'seed': 1850844603, 'style': 'fast-bank'}
    device = (-1, -1)
    probe = play_hockey(case, device, (VisualPolarity(), VisualPolarity()),
                        probe=True, frozen=False, probe_direction=(1, -1))
    online = play_hockey(case, device, (VisualPolarity(), VisualPolarity(1)),
                         probe=False, frozen=False, probe_direction=(1, -1))
    frozen = play_hockey(case, device, (VisualPolarity(), VisualPolarity(1)),
                         probe=False, frozen=True, probe_direction=(1, -1))
    assert probe['initial_sha256'] == online['initial_sha256'] == frozen['initial_sha256']
    assert probe['first_correct'] == [1, 1]
    assert probe['probes'] == 1
    assert online['after'][1]['polarity'] == -1
    assert online['after'][1]['contradictions'] == 1
    assert online['opposite'][1] == 1
    assert frozen['after'][1]['polarity'] == 1
    assert frozen['first_correct'][1] is None


def test_device_swaps_use_independent_pong_and_hockey_seeds():
    cases = schedule(2939016, 4, 2, 4)
    seeds = []
    for pair in cases:
        assert sorted(case['source_motor'] for case in pair) == [-1, 1]
        for case in pair:
            seeds.extend(case['pong'])
            seeds.extend(game['seed'] for game in case['hockey'])
    assert len(seeds) == len(set(seeds))
