"""The frozen complement check uses generic full-scene heldouts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sensory_history_complement import evaluation_cases


def test_complement_suite_includes_full_scenes_and_unseen_gap_windows():
    cases = evaluation_cases()
    families = {family for family, _, _, _ in cases}
    assert families == {'single', 'independent', 'crossing', 'occlusion',
                        'speed_change', 'noise', 'early_2', 'late_4'}
    assert sum(family == 'early_2' for family, _, _, _ in cases) == 48
    assert sum(family == 'late_4' for family, _, _, _ in cases) == 48
    assert all(at is None and hidden is None
               for family, _, at, hidden in cases if family == 'single')
