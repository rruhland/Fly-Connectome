"""Check that timing practice learns from matched public stroke evidence."""

from copy import deepcopy

import pytest

from scripts.experimental.tt8_contact import PRACTICE_ARMS, PROFILES, PairedTimingMemory, practice_samples


def test_paired_timing_prefers_a_supported_phase_and_updates_from_new_evidence():
    cues = [1., .25, -.25, .25, .25]
    fixed, early, late = PROFILES
    practice = [[cues, [.30, .85, .20], [1., 1., 1.]]] * 8
    memory = PairedTimingMemory(practice)
    assert memory.choose(cues) == early
    assert memory.choose([1., -1.5, 1.5, -1.5, -1.5]) == fixed
    for _ in range(8):
        memory.observe(early, cues, .05, 1.)
    assert memory.choose(cues) == fixed

    cold = PairedTimingMemory()
    for profile in (fixed, early, late):
        assert cold.choose(cues) == profile
        cold.observe(profile, cues, .40, .5)
        cold.observe(profile, cues, .40, .5)
    assert cold.choose(cues) == fixed


def test_practice_matches_starts_and_ignores_referee_fields():
    cue = [1., 0., 0., 0., 0.]
    groups = {}
    for name, profile in zip(PRACTICE_ARMS, PROFILES, strict=True):
        attempt = dict(profile=profile, cues=cue, score=.1 * profile, weight=1.)
        point = dict(seed=42, phase='mixed', contact_timing=[attempt], player_returns=10)
        groups[name] = dict(points=[point], replayed_points=1)
    report = dict(held=[dict(conditions=groups)])
    changed = deepcopy(report)
    changed['held'][0]['conditions']['fixed']['points'][0]['player_returns'] = 0
    assert practice_samples(report) == practice_samples(changed)
    groups['fixed']['points'][0]['seed'] = 43
    with pytest.raises(ValueError, match='schedules diverged'):
        practice_samples(report)
