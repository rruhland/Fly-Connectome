"""Independent sensory branches must not compete for the same latent units."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from learned_transition_units import TransitionPopulation
from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import scene_sequence
from split_sensory_pathway import SplitPopulation


def test_raw_event_activates_only_its_own_branch_after_blank():
    split = SplitPopulation(TransitionPopulation(channels=16, units=2),
                            TransitionPopulation(channels=2, units=2))
    code = torch.zeros((18, 32, 64))
    code[0, 10, 12] = 1
    split.step(code)
    assert split.correlation.previous_sources == []
    assert len(split.raw.previous_sources) == 1
    assert split.previous_sources == [(10, 12, split.raw.previous_sources[0][2]+2)]
    assert split.correlation.latent.sum() == 0


def test_coincidence_winner_is_unchanged_by_raw_branch_activity():
    correlation = TransitionPopulation(channels=16, units=2)
    split = SplitPopulation(correlation, TransitionPopulation(channels=2, units=2))
    code = torch.zeros((18, 32, 64))
    code[0, 10, 12] = 1
    code[2, 10, 12] = 1
    split.step(code)
    winner = split.correlation.previous_sources[0]
    assert winner == (10, 12, int(split.correlation.latent[:, 10, 12].argmax()))
    assert split.previous_sources[0] == winner
    assert split.previous_sources[1][2] >= split.correlation.units


def test_opposite_motion_can_have_identical_reappearance_evidence():
    events = []
    for direction in (-1, 1):
        obj = dict(shape='diamond', center=(16, 32-2*direction),
                   before=(0, direction), after=(0, direction),
                   hidden=(7, 8, 9))
        events.append(scene_sequence([obj], background=True))
    left, right = events
    assert torch.equal(left[10], right[10])
    assert not torch.equal(left[11], right[11])
    assert correlation_sequence(left)[10].sum() == 0
    assert correlation_sequence(right)[10].sum() == 0

    split = SplitPopulation(TransitionPopulation(channels=16, units=2),
                            TransitionPopulation(channels=2, units=2))
    sources = []
    for sequence in events:
        split.reset_state()
        for event, coincidence in zip(sequence[:11],
                                      correlation_sequence(sequence)[:11]):
            split.step(torch.cat((event, coincidence)))
        sources.append(split.previous_sources)
    assert sources[0] == sources[1]
