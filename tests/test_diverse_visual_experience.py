"""Checks for the matched unlabeled visual-experience distributions."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diverse_visual_experience import (TRAIN_SHAPES, TEST_SHAPES,
                                       diverse_training_sequences,
                                       square_training_sequences,
                                       unseen_shape_sequences)
from learned_latent_probe import training_sequences


def test_square_arm_matches_previous_input_and_budget():
    passes = square_training_sequences()
    previous = training_sequences()
    assert len(passes) == 3 and all(len(episodes) == 32 for episodes in passes)
    for (direction, original), (_, new_direction, _, _, sequence) in zip(
            previous, passes[0]):
        assert new_direction == direction
        assert all(torch.equal(a, b) for a, b in zip(original, sequence))


def test_diverse_training_has_matched_budget_and_balanced_factors():
    passes = diverse_training_sequences()
    assert len(passes) == 3
    assert all(len(episodes) == 32 for episodes in passes)
    for episodes in passes:
        assert {shape for shape, *_ in episodes} == set(TRAIN_SHAPES)
        assert ({(shape, direction, speed)
                 for shape, direction, speed, *_ in episodes}
                == {(shape, direction, speed)
                    for shape in TRAIN_SHAPES
                    for direction in ('up', 'down', 'left', 'right')
                    for speed in (1, 2)})
    assert ({background for episodes in passes
             for *_, background, _ in episodes} == {True, False})


def test_unseen_sequences_have_new_shapes_and_both_event_polarities():
    assert set(TRAIN_SHAPES).isdisjoint(TEST_SHAPES)
    cases = unseen_shape_sequences()
    assert len(cases) == 48
    assert {shape for shape, *_ in cases} == set(TEST_SHAPES)
    assert all(len(events) == 18 and events[3].shape == (2, 32, 64)
               for *_, events in cases)
    assert all(sum(int(frame.sum()) for frame in events) > 0
               for *_, events in cases)


def test_expanded_exposure_preserves_first_three_passes_and_adds_variants():
    baseline = diverse_training_sequences()
    expanded = diverse_training_sequences(epochs=12)
    assert len(expanded) == 12
    assert all(len(episodes) == 32 for episodes in expanded)
    for old, new in zip(baseline, expanded):
        for old_case, new_case in zip(old, new):
            assert old_case[:4] == new_case[:4]
            assert all(torch.equal(a, b) for a, b in zip(old_case[4],
                                                          new_case[4]))
    variants = [next(events for shape, direction, speed, _, events in episodes
                     if (shape, direction, speed) == ('square', 'up', 1))
                for episodes in expanded]
    distinct = []
    for events in variants:
        if not any(all(torch.equal(a, b) for a, b in zip(events, previous))
                   for previous in distinct):
            distinct.append(events)
    assert len(distinct) == 4
