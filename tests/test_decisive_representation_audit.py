"""The one-shot representation audit keeps probe splits and inputs causal."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_decisive_representation_audit import (DIRS, case_frames,
                                                fit_centroids, make_cases,
                                                predict_direction)


def test_calibration_and_transfer_splits_are_disjoint_and_balanced():
    cases = make_cases()
    calibration = [case for case in cases if case['split'] == 'calibration']
    assert len(calibration) == 16
    assert {obj['direction'] for case in calibration
            for obj in case['objects']} == set(DIRS)
    assert {obj['speed'] for case in calibration
            for obj in case['objects']} == {1}
    assert {obj['shape'] for case in calibration
            for obj in case['objects']} == {'square'}
    assert {case['split'] for case in cases} == {
        'calibration', 'position', 'speed', 'shape',
        'separated', 'crossing'}
    assert all(len(case['objects']) == 2 for case in cases
               if case['split'] in ('separated', 'crossing'))


def test_scene_frames_and_regions_do_not_use_future_motion():
    case = next(case for case in make_cases()
                if case['split'] == 'calibration')
    images, regions = case_frames(case)
    assert len(images) == 18
    assert len(regions) == 1
    assert images[0].shape == (1, 32, 64)
    assert images[0].equal(images[1])
    assert regions[0].any()
    assert not regions[0].all()


def test_centroid_probe_uses_only_calibration_features():
    rows = [{'direction': direction, 'feature': feature}
            for direction, feature in zip(DIRS, torch.eye(4))]
    model = fit_centroids(rows, 'feature')
    for row in rows:
        assert predict_direction(model, row['feature']) == row['direction']
