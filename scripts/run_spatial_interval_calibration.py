"""Frozen density and fresh causal calibration/test streams, without retuning."""

import math

import torch

from run_spatial_belief import collect, evaluate


if __name__ == '__main__':
    torch.set_num_threads(1)
    original = torch.load('checkpoints/m1a5/spatial-belief-data.pt', weights_only=True)
    fresh = collect(training_streams=64, train_seed=201001, event_seed=202000,
                    test_seed=205000, phases=(41, 43),
                    angles=(2*math.pi/15, 11*math.pi/30, 4*math.pi/5), scales=(.85, 1.35),
                    cache='checkpoints/m1a5/spatial-belief-calibration-data.pt')
    calibration = fresh['training']
    fresh['training'] = original['training']
    evaluate(fresh, calibration=calibration,
             out='docs/experiments/2026-09-26-spatial-interval-calibration-results.json')
