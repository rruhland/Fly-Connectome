"""Temporal-pathway arms score the same withheld future event."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from distributed_predictive_field import DistributedPredictiveField
from local_observation_model import LocalObservationModel
from local_temporal_mixture import LocalTemporalMixture
from run_absolute_refresh import contrast_frames
from run_local_temporal_mixture import evaluate_streams


def test_pathway_arms_share_one_clean_next_event_target():
    first = torch.zeros((2, 8, 10))
    second = torch.zeros_like(first)
    first[1, 4, 4] = 1.
    second[1, 4, 5] = 1.
    sequence = [first, second]
    unrelated = [second, first]
    streams = [(sequence, sequence, unrelated,
                contrast_frames(sequence), contrast_frames(unrelated))]
    experts = {name: DistributedPredictiveField(height=8, width=10)
               for name in ('slow', 'diverse')}
    observer = LocalObservationModel(height=8, width=10)
    gates = {name: LocalTemporalMixture(height=8, width=10)
             for name in ('aligned', 'shuffled')}
    result = evaluate_streams(streams, experts, observer, gates)
    assert all(row['events'] == 1 and row['targets'] == 1
               for row in result.values())
