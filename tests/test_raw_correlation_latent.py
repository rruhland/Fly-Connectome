"""Raw events must remain visible when coincidence input is blank."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from raw_correlation_latent import (decode_correlations,
                                    raw_correlation_sequence)


def test_raw_channel_preserves_first_event_after_blank_interval():
    blank = torch.zeros((2, 32, 64))
    event = blank.clone()
    event[0, 16, 16] = 1
    codes = raw_correlation_sequence([blank, event, blank, blank, event])
    assert len(codes) == 5
    assert all(code.shape == (18, 32, 64) for code in codes)
    assert codes[1][0, 16, 16] == 1
    assert codes[4][0, 16, 16] == 1
    assert codes[1][2:].sum() == 0
    assert codes[4][2:].sum() == 0


def test_primary_forecast_does_not_decode_predicted_raw_channels():
    prediction = torch.zeros((18, 32, 64))
    prediction[0, 16, 16] = 1
    assert decode_correlations(prediction).sum() == 0
    prediction[3, 16, 16] = .7
    assert decode_correlations(prediction)[0, 16, 16] == torch.tensor(.7)
