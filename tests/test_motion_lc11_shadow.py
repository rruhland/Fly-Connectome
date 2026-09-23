import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_lc11_shadow import (ShadowLC11, adaptive_release,
                                release_from_voltage, select_fraction)
from fly_connectome.dynamics import NeuronConfig


def test_shadow_readout_uses_measured_weight_and_one_tick_delay():
    bank = ShadowLC11([0], [0], [100.], 1, 1, NeuronConfig())
    assert not bank.step(torch.tensor([1.]))[0]
    assert bank.current[0] == 0
    assert bank.step(torch.tensor([0.]))[0]
    assert abs(float(bank.current[0])-100) < 1e-5


def test_shadow_readout_preserves_inhibitory_contact_sign():
    bank = ShadowLC11([0, 1], [0, 0], [100., -100.], 2, 1,
                      NeuronConfig())
    bank.step(torch.tensor([1., 1.]))
    assert not bank.step(torch.tensor([0., 0.]))[0]
    assert abs(float(bank.current[0])) < 1e-5


def test_release_is_nonnegative_local_and_bounded():
    voltage = torch.tensor([.001, .004, .010])
    rest = torch.tensor([.002, .001, .001])
    release = release_from_voltage(voltage, rest, .1)
    assert torch.allclose(release, torch.tensor([0., .1, .1]))


def test_adaptive_reference_uses_only_previous_local_voltage():
    baseline = torch.tensor([.001, .004])
    release = adaptive_release(torch.tensor([.004, .002]), baseline,
                               .1, .5)
    assert torch.allclose(release, torch.tensor([.1, 0.]))
    assert torch.allclose(baseline, torch.tensor([.0025, .003]))


def test_calibration_requires_both_polarities_and_quiet_blank():
    rows = [dict(fraction=.05, on_excess=6, off_excess=4,
                 blank_rate=0., peak_fraction=.1, finite=True),
            dict(fraction=.10, on_excess=5, off_excess=7,
                 blank_rate=.009, peak_fraction=.2, finite=True)]
    assert select_fraction(rows) == .10
    assert select_fraction(rows[:1]) is None
