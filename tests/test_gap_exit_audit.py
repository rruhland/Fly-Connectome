"""An additive-output gate cannot recover target events never proposed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gap_exit_audit import oracle_extra_f1


def test_oracle_extra_f1_keeps_fixed_errors_and_recovers_only_extra_false_alarms():
    fixed = dict(tp=2, fp=1, fn=3)
    fused = dict(tp=3, fp=2, fn=2)
    assert oracle_extra_f1(fixed, fused) == pytest.approx(6/9)


def test_oracle_extra_f1_requires_an_additive_forecast():
    with pytest.raises(ValueError):
        oracle_extra_f1(dict(tp=2, fp=1, fn=3),
                        dict(tp=1, fp=0, fn=4))
