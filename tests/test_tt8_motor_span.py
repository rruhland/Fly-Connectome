import numpy as np
import pytest

pytest.importorskip('mujoco')
from scripts.experimental.tt4_transfer_trial import ROOT
from scripts.experimental.tt6_memory import load, save
from scripts.experimental.tt8_motor_span import attach


def test_archived_one_axis_bank_is_distinct_from_later_multi_axis_rewards(tmp_path):
    path = tmp_path/'next.gz'
    reference = ROOT/'artifacts/tt6/continual-player.json.gz'
    attach(reference, ROOT/'artifacts/tt4/mini-first-person-player.json.gz', path)
    library, prior, response, timing = load(path)
    before = prior.span_memory.profiles[1].state()
    prior.span_memory.identify(np.eye(3))
    prior.observe(1, 0, 1., opportunity=True, outgoing=True)
    assert prior.span_memory.profiles[1].state() == before
    saved = tmp_path/'saved.gz'
    save(saved, library, prior, response, timing)
    _, fresh, _, _ = load(saved)
    assert fresh.span_memory.active is None
    assert fresh.span_memory.profiles[2].state() == prior.span_memory.profiles[2].state()
    old_two = fresh.span_memory.profiles[2].state()
    fresh.span_memory.observe_for_response(np.array([[1.], [0.]]), 0, 1.)
    assert fresh.span_memory.profiles[2].state() == old_two
    assert fresh.span_memory.profiles[1].counts[0] == before['counts'][0]+1
    assert fresh.numeric[1][0] == prior.numeric[1][0]+1
