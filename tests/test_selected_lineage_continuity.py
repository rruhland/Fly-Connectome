"""A reusable beam slot must not fabricate one continuous public track."""
from collections import deque
import copy

import numpy as np
import torch

from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.session import LearningSession, VISION_CONFIGURATION
from fly_connectome.vision.association import StatisticalObservationTracker
from fly_connectome.vision.interface import VisualStateEncoder
from fly_connectome.vision.memory import PatchAssociation
from fly_connectome.vision.state import StreamingVisualState


class ScriptedTracker(StatisticalObservationTracker):
    def __init__(self):
        super().__init__(height=32, width=32)
        self.measurements = []

    def proposals(self, sensory):
        return self.measurements


def history(x, *, seen=4, score=0., velocity=0.):
    return dict(position=torch.tensor([12., float(x)]),
                velocity=torch.tensor([0., float(velocity)]), last_seen=seen,
                mass=1., hits=5, score=score, proposal=None)


def switch_tracker():
    tracker = ScriptedTracker()
    tracker.frame = 4
    selected, alternate = history(4), history(24, score=-.1)
    tracker.slots = [{**selected, 'public_id': 41, 'confirmed': True,
                      'hypotheses': [selected, alternate]}]
    tracker.measurements = [(torch.tensor([12., 24.]), 1.)]
    return tracker


def test_observed_switch_preserves_beam_and_marks_parent_endpoint_break():
    tracker = switch_tracker()
    tracker.step(torch.zeros(8, 32, 32))
    slot = tracker.slots[0]
    assert slot['lineage_discontinuity']
    assert torch.equal(slot['position'], torch.tensor([12., 24.]))
    assert torch.equal(slot['velocity'], torch.zeros(2))
    assert len(slot['hypotheses']) == 4
    assert any(torch.equal(h['position'], torch.tensor([12., 4.]))
               for h in slot['hypotheses'])
    assert not any('_parent_is_selected' in h for h in slot['hypotheses'])


def test_missing_branch_switch_also_marks_break():
    tracker = switch_tracker()
    tracker.frame = 64
    tracker.slots[0]['last_seen'] = 0
    tracker.slots[0]['hypotheses'][0]['last_seen'] = 0
    tracker.slots[0]['hypotheses'][1]['last_seen'] = 60
    tracker.step(torch.zeros(8, 32, 32), observation_available=False)
    slot = tracker.slots[0]
    assert slot['lineage_discontinuity'] and slot['last_seen'] == 60


def test_endpoint_equality_normal_motion_occlusion_and_reacquisition_stay_continuous():
    tracker = ScriptedTracker()
    tracker.frame = 4
    selected = history(4, velocity=1.)
    tracker.slots = [{**copy.deepcopy(selected), 'confirmed': True,
                      'hypotheses': [selected]}]
    for x in (5, None, 7, 8):
        tracker.measurements = [] if x is None else [(torch.tensor([12., float(x)]), 1.)]
        tracker.step(torch.zeros(8, 32, 32), observation_available=x is not None)
        assert not tracker.slots[0]['lineage_discontinuity']
    assert torch.equal(tracker.slots[0]['velocity'], torch.tensor([0., 1.]))


def test_routing_revision_retires_context_forecasts_and_pending_credit():
    model = StreamingVisualState(height=32, width=32, memory=PatchAssociation())
    model.state.tracker = switch_tracker()
    model.next_id = 42
    model.state.keys[0] = torch.ones(51)
    model.state.confidence[0] = .001
    model.histories[41] = deque([torch.tensor([12., float(x)]) for x in range(5)], maxlen=5)
    model.forecast_cache[41, 6] = dict(id=41)
    model.pending[5] = [(41, 1, torch.ones(4, 2), torch.zeros(2), torch.zeros(2), None)]
    model.pending[8] = copy.deepcopy(model.pending[5])
    frame = torch.zeros(32, 32)
    result = model.step(torch.zeros(2, 32, 32), frame, learn=True)
    assert [e['id'] for e in result['entities']] == [42]
    assert not result['forecasts'] and set(model.histories) == {42}
    assert len(model.histories[42]) == 1
    assert not model.forecast_cache and not model.pending
    assert 0 not in model.state.keys and model.state.confidence[0] == 1.
    assert not len(model.state.memory.keys)
    assert not any(c.outcomes for c in model.calibration.values())
    assert not any(len(bank.keys) for bank in model.dynamics.values())


def test_occlusion_keeps_public_identity_and_encoder_velocity_until_reacquisition():
    model = StreamingVisualState(height=32, width=32)
    model.state.tracker = ScriptedTracker()
    tracker = model.state.tracker
    tracker.frame = 4
    selected = history(4, velocity=1.)
    tracker.slots = [{**copy.deepcopy(selected), 'public_id': 41,
                      'confirmed': True, 'hypotheses': [selected]}]
    model.next_id = 42
    encoder = VisualStateEncoder(32, 32, .02)
    frame = torch.zeros(32, 32)
    encoder.encode(dict(sample=4, entities=[dict(id=41, position=selected['position'],
        observed=True, observation_age=0, association_strength=1.)], forecasts=[]), frame)
    for x in (5, None, 7, 8):
        tracker.measurements = [] if x is None else [(torch.tensor([12., float(x)]), 1.)]
        result = model.step(torch.zeros(2, 32, 32), None if x is None else frame)
        encoded = encoder.encode(result, None if x is None else frame)
        entity = encoded['entities'][0]
        assert entity['id'] == 41 and entity['velocity_known']
        assert entity['observed'] == (x is not None)
        assert torch.equal(entity['observed_velocity'], torch.tensor([0., 1./(.02*32)]))
    assert model.next_id == 42


def test_revision_suppresses_encoder_jump_and_is_invariant_to_routing_names():
    outputs = []
    for offset in (0, 100):
        model = StreamingVisualState(height=32, width=32)
        model.state.tracker = switch_tracker()
        model.state.tracker.slots[0]['public_id'] += offset
        model.next_id = 42+offset
        encoder = VisualStateEncoder(32, 32, .02)
        frame = torch.zeros(32, 32)
        encoder.encode(dict(sample=4, entities=[dict(id=41+offset,
            position=torch.tensor([12., 4.]), observed=True, observation_age=0,
            association_strength=1.)], forecasts=[]), frame)
        result = model.step(torch.zeros(2, 32, 32), frame)
        encoded = encoder.encode(result, frame)
        entity = encoded['entities'][0]
        assert not entity['velocity_known']
        assert torch.equal(entity['observed_velocity'], torch.zeros(2))
        outputs.append(encoded)
    assert torch.equal(outputs[0]['indices'], outputs[1]['indices'])
    assert torch.equal(outputs[0]['values'], outputs[1]['values'])


def test_durable_recovery_preserves_revised_ids_and_all_retired_state(tmp_path):
    torch.set_num_threads(1)
    assert VISION_CONFIGURATION.endswith('+selected-lineage-continuity-v1')
    session = LearningSession(736187003, opponent_speed=.65, alpha=.001, lam=.995)
    # Seed the same generic ambiguous beam at a known durable boundary. Actual
    # camera frames then exercise the production tracker, not a test subclass.
    tracker = session.vision.state.tracker
    selected, alternate = history(4, seen=tracker.frame), history(24, seen=tracker.frame, score=.1)
    tracker.slots = [{**selected, 'public_id': 41, 'confirmed': True,
                      'hypotheses': [selected, alternate]}]
    session.vision.next_id = 42
    session.vision.histories[41] = deque([selected['position'].clone()], maxlen=9)
    session.vision.state.keys[0] = torch.ones(51)
    session.vision.pending[tracker.frame+3] = [(41, 1, torch.ones(4, 2),
        torch.zeros(2), torch.zeros(2), None)]
    initial = copy.deepcopy(session)
    a = DurableRun(tmp_path/'a.db', session=session, snapshot_transitions=2)
    records = [a.step() for _ in range(6)]
    expected, weights = fingerprint(a.session), a.session.learner.weights.copy()
    assert 41 not in a.session.vision.histories
    assert a.session.vision.next_id > 42
    a.stop()
    b = DurableRun(tmp_path/'b.db', session=initial, snapshot_transitions=2)
    actual = [b.step() for _ in range(3)]
    b.close_without_checkpoint()
    b = DurableRun(tmp_path/'b.db')
    assert b.replayed == 1
    actual.extend(b.step() for _ in range(3))
    assert actual == records and fingerprint(b.session) == expected
    np.testing.assert_array_equal(b.session.learner.weights, weights)
    b.stop()
