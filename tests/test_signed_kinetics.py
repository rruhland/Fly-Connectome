import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from signed_kinetics import SignedKineticsNetwork
from frame_prediction import FramePrediction
from fly_connectome.dynamics import Network, NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig


def network(cls):
    graph = Graph.from_contacts([10, 20, 30], [10, 20], [30, 30], [1, 1], [1, -1, 1], .5)
    return cls(graph, [1, 1], ['predictive', 'predictive'],
               config=NeuronConfig(dt=.001, tau_current=.005, threshold=100.))


def test_signed_currents_and_local_eligibility_use_matching_decay():
    net = network(SignedKineticsNetwork)
    rule = FramePrediction(net, LearningConfig(visual_target='input-arrivals-v1',
                           visual_eligibility='forecast-causal-v1', homeostasis_rate=0.))
    net.step(torch.zeros(1, 3))
    net.history[0, 0, :2] = True
    for t in range(12):
        a = net.step(torch.zeros(1, 3), capture_increments=True)
        rule.observe(a, torch.zeros(1))
        expected = .5*(math.exp(-t*.001/.020)-math.exp(-t*.001/.005))
        torch.testing.assert_close(a.predicted[0, 2], torch.tensor(expected), atol=1e-7, rtol=1e-6)
        signed = torch.tensor([math.exp(-t*.001/.020), -math.exp(-t*.001/.005)])
        torch.testing.assert_close(rule.values, signed, atol=1e-7, rtol=1e-6)
        assert net.excitatory_prediction.min() >= 0
        assert net.inhibitory_prediction.max() <= 0


def test_default_network_retains_original_common_decay():
    net = network(Network)
    torch.testing.assert_close(net.visual_decay(torch.tensor([0, 1])),
                               torch.full((2,), math.exp(-.001/.005)))
    net.predictive_current.fill_(.4)
    a = net.step(torch.zeros(1, 3))
    torch.testing.assert_close(a.predicted, torch.full((1, 3), .4*math.exp(-.001/.005)))


def test_warmup_spikes_reconstruct_long_lived_predictive_current(tmp_path):
    import numpy as np
    from test_training import trainer
    from controlled_visual import crop_payload, make_network
    from visual_preflight import delayed_traces
    model = trainer()
    path = tmp_path/'source.pt'
    model.save(path)
    payload = torch.load(path, weights_only=True)
    m = payload['metadata']
    m['pathways'] = ['predictive']*3
    m['config']['warmup_steps'] = 50
    m['neurons']['class_parameters'] = {t: {'rest_current': 2.} for t in m['retina']['cell_types']}
    crop, _, _ = crop_payload(payload, [0, 1, 2, 3])
    net = make_network(crop, m, predictive_kinetics='slow-excitation-v1', capture_warmup=True)
    assert net.warmup_spikes.any()
    a = net.step(torch.zeros_like(net.voltage))
    history = np.concatenate((net.warmup_spikes, a.spikes.numpy()))
    trace = delayed_traces(history, net.pre.numpy(), net.delays.numpy(), net.signs.numpy(),
                           net.visual_decay(torch.arange(net.e)).numpy())[-1]
    reconstructed = np.zeros(net.n)
    np.add.at(reconstructed, net.post.numpy(), trace*net.magnitudes.numpy())
    np.testing.assert_allclose(reconstructed, a.predicted[0].numpy(), atol=1e-6, rtol=1e-6)


def test_area_matched_impulse_integral_and_eligibility():
    from signed_kinetics import AreaMatchedKineticsNetwork
    graph = Graph.from_contacts([10, 20, 30], [10, 20], [30, 30], [1, 1], [1, -1, 1], .5)
    dt = 1/960
    net = AreaMatchedKineticsNetwork(graph, [1, 1], ['predictive']*2,
                                    config=NeuronConfig(dt=dt, threshold=100.))
    rule = FramePrediction(net, LearningConfig(visual_target='input-arrivals-v1',
                           visual_eligibility='forecast-causal-v1', homeostasis_rate=0.))
    gain = (1-math.exp(-dt/.020))/(1-math.exp(-dt/.005))
    net.step(torch.zeros(1, 3))
    net.history[0, 0, :2] = True
    area = 0.
    for tick in range(400):
        a = net.step(torch.zeros(1, 3), capture_increments=True)
        if tick < 12:
            rule.observe(a, torch.zeros(1))
            expected = torch.tensor([gain*math.exp(-tick*dt/.020), -math.exp(-tick*dt/.005)])
            torch.testing.assert_close(rule.values, expected, atol=1e-7, rtol=1e-6)
        area += net.excitatory_prediction[0, 2].item()*dt
        expected_current = .5*(gain*math.exp(-tick*dt/.020)-math.exp(-tick*dt/.005))
        torch.testing.assert_close(a.predicted[0, 2], torch.tensor(expected_current), atol=1e-7, rtol=1e-6)
    original_area = .5*dt/(1-math.exp(-dt/.005))
    assert abs(area/original_area-1) < 1e-6
