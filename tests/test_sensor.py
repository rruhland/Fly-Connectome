import torch
import pytest

from fly_connectome.sensor import EventCamera, Retina


def test_polarity_offsets_and_persistent_reference():
    camera = EventCamera(2, 2, 3)
    frame = torch.zeros(2, 2, 3, dtype=torch.bool)
    frame[0, 0, 1] = True
    events = camera.observe(frame)
    assert events.pixels.tolist() == [1]
    assert events.on.tolist() == [True]
    assert events.offsets.tolist() == [0, 1, 1]
    assert camera.observe(frame).pixels.numel() == 0
    frame[0, 0, 1] = False
    frame[1, 1, 2] = True
    events = camera.observe(frame)
    assert events.pixels.tolist() == [1, 5]
    assert events.on.tolist() == [False, True]
    assert events.offsets.tolist() == [0, 1, 2]


def test_full_field_hex_projection_and_no_l4_injection():
    retina = Retina(2, 4, [[0, 0], [1, 0]], [0, 1, 1, 0],
                    ['L1', 'L2', 'L3', 'L4'], {'L1': 'on', 'L2': 'off', 'L3': 'off'})
    camera = EventCamera(1, 2, 4)
    frame = torch.ones(1, 2, 4, dtype=torch.bool)
    projected = retina.project(camera.observe(frame))
    assert projected.tolist() == [[1., 0., 0., 0.]]
    projected = retina.project(camera.observe(torch.zeros_like(frame)))
    assert projected.tolist() == [[0., 1., 1., 0.]]
    assert set(retina.pixel_bins.tolist()) == {0, 1}
    with pytest.raises(ValueError, match='L1-L3'):
        Retina(2, 4, [[0, 0]], [0], ['L4'], {'L4': 'on'})


def test_events_are_detached_and_input_is_binary_only():
    camera = EventCamera(1, 2, 2)
    with pytest.raises(ValueError, match='binary'):
        camera.observe(torch.ones(1, 2, 2, requires_grad=True))


def test_signed_contrast_drives_only_lamina_and_one_event_polarity():
    retina = Retina(1, 1, [[0, 0]], [0, 0, 0, 0],
                    ['L1', 'L2', 'L3', 'L4'], dict.fromkeys(['L1', 'L2', 'L3'], 'contrast'))
    camera = EventCamera(1, 1, 1)
    on = camera.observe(torch.ones(1, 1, 1, dtype=torch.bool))
    assert on.on.tolist() == [True]
    assert retina.project(on).tolist() == [[-1., -1., -1., 0.]]
    off = camera.observe(torch.zeros(1, 1, 1, dtype=torch.bool))
    assert off.on.tolist() == [False]
    assert retina.project(off).tolist() == [[1., 1., 1., 0.]]
