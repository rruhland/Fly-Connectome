import sys
import hashlib
from importlib.resources import files
from pathlib import Path

import pytest
import torch

from fly_connectome.vision import ProbabilisticVisualState, load_default, load_legacy_default

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probabilistic_visual_state import ProbabilisticVisualState as Reference


def test_bundled_checkpoint_is_the_approved_artifact():
    data = files('fly_connectome.vision').joinpath('data/default.pt').read_bytes()
    assert hashlib.sha256(data).hexdigest() == 'a899338c94278854dc9a790c21e3cba746658abb7ac91fc3f85c84754f253173'


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('learn', [False, True])
def test_production_matches_approved_runtime_and_checkpoint(tmp_path, learn):
    torch.set_num_threads(1)
    model = load_legacy_default()
    checkpoint = tmp_path/'vision.pt'
    model.save(checkpoint)
    reference = Reference.load(checkpoint)
    previous = torch.zeros(64, 64)
    for t in range(24):
        image = torch.zeros(64, 64)
        image[30:33, 5+t:8+t] = 1.
        events = torch.stack(((previous-image).clamp(min=0), (image-previous).clamp(min=0)))
        previous = image
        if 15 <= t < 18:
            image, events = None, torch.zeros_like(events)
        equal(model.step(events, image, learn=learn), reference.step(events, image, learn=learn))
    model.save(checkpoint)
    reference.save(tmp_path/'reference.pt')
    equal(torch.load(checkpoint, weights_only=True), torch.load(tmp_path/'reference.pt', weights_only=True))
    loaded = ProbabilisticVisualState.load(checkpoint)
    assert loaded.observer.height == 64


def test_invalid_sensor_is_rejected_before_advancing_state():
    model = load_default()
    with pytest.raises(ValueError, match='shape'):
        model.step(torch.zeros(2, 32, 64), torch.zeros(64, 64))
    assert model.state.tracker.frame == -1
    with pytest.raises(ValueError, match='finite'):
        model.step(torch.zeros(2, 64, 64), torch.full((64, 64), torch.nan))
    assert model.state.tracker.frame == -1


def test_float64_sensor_matches_float32_baseline():
    a, b = load_default(), load_default()
    event = torch.zeros(2, 64, 64)
    frame = torch.zeros(64, 64)
    event[1, 10, 10] = frame[10, 10] = 1.
    equal(a.step(event.double(), frame.double()), b.step(event, frame))


def test_production_cli_uses_bundled_baseline(tmp_path, monkeypatch, capsys):
    from fly_connectome.__main__ import main
    stream = tmp_path/'stream.pt'
    output = tmp_path/'records.pt'
    checkpoint = tmp_path/'learned.pt'
    torch.save(dict(events=torch.zeros(3, 2, 64, 64), frames=torch.zeros(3, 64, 64),
                    available=torch.tensor([True, False, True])), stream)
    monkeypatch.setattr(sys, 'argv', ['fly_connectome', 'vision-run', str(stream),
        '--output', str(output), '--save', str(checkpoint), '--learn'])
    main()
    records = torch.load(output, weights_only=True)
    assert len(records['states']) == 3
    assert records['sensor'] == 'grayscale-every-sample-plus-events'
    assert checkpoint.exists()
    assert 'camera_samples' in capsys.readouterr().out
