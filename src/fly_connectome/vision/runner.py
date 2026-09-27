"""Offline tensor-stream entrypoint for the production online vision API."""

import time

import torch

from . import ProbabilisticVisualState, load_default


def run_stream(path, output, *, checkpoint=None, save=None, learn=False):
    model = load_default() if checkpoint is None else ProbabilisticVisualState.load(checkpoint)
    data = torch.load(path, weights_only=True, map_location='cpu')
    frames, events = data['frames'], data['events']
    shape = (model.observer.height, model.observer.width)
    if frames.ndim != 3 or tuple(frames.shape[1:]) != shape or events.shape != (len(frames), 2, *shape):
        raise ValueError('stream requires frames[T,H,W] and events[T,2,H,W] matching the checkpoint')
    available = data.get('available', torch.ones(len(frames), dtype=torch.bool))
    if available.dtype != torch.bool or available.shape != (len(frames),):
        raise ValueError('available must be a boolean vector of camera-sample length')
    states = []
    started = time.perf_counter()
    for event, frame, present in zip(events, frames, available):
        states.append(model.step(event, frame if present else None, learn=learn))
    seconds = time.perf_counter()-started
    torch.save(dict(sensor='grayscale-every-sample-plus-events', states=states), output)
    if save is not None:
        model.save(save)
    return dict(camera_samples=len(frames), seconds=seconds,
                samples_per_second=len(frames)/max(seconds, 1e-9), learning=learn, output=str(output))
