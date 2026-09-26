"""Local signed adjacency learned from visual co-occurrence error."""

import math

import torch

from confident_event_surface import ConfidentEventSurface


class DiagonalAffinity:
    """A repeated local pair predicts same-sign neighboring occupancy."""

    def __init__(self):
        self.weight = torch.zeros((2, 2))
        self.sources = torch.zeros((2, 2))
        self.matches = torch.zeros((2, 2))

    def observe(self, source, target):
        for sign in range(2):
            for column, direction in enumerate((1, -1)):
                left = source[sign, :-1, :-1] if direction == 1 else source[
                    sign, :-1, 1:]
                right = target[sign, 1:, 1:] if direction == 1 else target[
                    sign, 1:, :-1]
                count = float(left.sum())
                matched = float((left & right).sum())
                if count:
                    # Mean local prediction-error update, accumulated online.
                    total = self.sources[sign, column]+count
                    error = matched-self.weight[sign, column]*count
                    self.weight[sign, column] += error/total
                    self.sources[sign, column] = total
                    self.matches[sign, column] += matched

    def links_above(self, null):
        result = {}
        for sign_index, sign in enumerate((-1., 1.)):
            for column, direction in enumerate((1, -1)):
                expected = float(null.matches[sign_index, column])
                observed = float(self.matches[sign_index, column])
                result[(sign, direction)] = (
                    observed > expected+5*math.sqrt(expected+1))
        return result


@torch.no_grad()
def fit_diagonal_affinity(episodes):
    samples = []
    for events in episodes:
        _, height, width = events[0].shape
        surface = ConfidentEventSurface(height=height, width=width)
        for event in events:
            surface.step(event)
            if bool(event.any()):
                visible = surface.confident
                samples.append(torch.stack((visible & (surface.contrast < 0),
                                            visible & (surface.contrast > 0))))
    aligned = DiagonalAffinity()
    null = DiagonalAffinity()
    shuffled = DiagonalAffinity()
    for index, sample in enumerate(samples):
        aligned.observe(sample, sample)
        null.observe(sample, samples[(index+len(samples)//3) % len(samples)])
        shuffled.observe(sample, samples[(index+2*len(samples)//3)
                                         % len(samples)])
    return dict(aligned=aligned.links_above(null),
                shuffled=shuffled.links_above(null),
                counts=dict(aligned=aligned.matches.tolist(),
                            null=null.matches.tolist(),
                            shuffled=shuffled.matches.tolist()),
                weights=dict(aligned=aligned.weight.tolist(),
                             shuffled=shuffled.weight.tolist()))
