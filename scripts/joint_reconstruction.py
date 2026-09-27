"""Opt-in joint event reconstruction and sparse-frame local self-supervision."""

import torch
import torch.nn.functional as F


class TemporalPatchObserver:
    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.dictionary = torch.zeros(32, 200)
        self.filled = 0
        self.generator = torch.Generator().manual_seed(710)
        self.reset_state()

    def reset_state(self):
        self.history = torch.zeros(4, 2, self.height, self.width)
        self.fast = torch.zeros(2, self.height, self.width)
        self.slow = torch.zeros_like(self.fast)
        self.contrast = torch.zeros(self.height, self.width)

    @torch.no_grad()
    def step(self, raw, *, absolute=None, learn=False):
        self.history = torch.cat((self.history[1:], raw[None]))
        patches = F.unfold(self.history.reshape(1, 8, self.height, self.width),
                           5, padding=2)[0]
        if learn:
            active = ((raw.sum(0).flatten() > 0) &
                      (patches.sum(0) >= 2)).nonzero().flatten()
            selected = active[torch.randperm(len(active), generator=self.generator)[:64]]
            for pixel in selected:
                if self.filled == len(self.dictionary):
                    break
                atom = patches[:, pixel]
                self.dictionary[self.filled] = atom/atom.norm().clamp(min=1e-8)
                self.filled += 1
        residual = patches.clone()
        codes = torch.zeros(32, patches.shape[1])
        for _ in range(3):
            drive = self.dictionary @ residual
            strength, winner = drive.max(0)
            strength = (strength-.5).clamp(min=0.)
            codes.scatter_add_(0, winner[None], strength[None])
            residual -= self.dictionary[winner].T*strength
        reconstruction = self.dictionary.T @ codes
        # Decode the current central pixel; past slices are inference evidence.
        filtered = reconstruction[[6*25+12, 7*25+12]].reshape_as(raw).clamp(0., 1.)
        if learn and self.filled:
            usage = codes.sum(1).clamp(min=1.)
            self.dictionary.add_(.02*(codes @ (patches-reconstruction).T)
                                 /usage[:, None]).clamp_(min=0.)
            self.dictionary.div_(self.dictionary.norm(dim=1, keepdim=True)
                                  .clamp(min=1e-8))
        self.fast.mul_(.65).add_(filtered).clamp_(0., 1.)
        self.slow.mul_(.95).add_(filtered).clamp_(0., 1.)
        self.contrast.add_(filtered[1]-filtered[0]).clamp_(-1., 1.)
        if absolute is not None:
            self.contrast.copy_(absolute)
        return filtered, patches


class SparseFrameObserver:
    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.weights = torch.zeros(46)
        self.weights[4] = 1.  # observed signed change
        self.weights[2*9+4] = 1.  # prior reconstruction
        self.reset_state()

    def reset_state(self):
        self.fast = torch.zeros(2, self.height, self.width)
        self.slow = torch.zeros_like(self.fast)
        self.contrast = torch.zeros(self.height, self.width)

    @torch.no_grad()
    def step(self, raw, *, absolute=None, learn=False):
        features = torch.stack((raw[1]-raw[0], raw.sum(0), self.contrast,
                                self.fast[1]-self.fast[0],
                                self.slow[1]-self.slow[0]))
        patches = F.unfold(features[None], 3, padding=1)[0]
        patches = torch.cat((patches, torch.ones(1, patches.shape[1])))
        predicted = (self.weights @ patches).reshape_as(self.contrast).clamp(-1., 1.)
        filtered = raw
        if absolute is not None and learn:
            error = (absolute-predicted).flatten()
            exposure = 1+15*absolute.abs().flatten()
            normalized = error*exposure/patches.square().sum(0).clamp(min=1.)
            self.weights.add_(.05*(patches @ normalized)/exposure.sum())
            self.weights.clamp_(-2., 2.)
        self.contrast = predicted if absolute is None else absolute.clone()
        self.fast.mul_(.65).add_(filtered).clamp_(0., 1.)
        self.slow.mul_(.95).add_(filtered).clamp_(0., 1.)
        return filtered, patches
