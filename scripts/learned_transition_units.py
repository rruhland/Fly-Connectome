"""Opt-in unlabeled local transition units with delayed predictive credit."""

import torch
import torch.nn.functional as F


class TransitionPopulation:
    def __init__(self, *, channels=12, units=24, seed=0, homeostasis=True,
                 dictionary_eta=.08, prediction_eta=.5):
        self.channels = channels
        self.units = units
        self.homeostasis = homeostasis
        self.dictionary_eta = dictionary_eta
        self.prediction_eta = prediction_eta
        rng = torch.Generator().manual_seed(seed)
        self.dictionary = torch.rand((units, 2*channels*25),
                                     generator=rng)*.2
        self.predictive = torch.zeros((channels, units, 5, 5))
        self.usage = torch.zeros(units)
        self.total_assignments = 0
        self.dictionary_updates = 0
        self.predictive_updates = 0
        self.reset_state()

    def reset_state(self):
        self.previous = torch.zeros((self.channels, 32, 64))
        self.previous_sources = []
        self.pending_prediction = None
        self.latent = torch.zeros((self.units, 32, 64))

    def _predict(self, sources):
        canvas = torch.zeros((self.channels, 36, 68))
        for y, x, unit in sources:
            canvas[:, y:y+5, x:x+5] += self.predictive[:, unit]
        return canvas[:, 2:-2, 2:-2].clamp_(0, 1)

    @torch.no_grad()
    def step(self, current, *, learn_dictionary=False,
             learn_prediction=False, credit_target=None):
        if current.shape != (self.channels, 32, 64):
            raise ValueError('local sensory latent has wrong shape')
        patches = F.unfold(torch.cat((current, self.previous))[None],
                           kernel_size=5, padding=2)[0]
        sites = current.sum(0).flatten().nonzero().flatten()
        latent = torch.zeros((self.units, 32*64))
        sources = []
        if len(sites):
            selected = patches[:, sites]
            scores = self.dictionary @ selected
            if self.homeostasis:
                scores /= (self.dictionary.norm(dim=1)[:, None]
                           * selected.norm(dim=0)[None, :]).clamp(min=1e-6)
                scores -= .75*(self.usage/self.total_assignments
                                if self.total_assignments else self.usage)[:, None]
            winners = scores.argmax(0)
            latent[winners, sites] = 1
            sources = [(int(site//64), int(site%64), int(unit))
                       for site, unit in zip(sites, winners)]
            if learn_dictionary:
                self.usage += torch.bincount(winners, minlength=self.units)
                self.total_assignments += len(winners)
                for unit in winners.unique():
                    target = selected[:, winners == unit].mean(1)
                    self.dictionary[unit].lerp_(target, self.dictionary_eta)
                    self.dictionary_updates += 1
        latent = latent.view(self.units, 32, 64)

        if learn_prediction and self.pending_prediction is not None:
            error = (current if credit_target is None else credit_target
                     )-self.pending_prediction
            padded = F.pad(error[None], (2, 2, 2, 2))[0]
            denominator = max(len(self.previous_sources), 1)
            for y, x, unit in self.previous_sources:
                self.predictive[:, unit] += (self.prediction_eta/denominator
                                             * padded[:, y:y+5, x:x+5])
                self.predictive_updates += 1
            self.predictive.clamp_(0, 1)

        prediction = self._predict(sources)
        self.previous = current.clone()
        self.previous_sources = sources
        self.pending_prediction = prediction
        self.latent = latent
        return prediction
