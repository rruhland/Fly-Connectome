"""Opt-in nearby-site inhibition with causal, decaying latent continuity."""

import torch


class SparseCoherentState:
    def __init__(self, *, decay=.9, radius=1):
        self.decay = decay
        self.radius = radius
        self.reset_state()

    def reset_state(self):
        self.state = None

    @torch.no_grad()
    def step(self, observed):
        retained = (torch.zeros_like(observed) if self.state is None
                    else self.decay*self.state)
        candidates = torch.maximum(observed, retained)
        amplitude, winner = candidates.max(0)
        sites = (amplitude >= .5).nonzero(as_tuple=False).tolist()
        sites.sort(key=lambda site: float(amplitude[site[0], site[1]]),
                   reverse=True)
        suppressed = torch.zeros_like(amplitude, dtype=torch.bool)
        output = torch.zeros_like(observed)
        for y, x in sites:
            if suppressed[y, x]:
                continue
            output[int(winner[y, x]), y, x] = amplitude[y, x]
            suppressed[max(y-self.radius, 0):y+self.radius+1,
                       max(x-self.radius, 0):x+self.radius+1] = True
        self.state = output
        return output
