"""Opt-in sparse local motion assemblies with Hebbian transition affinity."""

import torch


def local_candidates(event, observed, *, radius=1):
    """Choose nonredundant event sites and collect local sensory evidence."""
    strength = event.sum(0)+.1*observed.sum(0)
    sites = (event.sum(0) > 0).nonzero(as_tuple=False).tolist()
    sites.sort(key=lambda site: float(strength[site[0], site[1]]),
               reverse=True)
    height, width = strength.shape
    blocked = torch.zeros((height, width), dtype=torch.bool)
    proposals = []
    for y, x in sites:
        if blocked[y, x]:
            continue
        low_y, high_y = max(0, y-radius), min(height, y+radius+1)
        low_x, high_x = max(0, x-radius), min(width, x+radius+1)
        seen_event = event[:, low_y:high_y, low_x:high_x].sum((1, 2))
        seen_code = observed[:, low_y:high_y, low_x:high_x].sum((1, 2))
        seen_event /= seen_event.sum().clamp(min=1e-8)
        seen_code /= seen_code.sum().clamp(min=1e-8)
        proposals.append((y, x, torch.cat((seen_event, seen_code))/2))
        blocked[low_y:high_y, low_x:high_x] = True
    return proposals


class LocalTransitionAffinity:
    """A shared local association kernel learned from successive event sets."""

    def __init__(self, *, features, radius=3):
        self.radius = radius
        width = 2*radius+1
        self.counts = torch.zeros((width, width, features, features))
        self.weights = None

    @torch.no_grad()
    def observe(self, previous, current):
        for sy, sx, source in previous:
            for ty, tx, target in current:
                dy, dx = ty-sy, tx-sx
                if max(abs(dy), abs(dx)) <= self.radius:
                    self.counts[dy+self.radius, dx+self.radius] += torch.outer(
                        target, source)

    @torch.no_grad()
    def finalize(self):
        area = self.counts.shape[0]*self.counts.shape[1]
        self.weights = ((self.counts+.01)/
                        (self.counts.sum((0, 1), keepdim=True)+.01*area))

    def score(self, previous, current):
        sy, sx, source = previous
        ty, tx, target = current
        dy, dx = ty-sy, tx-sx
        if max(abs(dy), abs(dx)) > self.radius:
            return 0.
        return float(target @ self.weights[
            dy+self.radius, dx+self.radius] @ source)


class MotionAssemblies:
    """Causal tokens that compete for local observations and survive gaps."""

    def __init__(self, *, affinity=None, radius=3, max_gap=8):
        self.affinity = affinity
        self.radius = radius
        self.max_gap = max_gap
        self.reset_state()

    def reset_state(self):
        self.tokens = []

    @torch.no_grad()
    def step(self, frame, candidates):
        choices = []
        for token_index, token in enumerate(self.tokens):
            age = frame-token['last_frame']
            if age < 1 or age > self.max_gap:
                continue
            source = (token['y'], token['x'], token['feature'])
            for candidate_index, candidate in enumerate(candidates):
                y, x, _ = candidate
                distance = max(abs(y-token['y']), abs(x-token['x']))
                if distance > self.radius:
                    continue
                score = 1-distance/(self.radius+1)
                if self.affinity is not None:
                    score += 2*self.affinity.score(source, candidate)
                choices.append((score, token_index, candidate_index))
        choices.sort(reverse=True)
        used_tokens = set()
        used_candidates = set()
        for _, token_index, candidate_index in choices:
            if token_index in used_tokens or candidate_index in used_candidates:
                continue
            used_tokens.add(token_index)
            used_candidates.add(candidate_index)
            token = self.tokens[token_index]
            y, x, feature = candidates[candidate_index]
            token['y'], token['x'] = y, x
            token['feature'] = (token['feature']+feature)/2
            token['last_frame'] = frame
            token['history'].append((frame, y, x))
        for index, (y, x, feature) in enumerate(candidates):
            if index not in used_candidates:
                self.tokens.append(dict(y=y, x=x, feature=feature.clone(),
                                        last_frame=frame,
                                        history=[(frame, y, x)]))
