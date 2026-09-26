"""Opt-in delayed observer credit from locally observed polarity returns."""

import torch

from local_observation_model import LocalObservationModel


class PolarityReturnTeacher:
    def __init__(self, *, window=8):
        self.window = window
        self.reset_state()

    def reset_state(self):
        self.pending = []

    def step(self, raw, features):
        for entry in self.pending:
            entry['target'] = torch.maximum(entry['target'],
                                             entry['raw']*raw.flip(0))
        self.pending.append(dict(raw=raw.clone(), features=features,
                                 target=torch.zeros_like(raw)))
        if len(self.pending) <= self.window:
            return None
        entry = self.pending.pop(0)
        return entry['features'], entry['raw'], entry['target']


@torch.no_grad()
def fit_return_observer(episodes):
    observer = LocalObservationModel()
    teacher = PolarityReturnTeacher()
    for episode in episodes:
        observer.reset_state()
        teacher.reset_state()
        for raw in episode:
            _, features = observer.step(raw)
            credit = teacher.step(raw, features)
            if credit is not None:
                observer.credit(*credit)
    observer.reset_state()
    return observer
