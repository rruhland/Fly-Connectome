"""Opt-in local Hebbian/recurrent visual latent learned from raw events."""

import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from generic_local_transition import (LocalTripletLearner, accumulate,
                                      empty_score, event_sequence, finish, item)
from generic_motion_probe import SHAPES


OUT = Path('docs/experiments/2026-09-23-learned-latent-results.json')
DIRECTIONS = ('up', 'down', 'left', 'right')
THRESHOLDS = (.05, .1, .2, .3, .5)


class LocalVisualLatent:
    """Unlabeled event dictionary and local latent-prediction synapses."""

    def __init__(self, *, channels=12, seed=0, sensory_eta=.08,
                 recurrent_eta=.5, homeostasis=False):
        self.channels = channels
        self.sensory_eta = sensory_eta
        self.recurrent_eta = recurrent_eta
        self.homeostasis = homeostasis
        rng = torch.Generator().manual_seed(seed)
        self.sensory = torch.rand((channels, 4*25), generator=rng)*.2
        self.recurrent = torch.zeros((channels, channels, 5, 5))
        self.reset_state()
        self.sensory_updates = 0
        self.recurrent_updates = 0
        self.usage = torch.zeros(channels)
        self.total_assignments = 0

    def reset_state(self):
        self.previous_events = torch.zeros((2, 32, 64))
        self.previous_sources = []
        self.pending_latent_prediction = None
        self.latent = torch.zeros((self.channels, 32, 64))

    def _predict_latent(self, sources):
        canvas = torch.zeros((self.channels, 36, 68))
        for y, x, code in sources:
            canvas[:, y:y+5, x:x+5] += self.recurrent[:, code]
        return canvas[:, 2:-2, 2:-2].clamp_(0, 1)

    @torch.no_grad()
    def step(self, current, *, learn=True, learn_sensory=True):
        if current.shape != (2, 32, 64):
            raise ValueError('two-polarity 32x64 event map required')
        patches = F.unfold(torch.cat((current, self.previous_events))[None],
                           kernel_size=5, padding=2)[0]
        sites = current.sum(0).flatten().nonzero().flatten()
        latent = torch.zeros((self.channels, 32*64))
        sources = []
        if len(sites):
            selected = patches[:, sites]
            scores = self.sensory @ selected
            if self.homeostasis:
                scores /= (self.sensory.norm(dim=1)[:, None]
                           * selected.norm(dim=0)[None, :]).clamp(min=1e-6)
                scores -= .75*(self.usage/self.total_assignments
                                if self.total_assignments else self.usage)[:, None]
            winners = scores.argmax(0)
            latent[winners, sites] = 1
            sources = [(int(site//64), int(site%64), int(code))
                       for site, code in zip(sites, winners)]
            if learn:
                self.usage += torch.bincount(winners, minlength=self.channels)
                self.total_assignments += len(winners)
        latent = latent.view(self.channels, 32, 64)

        if learn and self.pending_latent_prediction is not None:
            error = latent-self.pending_latent_prediction
            padded = F.pad(error[None], (2, 2, 2, 2))[0]
            denominator = max(len(self.previous_sources), 1)
            for y, x, code in self.previous_sources:
                self.recurrent[:, code] += (self.recurrent_eta/denominator
                                            * padded[:, y:y+5, x:x+5])
                self.recurrent_updates += 1
            self.recurrent.clamp_(0, 1)

        if learn and learn_sensory and len(sites):
            for code in winners.unique():
                indices = winners == code
                target = selected[:, indices].mean(1)
                self.sensory[code].lerp_(target, self.sensory_eta)
                self.sensory_updates += 1

        predicted_latent = self._predict_latent(sources)
        center = self.sensory[:, [12, 37]]
        predicted_events = torch.einsum('kc,khw->chw', center,
                                        predicted_latent).clamp_(0, 1)
        self.previous_events = current.clone()
        self.previous_sources = sources
        self.pending_latent_prediction = predicted_latent
        self.latent = latent
        return predicted_events


def training_sequences():
    cases = []
    for direction in DIRECTIONS:
        centers = ((16, 20), (16, 44)) if direction in ('up', 'down') else (
            (10, 32), (22, 32))
        for center in centers:
            for speed in (1, 2):
                for background in (True, False):
                    cases.append((direction, event_sequence(
                        [item('square', center, direction, speed)],
                        background=background)))
    return cases


def heldout_sequences():
    cases = []
    for shape in SHAPES:
        if shape == 'square':
            continue
        for direction in DIRECTIONS:
            for speed in (1, 2):
                for background in (True, False):
                    cases.append((shape, direction, speed, background,
                                  event_sequence([item(shape, (16, 32),
                                                       direction, speed)],
                                                 background=background)))
    return cases


@torch.no_grad()
def feature(model, sequence, *, include_recurrent):
    model.reset_state()
    counts = torch.zeros(model.channels)
    predicted_counts = torch.zeros(model.channels)
    for t, event in enumerate(sequence):
        model.step(event, learn=False)
        if 3 <= t < 14:
            counts += model.latent.sum((1, 2))
            predicted_counts += model.pending_latent_prediction.sum((1, 2))
    if include_recurrent:
        return torch.cat((counts, predicted_counts))/counts.sum().clamp(min=1)
    return counts/counts.sum().clamp(min=1)


def nearest_centroid_probe(model, train, heldout, *, include_recurrent):
    by_direction = {direction: [] for direction in DIRECTIONS}
    for direction, sequence in train:
        by_direction[direction].append(feature(model, sequence,
                                               include_recurrent=include_recurrent))
    centroids = torch.stack([torch.stack(by_direction[direction]).mean(0)
                             for direction in DIRECTIONS])
    results = {}
    correct = 0
    for shape, direction, speed, background, sequence in heldout:
        vector = feature(model, sequence, include_recurrent=include_recurrent)
        predicted = DIRECTIONS[torch.cdist(vector[None], centroids).argmin()]
        success = predicted == direction
        correct += success
        results[f'{shape}:{direction}:s{speed}:'
                f'{"dark" if background else "bright"}'] = dict(
                    predicted=predicted, correct=success)
    return dict(correct=correct, total=len(heldout),
                accuracy=correct/len(heldout), cases=results)


def add_threshold_counts(counts, prediction, target):
    actual = target > 0
    for threshold, row in counts.items():
        selected = prediction >= threshold
        row['tp'] += int((selected & actual).sum())
        row['fp'] += int((selected & ~actual).sum())
        row['fn'] += int((~selected & actual).sum())


def finish_threshold_counts(counts):
    return {str(threshold): 2*row['tp']/(2*row['tp']+row['fp']+row['fn'])
            if 2*row['tp']+row['fp']+row['fn'] else 0.
            for threshold, row in counts.items()}


def empty_threshold_counts():
    return {threshold: dict(tp=0, fp=0, fn=0) for threshold in THRESHOLDS}


def forecast_scores(models, heldout):
    scores = {name: empty_score() for name in models}
    thresholds = {name: empty_threshold_counts() for name in models}
    groups = {}
    for shape, direction, speed, background, sequence in heldout:
        label = f'{shape}:s{speed}:{"dark" if background else "bright"}'
        group = groups.setdefault(label, {name: empty_score() for name in models})
        for model in models.values():
            model.reset_state()
        for t, event in enumerate(sequence):
            predictions = {name: model.step(event, learn=False)
                           for name, model in models.items()}
            if 3 <= t < 14:
                for name, prediction in predictions.items():
                    accumulate(scores[name], prediction, sequence[t+1])
                    accumulate(group[name], prediction, sequence[t+1])
                    add_threshold_counts(thresholds[name], prediction,
                                         sequence[t+1])
    return dict(overall={name: finish(value) for name, value in scores.items()},
                threshold_f1={name: finish_threshold_counts(value)
                              for name, value in thresholds.items()},
                groups={label: {name: finish(value) for name, value in methods.items()}
                        for label, methods in groups.items()})


@torch.no_grad()
def latent_forecast_scores(models, heldout):
    scores = {name: empty_score() for name in models}
    thresholds = {name: empty_threshold_counts() for name in models}
    by_speed = {speed: {name: empty_score() for name in models}
                for speed in (1, 2)}
    for _, _, speed, _, sequence in heldout:
        for name, model in models.items():
            model.reset_state()
            predictions = []
            observations = []
            for event in sequence:
                model.step(event, learn=False)
                predictions.append(model.pending_latent_prediction.clone())
                observations.append(model.latent.clone())
            for t in range(3, 14):
                accumulate(scores[name], predictions[t], observations[t+1])
                accumulate(by_speed[speed][name], predictions[t],
                           observations[t+1])
                add_threshold_counts(thresholds[name], predictions[t],
                                     observations[t+1])
    return dict(overall={name: finish(value) for name, value in scores.items()},
                threshold_f1={name: finish_threshold_counts(value)
                              for name, value in thresholds.items()},
                by_speed={speed: {name: finish(value) for name, value in methods.items()}
                          for speed, methods in by_speed.items()})


@torch.no_grad()
def channel_usage(model, heldout):
    counts = torch.zeros(model.channels)
    for _, _, _, _, sequence in heldout:
        model.reset_state()
        for t, event in enumerate(sequence):
            model.step(event, learn=False)
            if 3 <= t < 14:
                counts += model.latent.sum((1, 2))
    frequencies = counts/counts.sum().clamp(min=1)
    active = frequencies > 0
    entropy = -(frequencies[active]*frequencies[active].log()).sum()
    return dict(fractions=frequencies.tolist(),
                effective_channels=float(entropy.exp()))


@torch.no_grad()
def training_fit(model, train):
    sensory_score = empty_score()
    latent_score = empty_score()
    for _, sequence in train:
        model.reset_state()
        predictions = []
        latent_predictions = []
        observations = []
        for event in sequence:
            predictions.append(model.step(event, learn=False))
            latent_predictions.append(model.pending_latent_prediction.clone())
            observations.append(model.latent.clone())
        for t in range(3, 14):
            accumulate(sensory_score, predictions[t], sequence[t+1])
            accumulate(latent_score, latent_predictions[t], observations[t+1])
    return dict(event=finish(sensory_score), latent=finish(latent_score))


@torch.no_grad()
def main(*, homeostasis=False):
    torch.set_num_threads(1)
    started = time.perf_counter()
    train = training_sequences()
    heldout = heldout_sequences()
    learned = LocalVisualLatent(seed=0, homeostasis=homeostasis)
    random_sensory = LocalVisualLatent(seed=0, homeostasis=homeostasis)
    initial_sensory = learned.sensory.clone()
    rng = random.Random(0)
    for _ in range(3):
        rng.shuffle(train)
        for _, sequence in train:
            learned.reset_state()
            for event in sequence:
                learned.step(event)

    shuffled = LocalVisualLatent(seed=0, homeostasis=homeostasis)
    shuffled.sensory.copy_(learned.sensory)
    shuffled.usage.copy_(learned.usage)
    shuffled.total_assignments = learned.total_assignments
    rng = random.Random(1)
    for _ in range(3):
        rng.shuffle(train)
        for _, sequence in train:
            shuffled.reset_state()
            permuted = sequence.copy()
            rng.shuffle(permuted)
            for event in permuted:
                shuffled.step(event, learn_sensory=False)
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    probe = dict(learned_state=nearest_centroid_probe(
                    learned, train, heldout, include_recurrent=True),
                 shuffled_state=nearest_centroid_probe(
                    shuffled, train, heldout, include_recurrent=True),
                 learned_sensory=nearest_centroid_probe(
                    learned, train, heldout, include_recurrent=False),
                 random_sensory=nearest_centroid_probe(
                    random_sensory, train, heldout, include_recurrent=False))
    forecast = forecast_scores(dict(learned=learned, shuffled_time=shuffled,
                                    fixed_correlation=fixed), heldout)
    latent_forecast = latent_forecast_scores(dict(learned=learned,
                                                  shuffled_time=shuffled), heldout)
    result = dict(training_episodes=3*len(train),
                  sensory_updates=learned.sensory_updates,
                  recurrent_updates=learned.recurrent_updates,
                  sensory_change=float((learned.sensory-initial_sensory).abs().mean()),
                  channel_usage=dict(learned=channel_usage(learned, heldout),
                                     random=channel_usage(random_sensory, heldout)),
                  probe=probe, forecast=forecast,
                  latent_forecast=latent_forecast,
                  training_fit=training_fit(learned, train),
                  elapsed_seconds=time.perf_counter()-started)
    output = OUT if not homeostasis else OUT.with_name(
        '2026-09-23-learned-latent-homeostasis-results.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(training_episodes=result['training_episodes'],
                          sensory_change=result['sensory_change'],
                          effective_channels={name: row['effective_channels']
                                              for name, row in result['channel_usage'].items()},
                          probe={name: dict(accuracy=value['accuracy'],
                                            correct=value['correct'])
                                 for name, value in probe.items()},
                          forecast={name: dict(f1=value['f1'],
                                               candidate_mse=value['candidate_mse'])
                                    for name, value in forecast['overall'].items()},
                          latent_forecast={name: dict(f1=value['f1'],
                                                      candidate_mse=value['candidate_mse'])
                                           for name, value in latent_forecast['overall'].items()},
                          training_fit={name: round(value['f1'], 3)
                                        for name, value in result['training_fit'].items()},
                          elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main(homeostasis='--homeostasis' in sys.argv[1:])
