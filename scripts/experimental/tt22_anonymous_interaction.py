"""Learn which anonymous scene pairs anticipate a change of motion."""

from dataclasses import dataclass

import numpy as np

from .tt22_relation import learn, predict


HORIZON = 3
UNARY = 3


def describe(actor, mover):
    delta = np.subtract(mover.position, actor.position)
    a, b = np.asarray(actor.velocity), np.asarray(mover.velocity)
    relative = b - a
    gap = np.linalg.norm(delta)
    closure = -delta @ relative / max(gap, .001)
    closest = np.clip(-delta @ relative / max(relative @ relative, 1.e-8), 0, HORIZON)
    minimum = np.linalg.norm(delta + closest * relative)
    speed_a, speed_b = np.linalg.norm(a), np.linalg.norm(b)
    direction = np.clip(a @ b / max(speed_a * speed_b, 1.e-7), -1., 1.)
    return np.array((np.tanh(speed_b / .012), np.tanh(mover.age / 9.),
                     np.tanh(np.max(np.abs(mover.position)) / .5),
                     np.exp(-gap / .082), np.tanh(closure / .018), np.exp(-minimum / .046),
                     np.exp(-minimum / .046) * np.tanh(closure / .018), direction,
                     np.tanh(speed_a / .020), np.tanh(np.linalg.norm(relative) / .020)))


def changes(current, future):
    advance = np.subtract(future.position, current.position) / HORIZON
    extrapolation = np.asarray(current.velocity)
    error = np.linalg.norm(advance - extrapolation)
    bigger = max(np.linalg.norm(advance), np.linalg.norm(extrapolation))
    return float(error > .0035 and error > .42 * bigger)


@dataclass
class Interactions:
    tick: np.ndarray
    actor: np.ndarray
    mover: np.ndarray
    rows: np.ndarray
    targets: np.ndarray


def examples(tape, *, actor_ids=None, stride=2):
    ticks, actors, movers, design, labels = [], [], [], [], []
    for tick in range(6, len(tape) - HORIZON, stride):
        present = tape[tick]
        future = {track.id: track for track in tape[tick + HORIZON]}
        eligible = [track for track in present if track.id in future and track.age >= 4
                    and np.linalg.norm(track.velocity) > .0015]
        for mover in eligible:
            target = changes(mover, future[mover.id])
            for actor in present:
                if actor.age < 4 or mover.id == actor.id or (actor_ids is not None and actor.id not in actor_ids):
                    continue
                if np.linalg.norm(np.subtract(mover.position, actor.position)) > .24:
                    continue
                ticks.append(tick)
                actors.append(actor.id)
                movers.append(mover.id)
                design.append(describe(actor, mover))
                labels.append(target)
    return Interactions(np.asarray(ticks, dtype=int), np.asarray(actors, dtype=int), np.asarray(movers, dtype=int),
                        np.asarray(design, dtype=float).reshape(-1, 10), np.asarray(labels, dtype=float))


def fit_source(episodes):
    design = np.concatenate([episode.rows for episode in episodes])
    labels = np.concatenate([episode.targets for episode in episodes])
    return learn(design, labels, penalty=.018), learn(design[:, :UNARY], labels, penalty=.018)


def score_source(models, episodes):
    design = np.concatenate([episode.rows for episode in episodes])
    labels = np.concatenate([episode.targets for episode in episodes])
    predictions = [predict(model, design[:, :len(model)-1]) for model in models]
    return dict(rows=len(labels), changes=int(labels.sum()),
                brier=[float(np.mean((probabilities - labels) ** 2)) for probabilities in predictions])


def evidence(models, trial, *, shuffled=None):
    """Associate repeated partner changes with all pinned patches in the practice video."""
    if not len(trial.rows):
        return {}
    pair, baseline = models
    relations = predict(pair, trial.rows)
    unary = predict(baseline, trial.rows[:, :UNARY])
    targets = trial.targets.copy()
    if shuffled is not None:
        for mover in np.unique(trial.mover):
            indices = np.flatnonzero(trial.mover == mover)
            _, positions, reverse = np.unique(trial.tick[indices], return_index=True, return_inverse=True)
            targets[indices] = shuffled.permutation(targets[indices[positions]])[reverse]
    # Support positive interactions above the independent trajectory expectation.
    # Narrow pairs get more weight than distant motion in the same image.
    weight = trial.rows[:, 5] ** 2
    relation = np.maximum(relations - unary, 0.)
    benefit = weight * relation * (targets - unary)
    # Soft evidence shrinkage makes one transient track less influential.
    return {int(actor): float(benefit[trial.actor == actor].sum())
            for actor in np.unique(trial.actor)}


def choose(votes, memory):
    usable = [key for key in votes if memory.get(key, 0.) > 0.]
    if not usable:
        return None
    winner = max(usable, key=lambda key: memory[key] * np.sqrt(votes[key][1]))
    return votes[winner][0]
