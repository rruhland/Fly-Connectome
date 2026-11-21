"""Associate anonymous visual changes with public reward times in native Pong."""

import argparse
from collections import deque
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch

from .tt3_arena import TT3Arena, heldout_cases
from .tt3_learning import play_point
from .tt3_numeric_transfer import PortableMemory, VisualPlayer
from .tt3_pong import source_episodes
from .tt3_visual import (RoleBinder, RoleModel, _RoleEvidence, _SceneHistory, _control_rows, _effects,
                         _fit_ranking, _reward_witness, _scaled)


SOURCE_SEED = 21_402_819
TABLE_SEED = 21_402_820
WINDOW = 9
FEATURES = ('speed', 'recent_travel', 'velocity_change', 'log_area', 'log_area_change',
            'recent_missing_fraction', 'time_since_seen', 'edge_proximity', 'edge_proximity_change',
            'closest_neighbor')
METHODS = ('learned_reward_time', 'shuffled_reward_time', 'supplied_witness')


def _anonymous_candidates(history, shape):
    """Represent every observed track near the selected clock time in the same way."""
    now = history[-1][0]
    paths = {}
    for tick, tracks, support in history:
        for track in tracks:
            if track.observed:
                paths.setdefault(track.id, []).append((tick, track, support[track.id][0]))
    marked = []
    for identity, path in paths.items():
        if len(path) < 3:
            continue
        final_tick, track, last_area = path[-1]
        positions = np.array([item.position for _, item, _ in path])
        velocities = np.array([item.velocity for _, item, _ in path])
        areas = [area for _, _, area in path]
        axial_extent = shape[0] / (2 * shape[1])
        bounds = np.array([.5] + [axial_extent] * (positions.shape[1] - 1))
        edge = np.max(np.abs(positions) / bounds, axis=1)
        nearest = min((np.linalg.norm(positions[-1] - other[-1][1].position)
                       for key, other in paths.items() if key != identity), default=1.)
        features = (np.linalg.norm(velocities[-1]), np.linalg.norm(positions[-1] - positions[0]),
                    np.linalg.norm(velocities[-1] - velocities[0]), np.log1p(last_area),
                    np.log(last_area / np.median(areas)), 1. - len(path) / len(history),
                    (now - final_tick) / len(history), edge[-1], edge[-1] - edge[0], nearest)
        marked.append((track, np.asarray(features, dtype=float)))
    return marked


def _relevance_contrasts(scene, history, candidates):
    recent = list(history)
    options, evidence = _effects(scene, recent[-1][1])
    actor = effect = None
    if options:
        rows = np.asarray([item.features for item in evidence])
        contrast = rows[:, 0] * rows[:, 1] * _control_rows(rows)[:, 2]
        winner = int(np.argmax(contrast))
        actor, effect = options[winner], evidence[winner].effect
    differences = {}
    for witness in candidates:
        comparison_actor = actor if actor is None or actor.id != witness.id else None
        compare = [item for item in candidates if comparison_actor is None or item.id != comparison_actor.id]
        selected = compare.index(witness)
        rows = []
        for tick, snapshot, _ in recent[-5::2]:
            current = {item.id: item for item in snapshot}
            if witness.id not in current or len(compare) < 2:
                continue
            observed_actor = current.get(comparison_actor.id) if comparison_actor is not None else None
            features = _scaled([scene.motion(current.get(item.id, item), observed_actor,
                                             effect if comparison_actor is not None else None, tick)
                                for item in compare])
            rows.extend(features[selected] - np.delete(features, selected, axis=0))
        differences[witness.id] = np.array(rows).reshape(-1, 5)
    return differences


def _schedule(samples, rng):
    true = [tick for tick, sample in enumerate(samples) if sample.reward and tick >= 16]
    available = [tick for tick in range(16, len(samples)) if all(abs(tick - real) >= 8 for real in true)]
    shuffled = rng.choice(available, min(len(true), len(available)), replace=False).tolist()
    eligible = [tick for tick in available if tick not in shuffled]
    neutral = rng.choice(eligible, min(4, len(eligible)), replace=False).tolist()
    return {tick: 'true' for tick in true} | {tick: 'shuffled' for tick in shuffled} | {
        tick: 'neutral' for tick in neutral}


def source_bags(seeds, rng, control=None):
    """Save frame-wise attribution candidates and transferable contrasts without labels."""
    bags = []
    for index, (seed, samples) in enumerate(zip(seeds, source_episodes(seeds, max_ticks=1080, batch_size=12), strict=True), 1):
        schedule = _schedule(samples, rng)
        scene = _SceneHistory()
        history = deque(maxlen=WINDOW)
        shape = samples[0].frames.shape[:2]
        for tick, sample in enumerate(samples):
            tracks = scene.update(sample.frames, sample.preceding_action)
            history.append((tick, tracks, scene.tracker.support))
            if control is not None and tick >= 16 and tick % 16 == 0:
                control.observe(scene, tracks, history, shape)
            if tick not in schedule:
                continue
            anonymous = _anonymous_candidates(history, shape)
            if len(anonymous) < 2:
                continue
            ids = [track.id for track, _ in anonymous]
            features = np.stack([row for _, row in anonymous])
            contrasts = _relevance_contrasts(scene, history, [track for track, _ in anonymous])
            # Existing comparator only. Attribution and transferable fits never read this field.
            witness = _reward_witness(history, shape) if schedule[tick] == 'true' else None
            bags.append(dict(seed=seed, tick=tick, clock=schedule[tick], ids=ids, features=features,
                             contrasts=[contrasts[key] for key in ids], witness=None if witness is None else witness.id))
        if index % 12 == 0 or index == len(seeds):
            print(f'Observed {index}/{len(seeds)} raw Pong episodes', flush=True)
    return bags


def _design(bags, center, scale):
    width = max(len(bag['ids']) for bag in bags)
    design = np.zeros((len(bags), width, len(center)))
    mask = np.zeros(design.shape[:2], dtype=bool)
    for i, bag in enumerate(bags):
        n = len(bag['ids'])
        design[i, :n] = np.clip((bag['features'] - center) / scale, -8., 8.)
        mask[i, :n] = True
    return design, mask


def _scores(design, mask, vector):
    temperature = .45
    logits = design @ vector[1:] / temperature
    logits[~mask] = -np.inf
    log_total = logsumexp(logits, axis=1)
    probability = np.exp(logits - log_total[:, None])
    pooled = vector[0] + temperature * (log_total - np.log(mask.sum(axis=1)))
    return pooled, probability


def fit_event_clock(bags, positive, rng):
    eligible = [bag for bag in bags if bag['clock'] in (positive, 'neutral')]
    labels = np.array([bag['clock'] == positive for bag in eligible], dtype=float)
    examples = np.concatenate([bag['features'] for bag in eligible])
    center = examples.mean(axis=0)
    scale = np.maximum(examples.std(axis=0), 1e-7)
    design, mask = _design(eligible, center, scale)
    weights = np.where(labels > 0, .5 / labels.sum(), .5 / (len(labels) - labels.sum()))
    penalty = .022

    def objective(vector):
        scores, _ = _scores(design, mask, vector)
        return float(weights @ (np.logaddexp(0., scores) - labels * scores)
                     + penalty * np.sum(vector[1:]**2) / 2)

    initial = np.vstack((np.zeros(design.shape[-1] + 1), rng.normal(0, .24, size=(3, design.shape[-1] + 1))))
    fits = [minimize(objective, vector, method='Powell', options={'maxiter': 120}) for vector in initial]
    best = min(fits, key=lambda result: result.fun)
    return dict(center=center, scale=scale, coefficients=best.x, training_loss=float(best.fun),
                optimizer='Powell scalar search', evaluations=sum(result.nfev for result in fits),
                positive=int(labels.sum()), negative=int(len(labels) - labels.sum()))


def _predictions(clock, bags):
    design, mask = _design(bags, clock['center'], clock['scale'])
    event, per_track = _scores(design, mask, clock['coefficients'])
    return event, np.argmax(per_track, axis=1)


def _portable_roles(bags, control, clock, event):
    events = [bag for bag in bags if bag['clock'] == event]
    selections = []
    if clock is None:
        for bag in events:
            if bag['witness'] in bag['ids']:
                selections.append((bag, bag['ids'].index(bag['witness'])))
    else:
        _, indices = _predictions(clock, events)
        selections = list(zip(events, indices, strict=True))
    contrasts = [bag['contrasts'][index] for bag, index in selections if len(bag['contrasts'][index])]
    rows = np.concatenate(contrasts)
    relevance = _fit_ranking(rows)
    base = control.model()
    return RoleModel(base.control_weights, relevance, base.control_floor, len(control.control), len(rows)), [
        dict(seed=bag['seed'], tick=bag['tick'], selected=bag['ids'][index]) for bag, index in selections]


def _audit_clock(clock, test):
    cases = [bag for bag in test if bag['clock'] != 'shuffled']
    events, selected = _predictions(clock, cases)
    labels = np.array([bag['clock'] == 'true' for bag in cases])
    pair = events[labels, None] - events[~labels][None, :]
    witness = [(bag['ids'][index] == bag['witness']) for bag, index in zip(cases, selected, strict=True)
               if bag['clock'] == 'true' and bag['witness'] in bag['ids']]
    return dict(auc=float((np.sum(pair > 0) + .5 * np.sum(pair == 0)) / pair.size),
                positive=int(labels.sum()), negative=int((~labels).sum()),
                reference_witness_agreement=sum(witness), reference_witness_cases=len(witness),
                rows=[dict(seed=bag['seed'], tick=bag['tick'], clock=bag['clock'], logit=float(logit),
                           selected=bag['ids'][index], comparator=bag['witness'])
                      for bag, logit, index in zip(cases, events, selected, strict=True)])


class DiscoveryBinder(RoleBinder):
    """Use a learned optical clock for terminal rewards in either visual space."""

    def __init__(self, model, clock, *, learn=False):
        self.clock = clock
        self.reward_attributions = []
        super().__init__(model, learn=learn)

    def _learn(self, tracks, reward=0.):
        initial_counts = len(self._bank.control), len(self._bank.relevance)
        self._bank.observe(self._scene, tracks, self._recent, self._image_shape)
        if reward:
            candidates = _anonymous_candidates(self._recent, self._image_shape)
            if len(candidates) >= 2:
                anonymous = [track for track, _ in candidates]
                bag = dict(ids=[track.id for track in anonymous], features=np.stack([row for _, row in candidates]))
                event, selected = _predictions(self.clock, [bag])
                contrast = _relevance_contrasts(self._scene, self._recent, anonymous)[bag['ids'][selected[0]]]
                self._bank.relevance.extend(contrast)
                self.reward_attributions.append(dict(tick=self._scene.tick, candidate=bag['ids'][selected[0]],
                                                     event_logit=float(event[0]), contrasts=len(contrast)))
        if initial_counts != (len(self._bank.control), len(self._bank.relevance)):
            self.model = self._bank.model(self._initial)


class DiscoveryPlayer(VisualPlayer):
    """Install the new binder while preserving the native numeric transfer controls."""

    def __init__(self, roles, memory, *, reward_clock, learn_roles=True, **kwargs):
        super().__init__(roles, memory, learn_roles=learn_roles, **kwargs)
        discovered = DiscoveryBinder(roles, reward_clock, learn=learn_roles)
        if getattr(self.binder, 'binder', None) is not None:
            self.binder.binder = discovered
        else:
            self.binder = discovered



def _table(roles, clocks, memory, matches, points):
    results = {method: [] for method in METHODS}
    for number in range(matches):
        match_seed = TABLE_SEED + number
        cases = heldout_cases(match_seed, points, variant='standard')
        for method, prior in roles.items():
            if method == 'supplied_witness':
                player_type, options = VisualPlayer, {}
            else:
                player_type, options = DiscoveryPlayer, {'reward_clock': clocks[method]}
            player = player_type(prior, PortableMemory.restore(memory), seed=match_seed,
                                 forecast=True, recall=True, learn_roles=True, **options)
            rows = []
            for case in cases:
                arena = TT3Arena(case.seed, view_seed=cases[0].view_seed, variant=case.variant,
                                 distractor=case.distractor)
                row = play_point(arena, player)
                row['case'] = asdict(case)
                row['view_seed_applied'] = cases[0].view_seed
                row['option'] = int(player.option)
                rows.append(row)
            attributions = getattr(player.binder.binder, 'reward_attributions', [])
            results[method].append(dict(seed=match_seed, points=rows, final_roles=asdict(player.binder.model),
                                        reward_attributions=attributions))
    totals = {}
    for method, courses in results.items():
        rows = [row for course in courses for row in course['points']]
        totals[method] = dict(points=len(rows), contact=sum(row['contacts'] > 0 for row in rows),
                              legal_return_points=sum(row['legal_returns'] > 0 for row in rows),
                              legal_returns=sum(row['legal_returns'] for row in rows),
                              sustained=sum(row['legal_returns'] >= 2 and row['rival_returns'] >= 1 for row in rows),
                              wins=sum(row['points_for'] for row in rows))
    return dict(totals=totals, matches=results)


def _array_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('memory', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--clock-output', type=Path)
    parser.add_argument('--train', type=int, default=60)
    parser.add_argument('--verify', type=int, default=20)
    parser.add_argument('--table-matches', type=int, default=3)
    parser.add_argument('--table-points', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = Path(__file__).resolve()
    sources = [path.with_name(name) for name in ('tt3_pong.py', 'tt3_arena.py', 'tt3_visual.py',
               'tt3_numeric_transfer.py', 'tt3_numeric.py', 'tt3_continuous.py', 'tt3_learning.py',
               'tt3_motor.py', 'tt3_flight.py', 'tt3_pixels.py', 'tt3_components.cpp', 'tt3_role_discovery.py')]
    sources.append(path.parents[2] / 'src/fly_connectome/pong.py')
    hashes = {source.name: sha256(source.read_bytes()).hexdigest() for source in sources}
    library = os.environ.get('TT3_COMPONENT_LIBRARY')
    backend = dict(name='compiled' if library else 'reference',
                   library_sha256=sha256(Path(library).read_bytes()).hexdigest() if library else None)
    rng = np.random.default_rng(SOURCE_SEED)
    seeds = (rng.choice(110_000_000, args.train + args.verify, replace=False) + 820_000_000).tolist()
    control = _RoleEvidence()
    source = source_bags(seeds[:args.train], rng, control=control)
    clocks = {
        'learned_reward_time': fit_event_clock(source, 'true', rng),
        'shuffled_reward_time': fit_event_clock(source, 'shuffled', rng),
    }
    event_names = ('true', 'shuffled', 'true')
    fitted = {name: _portable_roles(source, control, clocks.get(name), event)
              for name, event in zip(METHODS, event_names, strict=True)}
    roles = {name: value[0] for name, value in fitted.items()}
    args.checkpoint.write_text(json.dumps(asdict(roles['learned_reward_time']), indent=2) + '\n')
    if args.clock_output:
        args.clock_output.write_text(json.dumps(clocks['learned_reward_time'], default=_array_json, indent=2) + '\n')
    print('Role checkpoints ready:', {name: asdict(model) for name, model in roles.items()}, flush=True)
    test = source_bags(seeds[args.train:], rng)
    audits = {name: _audit_clock(clock, test) for name, clock in clocks.items()}
    print({name: {k: v for k, v in report.items() if k != 'rows'} for name, report in audits.items()}, flush=True)
    memory = json.loads(args.memory.read_text())
    table = _table(roles, clocks, memory, args.table_matches, args.table_points)
    if hashes != {source.name: sha256(source.read_bytes()).hexdigest() for source in sources}:
        raise ValueError('experiment sources changed during the run')
    result = dict(source_seed=SOURCE_SEED, table_seed=TABLE_SEED, train_seeds=seeds[:args.train],
                  test_seeds=seeds[args.train:], bag_feature_names=FEATURES, clocks=clocks, fitting_bags=source,
                  heldout_bags=test, action_control_examples=len(control.control), roles={name: asdict(model) for name, model in roles.items()},
                  attribution={name: value[1] for name, value in fitted.items()}, audit=audits, table=table,
                  memory=memory, segmentation_backend=backend, inputs_sha256=hashes)
    payload = (json.dumps(result, default=_array_json, allow_nan=False, separators=(',', ':')) + '\n').encode()
    args.receipt.write_bytes(gzip.compress(payload, mtime=0))
    print('Held-out table:', table['totals'], flush=True)


if __name__ == '__main__':
    main()
