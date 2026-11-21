"""Learn point value from the first outgoing section of archived RGB ball tracks."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from .tt4_arena import NET_HEIGHT
from .tt4_transfer_trial import ROOT, _hash, _read
from .tt18_plant import PLANTS, PlantSettings


TT18 = ROOT / 'artifacts/tt18'
CONTROL_DT = 1. / 30.
FEATURE_NAMES = ('launch_depth', 'launch_height', 'forward_speed', 'rise_per_depth',
                 'lateral_at_net', 'vertical_at_net_linear', 'early_forward_progress')
SHUFFLE_SEED = 363646633


def outgoing(sightings, role, width, half_length):
    """Find a near-side reversal from RGB bearings in table-relative metres."""
    if len(sightings) < 6:
        return None
    ticks, coordinates = zip(*sightings, strict=True)
    ticks = np.asarray(ticks)
    points = np.asarray(coordinates) / np.array((width, half_length, NET_HEIGHT))
    interval = np.diff(ticks)
    velocity = np.diff(points, axis=0) / (CONTROL_DT * interval[:, None])
    for i in np.flatnonzero((points[:-1, 1] < -.39) & (points[:-1, 1] > -1.85)
                            & (velocity[:, 1] > .58) & (interval <= 3)):
        frames = np.flatnonzero((ticks[:-1] >= ticks[i] - 7) & (ticks[:-1] < ticks[i]) & (interval <= 3))
        if len(frames) == 0 or i + 1 >= len(velocity) or interval[i + 1] > 3 or velocity[i + 1, 1] < .25:
            continue
        preceding = velocity[frames]
        if role == -1:
            plausible = np.any(preceding[:, 1] < -.65)
        else:
            plausible = np.any((abs(preceding[:, 1]) < .50) & (preceding[:, 2] < -1.1))
        if not plausible:
            continue
        later = np.flatnonzero((np.arange(len(points)) > i + 1) & (ticks <= ticks[i] + 12))
        end = i + 2 if len(later) == 0 else int(later[-1])
        returning = [k for k in range(i + 1, end) if interval[k] <= 3 and velocity[k, 1] < -.20]
        end = min(returning, default=end)
        origin = points[i]
        speed = np.mean(velocity[i:i + 2], axis=0)
        slope = speed[[0, 2]] / speed[1]
        predicted_net = origin[[0, 2]] - origin[1] * slope
        progress = float(np.clip(np.max(points[i + 1:end + 1, 1]) - origin[1], 0., 2.5))
        values = np.array((origin[1], origin[2], speed[1], slope[1], *predicted_net, progress))
        return np.clip(values, (-1.85, -.25, .3, -6., -3., -8., 0.),
                       (-.39, 5., 9., 8., 3., 12., 2.5))
    return None


@dataclass(frozen=True)
class RewardModel:
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    rate: float

    @classmethod
    def fit(cls, features, rewards):
        values = np.asarray(features)
        truth = np.asarray(rewards, dtype=float) > 0.
        center = np.median(values, axis=0)
        scale = np.maximum(np.std(values, axis=0), (.08, .18, .35, .35, .15, .5, .10))
        design = np.column_stack((np.ones(len(values)), np.clip((values - center) / scale, -3., 3.)))
        rate = (truth.sum() + .5) / (len(truth) + 1.)
        prior = np.r_[np.log(rate / (1. - rate)), np.zeros(values.shape[1])]
        penalty = np.r_[.25, np.full(values.shape[1], 5.)]

        def loss(weights):
            logit = design @ weights
            delta = weights - prior
            value = np.logaddexp(0., logit).sum() - truth @ logit + .5 * delta @ (penalty * delta)
            gradient = design.T @ (expit(logit) - truth) + penalty * delta
            return value, gradient

        result = minimize(loss, prior, jac=True, method='L-BFGS-B')
        if not result.success:
            raise ValueError('the source reward predictor did not converge')
        return cls(center, scale, result.x, float(rate))

    def predict(self, features):
        scaled = np.clip((features - self.center) / self.scale, -3., 3.)
        return float(expit(self.weights[0] + scaled @ self.weights[1:]))

    def state(self):
        return dict(center=self.center.tolist(), scale=self.scale.tolist(), weights=self.weights.tolist(), rate=self.rate)


def source_rows():
    combined = sorted(path for path in TT18.glob('combined-*.json.gz')
                      if path.name.split('-')[1] in ('349051097', '418336241') and '-player' not in path.name)
    online = sorted(path for path in TT18.glob('online-gantry-compact-*.json.gz') if '-player' not in path.name)
    nominal, compact = PlantSettings(), PLANTS['compact']
    rows, receipts = [], []
    for path in (*combined, *online):
        saved = _read(path)
        if 'games' in saved:
            grouped = [(str(game['seed']), game['points'], True) for game in saved['games']]
        else:
            grouped = [(str(saved['seed']), saved['points'], False)]
        receipts.append(dict(path=path.name, sha256=_hash(path), points=sum(len(points) for _, points, _ in grouped)))
        for group, points, changed in grouped:
            for row in points:
                table = compact if changed or row.get('changed') else nominal
                seen = [(tick, sample[0]) for tick, sample in enumerate(row['optical']) if sample[0] is not None]
                features = outgoing(seen, row['server'], table.table_half_width, table.table_half_length)
                rows.append(dict(group=group, archive=path.name, seed=row['seed'], role=row['server'],
                                 reward=row['public_reward'], observed_frames=len(seen),
                                 feature=None if features is None else features.tolist()))
    return rows, receipts


def _auc(labels, probabilities):
    truth = np.asarray(labels) > 0.
    positive, negative = int(truth.sum()), int((~truth).sum())
    if not positive or not negative:
        return None
    return float((rankdata(probabilities)[truth].sum() - positive * (positive + 1.) / 2.) / (positive * negative))


def train_source():
    all_rows, receipts = source_rows()
    rows = [row for row in all_rows if row['feature'] is not None]
    random = np.random.default_rng(SHUFFLE_SEED)
    for group in sorted({row['group'] for row in rows}):
        for role in (-1, 1):
            grouped = [row for row in rows if (row['group'], row['role']) == (group, role)]
            for row, value in zip(grouped, random.permutation([row['reward'] for row in grouped]), strict=True):
                row['shuffled_reward'] = float(value)
    models, audit = {}, {}
    for mode, reward_key in (('gantry_reward', 'reward'), ('shuffled_reward', 'shuffled_reward')):
        models[mode] = {}
        audit[mode] = {}
        for role in (-1, 1):
            selected = [row for row in rows if row['role'] == role]
            if len(selected) < 12:
                raise ValueError(f'insufficient outgoing gantry traces for role {role}')
            model = RewardModel.fit([row['feature'] for row in selected], [row[reward_key] for row in selected])
            models[mode][role] = model
            held_predictions, held_rewards = [], []
            for group in sorted({row['group'] for row in selected}):
                train = [row for row in selected if row['group'] != group]
                held = [row for row in selected if row['group'] == group]
                predictor = RewardModel.fit([row['feature'] for row in train], [row[reward_key] for row in train])
                held_predictions.extend(predictor.predict(row['feature']) for row in held)
                held_rewards.extend(row['reward'] for row in held)
            audit[mode][str(role)] = dict(qualified=len(selected), wins=sum(row['reward'] > 0 for row in selected),
                                         held_gantry_auc=_auc(held_rewards, held_predictions), state=model.state())
    manifest = dict(receipts=receipts, feature_names=FEATURE_NAMES, optical_dt=CONTROL_DT, shuffle_seed=SHUFFLE_SEED,
                    points=len(all_rows), qualified=len(rows), observed_frames=sum(row['observed_frames'] for row in all_rows),
                    models=audit, rows=all_rows)
    return models, manifest
