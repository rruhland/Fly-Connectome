"""One pre-registered conditional spatial-distribution comparison."""

import copy
import json
import math
import time
from collections import deque
from pathlib import Path

import torch

from run_local_motion_dynamics import KINDS, visual_episode, fixed_displacement
from run_zero_shot_visual_transfer import corrupt_events
from spatial_belief import SpatialBelief, MarginalCalibration, mixture_log_prob, mixture_pit
from statistical_visual_state import StatisticalVisualCandidate
from streaming_visual_state import StreamingVisualState
from visual_history_benchmark import SHAPES


FIXED = ('persistence', 'velocity', 'mean_velocity', 'lag_two', 'acceleration')
HORIZONS = (1, 4, 8)


def observed_stream(model, case, seed):
    events = corrupt_events(case['events'], seed=seed, dropout=.15, false_rate=.0005)
    noise = torch.Generator().manual_seed(seed+1000)
    model.reset_state()
    positions = []
    for t, (event, binary) in enumerate(zip(events, case['visible'])):
        image = (.25+.05*math.sin(t/9)+.35*binary.float()+
                 .02*torch.randn(binary.shape, generator=noise)).clamp(0., 1.)
        state = model.step(event, image)
        visible = [e for e in state['entities'] if e['observed']]
        entity = min(visible, key=lambda e: e['id']) if visible else None
        positions.append(None if entity is None else (entity['id'], entity['position']))
    return positions


def past(positions, t):
    rows = positions[t-4:t+1]
    if len(rows) != 5 or any(p is None for p in rows) or len({p[0] for p in rows}) != 1:
        return None
    locations = torch.stack([p[1] for p in rows])
    return rows[-1][0], locations[-1], locations[1:]-locations[:-1]


@torch.no_grad()
def collect(*, training_streams=192, train_seed=141001, event_seed=142000,
            test_seed=195000, phases=(31, 37), angles=(math.pi/15, 4*math.pi/15, 11*math.pi/15),
            scales=(.8, 1.3), cache='checkpoints/m1a5/spatial-belief-data.pt'):
    source = StreamingVisualState.load('checkpoints/m1a5/streaming-candidate.pt')
    state = StatisticalVisualCandidate(height=64, width=64, observer=source.observer,
                                       memory=copy.deepcopy(source.state.memory))
    training, testing = {h: [] for h in HORIZONS}, []
    generator = torch.Generator().manual_seed(train_seed)
    started = time.perf_counter()
    for trial in range(math.ceil(training_streams/len(KINDS))):
        for index, kind in enumerate(KINDS):
            if trial*len(KINDS)+index >= training_streams:
                break
            angle = float(torch.rand((), generator=generator))*2*math.pi
            scale = .6+float(torch.rand((), generator=generator))
            phase = int(torch.randint(8, (), generator=generator))
            case, _ = visual_episode(kind, phase, 'dot' if trial % 2 else 'square',
                                      angle=angle, scale=scale, background=bool(trial % 2))
            positions = observed_stream(state, case, event_seed+trial*len(KINDS)+index)
            # Endpoint-time ordering: only a visible same-ID endpoint supplies credit.
            for end in range(len(positions)):
                if positions[end] is None:
                    continue
                for h in HORIZONS:
                    t = end-h
                    context = past(positions, t) if t >= 4 else None
                    if context is not None and context[0] == positions[end][0]:
                        training[h].append((context[2], positions[end][1]-context[1]))
    training_seconds = time.perf_counter()-started
    print('collected training', training_seconds, flush=True)
    SHAPES.update(L=((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
                  T=((-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)),
                  line=tuple((0, x) for x in range(-2, 3)))
    scenes = missing = samples = 0
    for kind in KINDS:
        for shape in ('L', 'T', 'line'):
            for phase in phases:
                for angle in angles:
                    for scale in scales:
                        case, truth = visual_episode(kind, phase, shape, angle=angle, scale=scale,
                                                     background=bool(scenes % 2))
                        positions = observed_stream(state, case, test_seed+scenes)
                        for t in range(6, len(positions)):
                            context = past(positions, t)
                            for h in HORIZONS:
                                if t+h >= len(positions):
                                    continue
                                samples += 1
                                if context is None:
                                    missing += 1
                                else:
                                    testing.append((kind, h, context[2], truth[t+h]-context[1]))
                        scenes += 1
    data = dict(training=training, testing=testing, scenes=scenes, samples=samples,
                missing=missing, training_observation_seconds=training_seconds,
                collection_seconds=time.perf_counter()-started)
    torch.save(data, cache)
    return data


@torch.no_grad()
def evaluate(data, *, calibration=None, out='docs/experiments/2026-09-26-spatial-belief-results.json'):
    models, controls, shuffled = {}, {}, {}
    started = time.perf_counter()
    for h in HORIZONS:
        model = SpatialBelief()
        baseline = {name: SpatialBelief() for name in FIXED}
        for history, target in data['training'][h]:
            model.observe(history, target)
            for name, control in baseline.items():
                control.observe(history, target-fixed_displacement(history, h, name))
        models[h], controls[h] = model, baseline
        shuffled[h] = copy.deepcopy(model)
        permutation = torch.randperm(len(model.values), generator=torch.Generator().manual_seed(196000+h))
        shuffled[h].values = model.values[permutation]
    training_seconds = time.perf_counter()-started
    def distributions_for(h, history):
        _, basis, scale = SpatialBelief.encode(history)
        model = models[h]
        distributions = dict(learned=model.distribution(history), shuffled=shuffled[h].distribution(history))
        weights = torch.ones(len(model.values))/len(model.values)
        distributions['unconditional'] = model.values @ basis*scale, weights
        for name in FIXED:
            centers = controls[h][name].values @ basis*scale+fixed_displacement(history, h, name)
            distributions[name] = centers, weights
        return distributions

    intervals = {h: {n: MarginalCalibration() for n in ('learned', 'shuffled', 'unconditional', *FIXED)}
                 for h in HORIZONS}
    if calibration is not None:
        for h, rows in calibration.items():
            for history, target in rows:
                for name, (centers, mass) in distributions_for(h, history).items():
                    intervals[h][name].observe(mixture_pit(centers, mass, target))
    bounds = {h: {n: c.bounds() for n, c in row.items()} for h, row in intervals.items()}
    scores = {k: {h: {n: [] for n in ('learned', 'shuffled', 'unconditional', *FIXED)}
                   for h in HORIZONS} for k in KINDS}
    for kind, h, history, target in data['testing']:
        distributions = distributions_for(h, history)
        for name, (centers, mass) in distributions.items():
            pit = mixture_pit(centers, mass, target)
            scores[kind][h][name].append([
                float(-mixture_log_prob(centers, mass, target)),
                float(((centers*mass[:, None]).sum(0)-target).norm()),
                float(((pit >= .05) & (pit <= .95)).float().mean()),
                float(((pit >= bounds[h][name][0]) & (pit <= bounds[h][name][1])).float().mean())])
    means = lambda rows: dict(zip(('nll', 'mean_point_error', 'marginal_90_coverage', 'calibrated_marginal_90_coverage'),
                                   torch.tensor(rows).mean(0).tolist()))
    families = {k: {h: {n: means(v) for n, v in row.items()} for h, row in family.items()}
                for k, family in scores.items()}
    totals = {h: {n: means([v for k in KINDS for v in scores[k][h][n]])
                   for n in scores[KINDS[0]][h]} for h in HORIZONS}
    aggregate = all(totals[h]['learned']['nll'] <= min(totals[h][n]['nll'] for n in (*FIXED, 'shuffled'))-.2
                    for h in (4, 8))
    family = all(row[h]['learned']['nll'] <= min(row[h][n]['nll'] for n in FIXED)+.2
                 for row in families.values() for h in (4, 8))
    coverage_key = 'marginal_90_coverage' if calibration is None else 'calibrated_marginal_90_coverage'
    calibrated = all(.85 <= totals[h]['learned'][coverage_key] <= .95 for h in (4, 8))
    result = {k: v for k, v in data.items() if k not in ('training', 'testing')}
    result.update(means=totals, families=families, aggregate_gate=aggregate, family_gate=family,
                  calibration_gate=calibrated, coverage_gate=1-data['missing']/data['samples'] >= .95,
                  association_training_seconds=training_seconds, evaluation_seconds=time.perf_counter()-started-training_seconds)
    Path(out).write_text(json.dumps(result, indent=2)+'\n')
    torch.save({h: dict(keys=m.keys, values=m.values, seen=m.seen) for h, m in models.items()},
               'checkpoints/m1a5/spatial-belief-experiment.pt')
    if calibration is not None:
        torch.save({h: {n: list(c.ranks) for n, c in row.items()} for h, row in intervals.items()},
                   'checkpoints/m1a5/spatial-belief-calibration.pt')
    print(json.dumps(dict(means=totals, gates=[aggregate, family, calibrated, result['coverage_gate']])), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    evaluate(collect())
