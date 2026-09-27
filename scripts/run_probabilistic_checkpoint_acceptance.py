"""End-to-end checkpoint replay on the frozen spatial-belief transfer split."""

import json
import hashlib
import math
import time
from pathlib import Path

import torch

from probabilistic_visual_state import ProbabilisticVisualState
from run_local_motion_dynamics import KINDS, visual_episode
from run_zero_shot_visual_transfer import corrupt_events
from spatial_belief import mixture_log_prob
from visual_history_benchmark import SHAPES


def parameter_fingerprint(model):
    digest = hashlib.sha256()
    tensors = [model.observer.weights, model.observer.bias,
               model.state.memory.keys, model.state.memory.values, model.state.memory.metric]
    for h, m in model.dynamics.items():
        tensors.extend((m.keys, m.values, m.generator.get_state()))
        tensors.extend(m.calibration.ranks)
        digest.update(str((h, m.capacity, m.seen, list(model.calibration[h].outcomes),
                           list(model.calibration[h].residuals))).encode())
    for tensor in tensors:
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def main(*, model=None, out=Path('docs/experiments/2026-09-26-probabilistic-checkpoint-results.json')):
    torch.set_num_threads(1)
    if model is None:
        model = ProbabilisticVisualState.load('checkpoints/m1a5/probabilistic-visual-candidate.pt')
    before = parameter_fingerprint(model)
    SHAPES.update(L=((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
                  T=((-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)),
                  line=tuple((0, x) for x in range(-2, 3)))
    rows = {k: {h: [] for h in (1, 4, 8)} for k in KINDS}
    scenes = samples = missing = frames = 0
    elapsed = 0.
    for kind in KINDS:
        for shape in ('L', 'T', 'line'):
            for phase in (41, 43):
                for angle in (2*math.pi/15, 11*math.pi/30, 4*math.pi/5):
                    for scale in (.85, 1.35):
                        case, truth = visual_episode(kind, phase, shape, angle=angle, scale=scale,
                                                     background=bool(scenes % 2))
                        events = corrupt_events(case['events'], seed=205000+scenes, dropout=.15, false_rate=.0005)
                        generator = torch.Generator().manual_seed(206000+scenes)
                        model.reset_state()
                        for t, (event, binary) in enumerate(zip(events, case['visible'])):
                            image = (.25+.05*math.sin(t/9)+.35*binary.float()+
                                     .02*torch.randn(binary.shape, generator=generator)).clamp(0., 1.)
                            start = time.perf_counter()
                            state = model.step(event, image)
                            elapsed += time.perf_counter()-start
                            frames += 1
                            visible = [e for e in state['entities'] if e['observed']]
                            identity = min(e['id'] for e in visible) if visible else None
                            if t < 6:
                                continue
                            for h in rows[kind]:
                                if t+h >= len(truth):
                                    continue
                                samples += 1
                                forecasts = [f for f in state['forecasts'] if f['id'] == identity and
                                             f['horizon_samples'] == h and f['evidence_age'] == 0]
                                if not forecasts:
                                    missing += 1
                                    continue
                                forecast = forecasts[0]
                                lo, hi = forecast['marginal_interval_90']
                                target = truth[t+h]
                                rows[kind][h].append([
                                    float(-mixture_log_prob(forecast['mixture_centers'], forecast['mixture_weights'], target)),
                                    float((forecast['position']-target).norm()),
                                    float(((target >= lo) & (target <= hi)).float().mean())])
                        scenes += 1
    mean = lambda values: dict(zip(('nll', 'mean_point_error', 'calibrated_marginal_90_coverage'),
                                    torch.tensor(values).mean(0).tolist()))
    totals = {h: mean([v for k in KINDS for v in rows[k][h]]) for h in (1, 4, 8)}
    reference = json.loads(Path('docs/experiments/2026-09-26-spatial-interval-calibration-results.json').read_text())
    difference = max(abs(totals[h][n]-reference['means'][str(h)]['learned'][n])
                     for h in totals for n in totals[h])
    result = dict(scenes=scenes, samples=samples, missing=missing, means=totals,
        families={k: {h: mean(v) for h, v in row.items()} for k, row in rows.items()},
        reference_max_difference=difference, replay_gate=difference < 1e-5,
        frozen_parameters=before == parameter_fingerprint(model), parameter_sha256=before,
        frames=frames, seconds=elapsed, fps=frames/elapsed,
        runtime_module=type(model).__module__,
        production_promoted=type(model).__module__.startswith('fly_connectome.vision'))
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
