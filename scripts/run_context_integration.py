"""One predeclared active-camera regression run for the approved upgrade."""
import json
import math
from pathlib import Path

import torch

from fly_connectome.vision import load_legacy_default
from fly_connectome.vision.state import ProbabilisticVisualState
from context_retention_audit import score
from run_local_motion_dynamics import KINDS, visual_episode
from run_probabilistic_checkpoint_acceptance import main as transfer
from run_probabilistic_scenes import main as recovery
from run_zero_shot_visual_transfer import corrupt_events


@torch.no_grad()
def main(*, prefix='docs/experiments/2026-09-27-context-consensus-integration',
         checkpoint=Path('checkpoints/m1a5/context-consensus-integration.pt')):
    torch.set_num_threads(1)
    model = load_legacy_default().upgrade_temporal_context()
    generator = torch.Generator().manual_seed(291027)
    # Fixed before evaluation: 192 training scenes, six generic motion families,
    # original training shapes, fresh seed. Hidden centers never enter learning.
    for index in range(192):
        angle = float(torch.rand((), generator=generator))*2*math.pi
        scale = .6+float(torch.rand((), generator=generator))
        phase = int(torch.randint(8, (), generator=generator))
        case, _ = visual_episode(KINDS[index % 6], phase,
            'dot' if (index//6) % 2 else 'square', angle=angle, scale=scale,
            background=bool((index//6) % 2))
        events = corrupt_events(case['events'], seed=292000+index, dropout=.15, false_rate=.0005)
        model.reset_state()
        for t, (event, binary) in enumerate(zip(events, case['visible'])):
            image = (.25+.05*math.sin(t/9)+.35*binary.float()+
                     .02*torch.randn(binary.shape, generator=generator)).clamp(0, 1)
            model.step(event, image, learn=True)
        if index % 32 == 31:
            print(f'Trained {index+1}/192 scenes', flush=True)
    model.save(checkpoint)
    model = ProbabilisticVisualState.load(checkpoint)
    counts = {h: dict(short=len(m.short.keys), long=len(m.long.keys)) for h, m in model.dynamics.items()}
    assert all(m['long'] >= 32 for m in counts.values())
    transfer(model=model, out=Path(prefix+'-transfer.json'))
    transfer_path = Path(prefix+'-transfer.json')
    replay = json.loads(transfer_path.read_text())
    replay['production_promoted'] = False
    replay['stage'] = 'active integration candidate; legacy replay equality is not a gate'
    transfer_path.write_text(json.dumps(replay, indent=2)+'\n')
    recovery(model=model, out=Path(prefix+'-recovery.json'))
    context = score(model, model.state.memory)
    # Noisy blank input must not acquire confident phantom entities.
    model.reset_state()
    extras = 0
    for _ in range(600):
        frame = (.25+.02*torch.randn(64, 64, generator=generator)).clamp(0, 1)
        events = (torch.rand(2, 64, 64, generator=generator) < .0005).float()
        state = model.step(events, frame)
        extras += sum(e['association_strength'] >= .5 for e in state['entities'])
    results = json.loads(Path(prefix+'-transfer.json').read_text())
    old = json.loads(Path('docs/experiments/2026-09-26-spatial-interval-calibration-results.json').read_text())
    fixed = ('persistence', 'velocity', 'mean_velocity', 'lag_two', 'acceleration')
    gates = dict(
        aggregate=all(results['means'][h]['nll'] <= min(old['means'][h][n]['nll']
            for n in (*fixed, 'shuffled'))-.2 for h in ('4', '8')),
        family=all(row[h]['nll'] <= min(old['families'][k][h][n]['nll'] for n in fixed)+.2
            for k, row in results['families'].items() for h in ('4', '8')),
        calibration=all(.85 <= results['means'][h]['calibrated_marginal_90_coverage'] <= .95 for h in ('4', '8')),
        observation_coverage=1-results['missing']/results['samples'] >= .95,
        frozen=results['frozen_parameters'], context=all(r['hits'] == r['total'] for r in context.values()),
        blank=extras == 0)
    scenes = json.loads(Path(prefix+'-recovery.json').read_text())
    gates['recovery'] = all(scenes[c]['state_gate'] for c in ('image_noise', 'outage'))
    gates['cached_forecasts'] = (scenes['outage']['missing_cached_forecasts'] == 0
                                and scenes['outage']['cached_forecasts'] > 0)
    a = scenes['ambiguous_prefix']
    gates['ambiguity'] = (a['max_mixture_difference'] == 0 and a['max_probability_difference'] == 0
                         and a['continued_reidentified'] == 8 and a['vanished_reidentified'] == 0
                         and a['vanished_confident_support'] == 0)
    summary = dict(training_scenes=192, training_seed=291027, banks=counts,
                   context=context, blank_confident_entities=extras, gates=gates)
    Path(prefix+'-results.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    if not all(gates.values()):
        raise SystemExit('Integration gates failed; keep the production default unchanged.')


if __name__ == '__main__':
    main()
