"""Frozen local latent transition on multi-pattern and changing scenes."""

import copy
import json
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import make_robust_cases, scene_sequence
from gap_timing_transfer import MODEL_OUT
from run_local_hidden_transition import (add_score, empty_score, finish,
                                         train as train_transition)
from run_local_visibility_likelihood import clean_training_cases


OUT = Path('docs/experiments/2026-09-25-hidden-state-transfer-results.json')
ARMS = ('learned', 'shuffled_credit', 'frozen', 'persistence')
VISIBLE_FAMILIES = ('independent', 'crossing', 'speed_change', 'noise')


@torch.no_grad()
def evaluate(models, cases):
    visible = {family: {name: empty_score() for name in ARMS}
               for family in VISIBLE_FAMILIES if any(
                   case[0] == family for case in cases)}
    speed = {phase: {name: empty_score() for name in ARMS}
             for phase in ('at_change', 'one_after', 'two_after')}
    hidden = {str(i): {name: empty_score()
                       for name in (*ARMS[:3], 'pre_gap_hold')}
              for i in range(1, 4)}
    reference = copy.deepcopy(models['learned'].encoder)
    for family, meta, events in cases:
        codes = correlation_sequence(events)
        for model in models.values():
            model.reset_state()
        if family == 'occlusion':
            step = meta['speed']
            obj = dict(shape=meta['shape'], center=(16, 32),
                       before=(0, step), after=(0, step))
            counterfactual = scene_sequence(
                [obj], background=meta['background'])
            reference_codes = correlation_sequence(counterfactual)
            reference.reset_state()
        for t in range(14):
            predictions = {name: model.pending for name, model in
                           models.items()}
            persistence = models['learned'].observed.clone()
            for model in models.values():
                model.step(codes[t])
            if family in visible and 4 <= t <= 13:
                target = models['learned'].observed
                predictions['persistence'] = persistence
                for name, prediction in predictions.items():
                    add_score(visible[family][name], prediction, target)
                    if family == 'speed_change' and t in (8, 9, 10):
                        phase = {8: 'at_change', 9: 'one_after',
                                 10: 'two_after'}[t]
                        add_score(speed[phase][name], prediction, target)
            if family == 'occlusion':
                reference.step(reference_codes[t])
                if t == 6:
                    held = models['learned'].state.clone()
                if t in (7, 8, 9):
                    for name, model in models.items():
                        add_score(hidden[str(t-6)][name], model.state,
                                  reference.latent)
                    add_score(hidden[str(t-6)]['pre_gap_hold'], held,
                              reference.latent)
    return dict(visible={family: {name: finish(row)
                                 for name, row in arms.items()}
                         for family, arms in visible.items()},
                speed_change={phase: {name: finish(row)
                                      for name, row in arms.items()}
                              for phase, arms in speed.items()},
                occlusion_hidden={frame: {name: finish(row)
                                          for name, row in arms.items()}
                                  for frame, arms in hidden.items()})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    clean = clean_training_cases()
    trained = train_transition(
        saved['models']['learned_split'].code.motion,
        [(kind, '', events) for kind, events in clean])
    models = {name: trained[name] for name in ARMS[:3]}
    cases = make_robust_cases()
    result = dict(training_episodes=len(clean), cases=len(cases),
                  scores=evaluate(models, cases),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        visible={family: {name: round(row['f1'], 3)
                         for name, row in arms.items()}
                 for family, arms in result['scores']['visible'].items()},
        speed={phase: {name: round(row['f1'], 3)
                       for name, row in arms.items()}
               for phase, arms in result['scores']['speed_change'].items()},
        hidden={frame: {name: round(row['f1'], 3)
                        for name, row in arms.items()}
                for frame, arms in result['scores']['occlusion_hidden'].items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
