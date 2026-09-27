"""One generic patch-association state audit across physical rotations."""

import json
import time
from functools import partial
from pathlib import Path

import torch

from associative_patch_state import AssociativePatchState, PatchAssociation
from event_only_observer_teacher import fit_event_only_observer
from generic_native_cadence import scene_events
from local_observation_model import LocalObservationModel
from run_local_context_memory import score, training_cases
from run_visual_context_controls import context_cases
from run_visual_history_controls import cases, two_mover_cases
from run_zero_shot_visual_transfer import corrupt_events
from sensor_budget import anchored_sequence
from visual_history_benchmark import render_context_case
from visual_history_controls import sensory_sequence


OUT = Path('docs/experiments/2026-09-26-associative-patch-state-results.json')


def rotate_case(case, turns):
    result = dict(case)
    for name in ('events', 'visible', 'visible_objects', 'hidden', 'intensity'):
        result[name] = [torch.rot90(value, turns, (-2, -1)) for value in case[name]]
    if 'hidden_by_entity' in case:
        result['hidden_by_entity'] = [[torch.rot90(value, turns, (-2, -1))
                                      for value in row] for row in case['hidden_by_entity']]
    return result


def observer_for(base, turns):
    height, width = (64, 32) if turns % 2 else (32, 64)
    observer = LocalObservationModel(height=height, width=width)
    observer.weights.copy_(base.weights)
    observer.bias.copy_(base.bias)
    return observer


@torch.no_grad()
def train_memory(base, sequence, *, shuffled=False, transform=None,
                 state_type=AssociativePatchState):
    memory = PatchAssociation()
    for turns in range(4):
        observer = observer_for(base, turns)
        state = state_type(height=observer.height, width=observer.width,
                                       memory=memory)
        for original in training_cases(shuffled=shuffled):
            if transform is not None:
                original = transform(original)
            case = rotate_case(original, turns)
            state.reset_state()
            for sensory in sequence(observer, case, hybrid=False):
                state.step(sensory, learn=True)
    return memory


@torch.no_grad()
def disappearance(state, observer, sequence, turns):
    continued, vanished, differences = [], [], []
    for original in context_cases():
        gone = render_context_case(direction=original['direction'], speed=original['speed'],
            y=original['y'], shape=original['shape'], cue_sign=original['cue_sign'],
            background=original['background'], disappear=True)
        capture = []
        for case in (rotate_case(original, turns), rotate_case(gone, turns)):
            state.reset_state()
            snapshots = {}
            for frame, sensory in enumerate(sequence(observer, case, hybrid=False)):
                field = state.step(sensory)
                if frame in (original['decision'], original['reveal']+2):
                    snapshots[frame] = field
            capture.append(snapshots)
        decision, reveal = original['decision'], original['reveal']+2
        mask = torch.rot90(original['visible_objects'][reveal], turns, (-2, -1))
        differences.append(float((capture[0][decision]-capture[1][decision]).abs().max()))
        continued.append(float(capture[0][reveal][mask].max()))
        vanished.append(float(capture[1][reveal][mask].max()))
    return dict(cases=16, maximum_pre_reveal_pair_difference=max(differences),
                continued_above_half=sum(x >= .5 for x in continued),
                vanished_above_half=sum(x >= .5 for x in vanished))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    episodes = [corrupt_events(scene_events(seed), seed=10000+seed) for seed in range(64)]
    base = fit_event_only_observer(episodes)
    results = dict(arms={}, noise_seed_base=60000)
    for sensor, sequence in (('visible_frames', partial(anchored_sequence, period=1)),
                              ('event_only', sensory_sequence)):
        arm = {}
        for credit in ('aligned', 'shuffled', 'frozen'):
            memory = (PatchAssociation() if credit == 'frozen' else
                      train_memory(base, sequence, shuffled=credit == 'shuffled'))
            row = dict(examples=len(memory.keys), orientations={})
            for turns in range(4):
                observer = observer_for(base, turns)
                state = AssociativePatchState(height=observer.height, width=observer.width,
                                               memory=memory)
                context = [rotate_case(case, turns) for case in context_cases()]
                kwargs = dict(hybrid=False, sequence=sequence, noise_seed=60000)
                result = dict(
                    clean=score(state, observer, context, **kwargs),
                    known_noise=score(state, observer, context, noise=(.1, .001), **kwargs),
                    heavy_noise=score(state, observer, context, noise=(.2, .002), **kwargs),
                    constant_motion=score(state, observer,
                        [rotate_case(case, turns) for case in cases(heldout=True)], **kwargs),
                    two_movers=score(state, observer,
                        [rotate_case(case, turns) for case in two_mover_cases()], multi=True, **kwargs),
                    disappearance=disappearance(state, observer, sequence, turns))
                row['orientations'][str(turns*90)] = result
                arm[credit] = row
                results['arms'][sensor] = arm
                results['seconds'] = round(time.perf_counter()-started, 2)
                OUT.write_text(json.dumps(results, indent=2)+'\n')
                print(sensor, credit, turns*90, json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
