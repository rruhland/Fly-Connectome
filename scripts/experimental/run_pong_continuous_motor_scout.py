"""Compare within-rally gain inference to short calibration on matched starts."""

import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path
import signal
import time
from unittest.mock import patch

import torch

from fly_connectome.control.session import model_bytes, source_signature
from pong_continuous_motor_observer import ContinuousMotorController, ContinuousMotorEffects, VisualPlant
from pong_visual_button_effects import VisualButtonEffects
from run_pong_fast_mid_rally_scout import EarlyVisualIntentController
from run_pong_inverse_cycle import save, sha
import run_pong_mid_rally_remap as mid
from run_pong_surprise_recovery import reached_deadline


class RecordedContinuousController(ContinuousMotorController):
    def choose(self, entities, baseline_action):
        action, decision = super().choose(entities, baseline_action)
        self.learner.trace.append(dict(elapsed_seconds=self.elapsed,
            intended=decision['intended'], requested=action, gains=self.learner.effects.copy(),
            gain_variances=self.learner.variances, supported_order=self.order_confirmed))
        return action, decision


class RecordedShorterController(EarlyVisualIntentController):
    def choose(self, entities, baseline_action):
        action, decision = super().choose(entities, baseline_action)
        self.learner.trace.append(dict(requested=action))
        return action, decision


def play(condition, arm, plant, checkpoint):
    seed, forces, switch = condition['seed'], condition['forces'], condition['switch_tick']
    sorted_forces = sorted(forces)
    request_order = tuple(sorted_forces.index(force) for force in forces)
    native_camera = mid.PongCamera

    class ChangedCamera(native_camera):
        def step(self, command):
            if self.total_ticks >= mid.SWITCH_TICK:
                self.actions = sorted_forces
            return super().step(command)

    if arm == 'shorter':
        learner = VisualButtonEffects.from_state(checkpoint)
        controller = RecordedShorterController
    else:
        learner = ContinuousMotorEffects(plant, updates_enabled=arm == 'live')
        controller = RecordedContinuousController
    initial_probes = learner.dither_count
    learner.trace = []
    initial_state = learner.state()
    with (patch.object(mid, 'PongCamera', ChangedCamera),
          patch.object(mid, 'CYCLED', request_order),
          patch.object(mid, 'SWITCH_TICK', switch),
          patch.object(mid, 'VisualIntentController', controller)):
        row = mid.play(seed, learner)
    final_state = learner.state()
    # The shorter arm serializes instance fields, including transient harness telemetry.
    final_state.pop('trace', None)
    initial_state.pop('trace', None)
    if arm == 'frozen':
        assert final_state == initial_state
    first = row['post_change_events'][0] if row['post_change_events'] else None
    through_first = math.ceil(first['tick'] * 5 / 12) if first else len(learner.trace)
    return dict(row, condition=condition, arm=arm,
                probe_frames=learner.dither_count - initial_probes,
                final_state=final_state,
                original_requests=[item['requested'] for item in learner.trace[:row['prefix_requests']]],
                changed_trace_through_first_opportunity=learner.trace[row['prefix_requests']:through_first])


def run(directory, schedule_path, fit_path, checkpoint_path, arms, wall_budget):
    if len(arms) != len(set(arms)) or wall_budget <= 0:
        raise ValueError('unique arms and a positive wall budget required')
    schedule = json.loads(schedule_path.read_text())
    seeds = [condition['seed'] for condition in schedule]
    labels = sorted({condition['layout'] for condition in schedule})
    for condition in schedule:
        forces, switch = condition['forces'], condition['switch_tick']
        if len(forces) != 3 or len(set(forces)) != 3 or not all(-1 <= force <= 1 for force in forces):
            raise ValueError('three distinct actuator values in [-1, 1] required')
        if type(switch) is not int or switch % 12 or not 0 < switch < mid.END_TICK:
            raise ValueError('switch must fall on a supported camera boundary inside the episode')
    plant = VisualPlant(**json.loads(fit_path.read_text())['plant'])
    checkpoint = json.loads(checkpoint_path.read_text())
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('a nonempty schedule with unique starts is required')
    torch.set_num_threads(1)
    model_bytes()
    signal.signal(signal.SIGALRM, reached_deadline)
    directory.mkdir(parents=True, exist_ok=False)
    files = [schedule_path, fit_path, checkpoint_path, Path(__file__),
             Path(__file__).with_name('pong_continuous_motor_observer.py'),
             Path(__file__).with_name('run_pong_fast_mid_rally_scout.py'),
             *(Path(__file__).with_name(name) for name in mid.FILES)]
    save(directory / 'conditions.json', schedule)
    save(directory / 'continuous-normal.json', ContinuousMotorEffects(plant).state())
    save(directory / 'shorter-normal.json', checkpoint)
    files.extend(directory / name for name in ('conditions.json', 'continuous-normal.json', 'shorter-normal.json'))
    spec = dict(source=source_signature(), files={str(path): sha(path) for path in files},
                arms=arms, schedule=schedule, layouts=labels,
                max_games=len(seeds) * len(arms), wall_budget=wall_budget, workers=1,
                scope='paired development with independent normal model at each start')
    save(directory / 'spec.json', spec)
    digest = sha(directory / 'spec.json')
    began = time.monotonic()
    rows = {arm: [] for arm in arms}
    active = directory / 'active.json'
    try:
        for index, condition in enumerate(schedule):
            for arm in (arms if index % 2 == 0 else reversed(arms)):
                remaining = wall_budget - (time.monotonic() - began)
                if remaining <= 0:
                    raise TimeoutError('wall budget reached before a game')
                save(active, dict(index=index, condition=condition, arm=arm, max_transitions=1000))
                signal.setitimer(signal.ITIMER_REAL, remaining)
                try:
                    row = play(condition, arm, plant, checkpoint)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                save(directory / f'{index:03}-{arm}.json', row)
                rows[arm].append(row)
                print(index, condition['layout'], arm, [event['hit'] for event in row['post_change_events']],
                      row['first_contradiction_seconds'], flush=True)
                active.unlink()
    except TimeoutError:
        pass
    games = [game for arm in arms for game in rows[arm]]
    by_arm = {arm: {game['condition']['seed']: game for game in rows[arm]} for arm in arms}
    matched = [tuple(by_arm[arm][seed] for arm in arms) for seed in seeds
               if all(seed in by_arm[arm] for arm in arms)]
    prefix_parity = sum(len({game['prefix_sha256'] for game in pair}) == 1 for pair in matched)
    normal_action_differences = sum(len(set(actions)) != 1 for pair in matched
        for actions in zip_longest(*(game['original_requests'] for game in pair)))
    first_clocks = [tuple(next((event['tick'] for event in game['post_change_events']), None)
                          for game in pair) for pair in matched]
    model_bytes()
    summary = dict(complete=len(games) == spec['max_games'], paired_starts=len(matched),
                   prefix_parity=prefix_parity, normal_action_differences=normal_action_differences,
                   initials=len(games),
                   matched_first_clocks=sum(all(clock is not None for clock in clocks)
                                            and len(set(clocks)) == 1 for clocks in first_clocks),
                   transitions=sum(game['transitions'] for game in games),
                   pending_camera_grant=1000 if active.exists() else 0,
                   spec_sha256=digest, wall_seconds=time.monotonic() - began,
                   unchanged_sources=spec['source'] == source_signature()
                       and sha(directory / 'spec.json') == digest
                       and all(sha(Path(path)) == expected for path, expected in spec['files'].items()),
                   arms={arm: mid.metrics(arm_games) for arm, arm_games in rows.items()},
                   by_layout={label: {arm: mid.metrics([game for game in arm_games
                       if game['condition']['layout'] == label]) for arm, arm_games in rows.items()}
                       for label in labels})
    save(directory / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('schedule', type=Path)
    parser.add_argument('fit', type=Path)
    parser.add_argument('legacy_checkpoint', type=Path)
    parser.add_argument('--arms', nargs='+', choices=('live', 'shorter', 'frozen'), default=['live', 'shorter'])
    parser.add_argument('--wall-budget', type=float, default=180)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.schedule, args.fit, args.legacy_checkpoint,
                         args.arms, args.wall_budget), sort_keys=True))
