"""Measure RGB-only cold and practiced arm transitions with the game absent."""

import argparse
import json
from pathlib import Path
from statistics import median

from planar_position_arm import LINK_LENGTHS, MAX_JOINT_SPEED, MAX_RELATIVE_SETPOINT, SERVO_RESPONSE
from pong_button_panel import (
    FAST_GEAR,
    PONG_TICK_SECONDS,
    PRESS_RADIUS,
    PRESS_TICKS,
    RELEASE_RADIUS,
    REACHING_TICK_SECONDS,
    SOURCE_GEAR,
    ButtonPanel,
    VisualButtonMotor,
    pretrain_motor,
)


PRACTICE_TICKS = 80
MAX_TRANSITION_TICKS = 60
HOLD_CHECK_TICKS = 3
PHASES = ('first_press', 'opposite_switch', 'release_after_switch', 'press_from_neutral',
          'release_after_press', 'opposite_press_from_neutral')


def measure(scene_seed, motor_seed, gear, memory):
    panel = ButtonPanel(scene_seed, gear=gear)
    motor = VisualButtonMotor(motor_seed, memory=memory)
    first = scene_seed % 2
    timings = []
    unstable_ticks = 0
    for target in (first, 1 - first, None, first, None, 1 - first):
        for ticks in range(1, MAX_TRANSITION_TICKS + 1):
            panel.step(motor.act(panel.frame(), target))
            if panel.active_button == target:
                timings.append(ticks)
                for _ in range(HOLD_CHECK_TICKS):
                    panel.step(motor.act(panel.frame(), target))
                    unstable_ticks += int(panel.active_button != target)
                break
        else:
            timings.append(None)
    return {'transition_ticks': timings, 'unstable_ticks': unstable_ticks, 'physics_ticks': panel.steps,
            'calibration_ticks': len(motor.probe_ticks)}


def summarize(cases, condition):
    observations = [case[condition] for case in cases]
    phases = {}
    for phase, observed_ticks in zip(PHASES, zip(*(item['transition_ticks'] for item in observations))):
        completed = [ticks for ticks in observed_ticks if ticks is not None]
        phases[phase] = {'successes': len(completed), 'attempts': len(observations),
                         'median_ticks': median(completed) if completed else None,
                         'max_ticks': max(completed) if completed else None}
    return {'phases': phases, 'unstable_ticks': sum(item['unstable_ticks'] for item in observations)}


def run(cases):
    if cases < 1:
        raise ValueError('at least one panel case is required')
    configurations = {}
    for name, gear in (('source', SOURCE_GEAR), ('fast', FAST_GEAR)):
        results = []
        for seed in range(cases):
            practice_seed, motor_seed = 8500 + seed, 100 + seed
            memory, practice = pretrain_motor(practice_seed, ticks=PRACTICE_TICKS, gear=gear)
            results.append({'panel_seed': seed, 'practice_seed': practice_seed, 'motor_seed': motor_seed,
                            'practice': practice, 'practice_memory': memory,
                            'cold': measure(seed, motor_seed, gear, None),
                            'arm_practiced': measure(seed, motor_seed, gear, memory)})
        configurations[name] = {'gear': gear, 'cases': results,
                                'speed_limits_rad_s': [gear * speed for speed in MAX_JOINT_SPEED],
                                'cold': summarize(results, 'cold'),
                                'arm_practiced': summarize(results, 'arm_practiced')}
    return {'schema': 1, 'tick_seconds': PONG_TICK_SECONDS, 'practice_ticks': PRACTICE_TICKS,
            'transition_phases': PHASES, 'transition_tick_cap': MAX_TRANSITION_TICKS,
            'hold_check_ticks': HOLD_CHECK_TICKS, 'panel': {'press_radius': PRESS_RADIUS,
                                                           'release_radius': RELEASE_RADIUS,
                                                           'press_dwell_ticks': PRESS_TICKS},
            'source_arm': {'link_lengths': LINK_LENGTHS, 'max_setpoint_deltas_rad': MAX_RELATIVE_SETPOINT,
                           'servo_response_at_source_tick': SERVO_RESPONSE,
                           'source_tick_seconds': REACHING_TICK_SECONDS},
            'gearing': configurations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = run(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
