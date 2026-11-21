"""Replay the first studied match the carried learner won after returning the ball."""
import argparse
import json
from pathlib import Path

import torch

from arm_plays_pong import MatchCase, run_match
from arm_pong_arrival import ArrivalController
from button_pong_skill import DesiredPongPolicy
from pong_button_panel import SOURCE_GEAR, VisualButtonMotor, pretrain_motor
from render_arm_pong_rewire import Film, replay_physics


def winning_row(rows, earned_points):
    for index, row in enumerate(rows):
        if row['winner'] == 'own' and row['scored_after_own_return'] >= earned_points:
            return index, row
    raise ValueError('no completed winning match with an earned point')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('qualification', type=Path, help='match directory or arrival test JSON')
    parser.add_argument('output', type=Path)
    parser.add_argument('--earned-points', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.qualification.is_dir():
        files = sorted((args.qualification / 'matches').glob('*-both_live.json'))
        index, reference = winning_row((json.loads(file.read_text()) for file in files), args.earned_points)
        manifest = json.loads((args.qualification / 'manifest.json').read_text())
        memory = json.loads((args.qualification / 'arm-motor.json').read_text())
        name = files[index].name
        first_to, max_ticks, speed = manifest['first_to'], manifest['max_ticks'], manifest['opponent_speed']
        if manifest['gear'] != SOURCE_GEAR:
            raise ValueError('this physical replay uses original gearing')
        release_planner = False
    else:
        study = json.loads(args.qualification.read_text())
        index, reference = winning_row(study['outcomes']['release'], args.earned_points)
        memory, _ = pretrain_motor(147884320, gear=SOURCE_GEAR)
        name = f'release/{index}'
        first_to, max_ticks, speed = 3, study['limit_seconds'] * 120, .35
        release_planner = True
    case = MatchCase(**reference['case'])
    film = Film(switch=None)
    motor = VisualButtonMotor(case.controller_seed ^ 0x6D54F893, memory=memory)
    policy = DesiredPongPolicy.load(args.checkpoint, learn=True, exploration=.10)
    controller = ArrivalController(policy, motor, case.controller_seed, release=True) if release_planner else None
    match = run_match(case, policy, motor, first_to=first_to, max_ticks=max_ticks, opponent_speed=speed,
                      panel_kwargs=dict(gear=SOURCE_GEAR), controller=controller, callback=film.observe, record=True)
    for key in ('points', 'arm_actions_sha256', 'drives_sha256'):
        if json.loads(json.dumps(match[key])) != reference[key]:
            raise ValueError(f'qualification mismatch in {key}')
    counts = replay_physics(case, match['trajectory'], switch=None, opponent_speed=speed)
    film.save(args.output)
    print(json.dumps(dict(source_match=name, counts=counts)))


if __name__ == '__main__':
    main()
