"""Measure native opening reach with an initially empty request-delay queue."""

import argparse
from pathlib import Path

import torch

from fly_connectome.pong import Physics, Pong
from run_pong_inverse_cycle import save
from run_pong_rapid_recovery_qualification import read


def opening_intervals(cases):
    """Execute opposite extreme requests after the forced neutral interval."""
    seeds = [case['seeds'][1] for case in cases]
    size = len(seeds)
    game = Pong(seeds * 2, Physics(opponent_speed=.65))
    physics = game.config
    neutral_ticks = torch.tensor([case['delay'] * 12 // 5 for case in cases] * 2)
    drives = torch.tensor([-1.] * size + [1.] * size)
    plane = physics.paddle_x + physics.ball_radius
    margin = physics.paddle_half + physics.ball_radius
    openings = {}
    for tick in range(2400):
        x, y = game.ball[:size].T
        vx, vy = game.ball_velocity[:size].T
        incoming = (vx < 0) & (x >= plane) & (x + vx * physics.dt <= plane)
        next_y = y + vy * physics.dt
        next_y = torch.where(next_y < physics.ball_radius, 2 * physics.ball_radius-next_y, next_y)
        next_y = torch.where(next_y > 1-physics.ball_radius, 2*(1-physics.ball_radius)-next_y, next_y)
        game.step(drives * (tick >= neutral_ticks), auto_reset=False)
        for i in incoming.nonzero().flatten().tolist():
            if i in openings:
                continue
            lower = float(game.body.position[size+i])
            upper = float(game.body.position[i])
            target = float(next_y[i])
            openings[i] = dict(tick=tick+1, ball_vertical=target, paddle_interval=[lower, upper],
                               feasible=lower-margin <= target <= upper+margin)
        if len(openings) == size:
            break
    return openings


def summarize(directory):
    cases = read(directory / 'schedules.json')
    intervals = opening_intervals(cases)
    items = []
    for index, case in enumerate(cases):
        opening = intervals.get(index)
        row = dict(index=case['index'], seed=case['seeds'][1], delay_commands=case['delay'], opening=opening)
        for arm in ('live', 'timing_frozen'):
            source = directory / 'run' / arm / f"stream-{case['index']:02d}.json"
            episode = read(source)['episodes'][1]
            events = episode['events']
            if ((events[0]['tick'] if events else None) != (opening['tick'] if opening else None)):
                raise ValueError('opening timing differs between reach computation and recorded play')
            returned = bool(events and events[0]['hit'])
            if returned and not opening['feasible']:
                raise ValueError('observed return lies outside the reach interval')
            row[f'{arm}_first_return'] = returned
        items.append(row)
    feasible = [row for row in items if row['opening'] and row['opening']['feasible']]
    return dict(planned=len(items), first_opportunities=len(intervals), inside_extremal_interval=len(feasible),
                outside_extremal_interval=len(intervals)-len(feasible),
                live_returns_inside=sum(row['live_first_return'] for row in feasible),
                timing_frozen_returns_inside=sum(row['timing_frozen_first_return'] for row in feasible),
                included_in_primary_denominator=len(items), openings=items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qualification_directory', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    save(arguments.output, summarize(arguments.qualification_directory))
