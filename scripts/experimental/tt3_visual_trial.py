"""Reproduce paired measurements of roles bound from native Pong and table pixels."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import tt3_arena, tt3_pong, tt3_visual


FRAMES = (16, 24, 32)
VIEWS = ('ordinary', 'moving_decoy', 'fast_transverse_decoy')
METHODS = ('source_weights', 'speed_only', 'unrelated_commands')
ROLES = ('controlled', 'relevant')
MATCH_TOLERANCE = .019
SOURCE_SEEDS = tuple(range(1000, 1024))
TARGET_OFFSETS = dict(physics=100, appearance=9202, commands=371, unrelated=89201)


class _FastDecoyArena(tt3_arena.TT3Arena):
    """An evaluator-only visual manipulation with the same underlying point."""
    amplitude = .68
    angular_speed = 11.

    def decoy(self):
        phase = self.angular_speed * self.frames / tt3_arena.HZ + self._decoy_phases[0]
        return self.amplitude * math.sin(phase), .42, .31

    def decoy_speed(self):
        phase = self.angular_speed * self.frames / tt3_arena.HZ + self._decoy_phases[0]
        return abs(self.amplitude * self.angular_speed * math.cos(phase))

    def frame(self):
        return tuple(tt3_arena._render(camera, self._pose, self.rival, self.rival_normal,
                                      self.ball, self._colors, self.decoy()) for camera in (0, 1))


def _camera_position(world):
    width, height = tt3_arena.IMAGE_WIDTH, tt3_arena.IMAGE_HEIGHT
    horizontal, top = tt3_arena._project(world, 0)
    _, side = tt3_arena._project(world, 1)
    return (np.array((horizontal, top, side)) + .5 - (width / 2, height / 2, height / 2)) / width


def _matches(track, position):
    return track is not None and bool(np.linalg.norm(np.asarray(track.position) - position) < MATCH_TOLERANCE)


def _source_fit():
    practice = list(tt3_pong.source_episodes(SOURCE_SEEDS))
    model = tt3_visual.RoleModel.fit(practice)
    relocated = []
    for episode in practice:
        reward = sum(sample.reward for sample in episode)
        anchor = min(18, len(episode) - 1)
        relocated.append([replace(sample, reward=reward if index == anchor else 0.)
                          for index, sample in enumerate(episode)])
    try:
        comparison = asdict(tt3_visual.RoleModel.fit(relocated))
    except ValueError as error:
        comparison = dict(error=str(error))
    return model, comparison


def _probe(model, index, view):
    arena_type = _FastDecoyArena if view == 'fast_transverse_decoy' else tt3_arena.TT3Arena
    seeds = {name: index + offset for name, offset in TARGET_OFFSETS.items()}
    arena = arena_type(seeds['physics'], view_seed=seeds['appearance'], distractor=view != 'ordinary')
    speed_model = replace(model, relevance_weights=(1., 0., 0., 0., 0.))
    binders = tuple(tt3_visual.RoleBinder(prior) for prior in (model, speed_model, model))
    frames = arena.reset()
    for binder in binders:
        binder.update(frames)
    rng = np.random.default_rng(seeds['commands'])
    unrelated_rng = np.random.default_rng(seeds['unrelated'])
    response = {}
    faster = 0
    for tick in range(1, max(FRAMES) + 1):
        if tick % 4 == 1:
            request, unrelated = (random.choice((-1, 0, 1), size=3)
                                  for random in (rng, unrelated_rng))
        frames, _, done, _ = arena.step(request)
        if isinstance(arena, _FastDecoyArena):
            faster += arena.decoy_speed() > math.hypot(*arena.velocity)
        roles = tuple(binder.update(frames, command)
                      for binder, command in zip(binders, (request, request, unrelated), strict=True))
        if tick in FRAMES:
            control, relevant = (_camera_position(position) for position in (arena.paddle, arena.ball))
            response[tick] = {name: dict(controlled=_matches(binding.controlled, control),
                                         relevant=_matches(binding.relevant, relevant))
                              for name, binding in zip(METHODS, roles, strict=True)}
        if done:
            break
    return dict(view=view, seeds=seeds, responses=response, frames_observed=tick,
                fast_decoy_frames_above_relevant_speed=int(faster))


def _paired(rows, tick, role, comparison):
    hits = [tuple(row['responses'].get(tick, {}).get(method, {}).get(role, False)
                  for method in ('source_weights', comparison)) for row in rows]
    return dict(source_correct=sum(first for first, _ in hits),
                comparator_correct=sum(second for _, second in hits),
                source_only=sum(first and not second for first, second in hits),
                comparator_only=sum(second and not first for first, second in hits), cases=len(rows))


def run(count):
    torch.set_num_threads(1)
    model, relocation = _source_fit()
    rows = [_probe(model, index, view) for view in VIEWS for index in range(count)]
    paired = {}
    for view in VIEWS:
        subset = [row for row in rows if row['view'] == view]
        paired[view] = {tick: dict(controlled=_paired(subset, tick, 'controlled', 'unrelated_commands'),
                                   relevant=_paired(subset, tick, 'relevant', 'speed_only'))
                        for tick in FRAMES}
    sources = [Path(__file__)] + [Path(module.__file__) for module in (tt3_arena, tt3_pong, tt3_visual)]
    return dict(source_seeds=list(SOURCE_SEEDS), source_model=asdict(model),
                reward_clock_relocation=relocation, match_tolerance=MATCH_TOLERANCE,
                target_cases_per_view=count, target_seed_offsets=TARGET_OFFSETS,
                rows=rows, paired=paired,
                source_hashes={path.name: sha256(path.read_bytes()).hexdigest() for path in sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cases', type=int, default=30)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error('at least one paired case is required')
    result = run(args.cases)
    args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(json.dumps(result['paired'], indent=2))


if __name__ == '__main__':
    main()
