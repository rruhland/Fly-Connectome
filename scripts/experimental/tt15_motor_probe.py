"""Separate suspicious game motion from a repeatable dry motor response."""

import numpy as np

from .tt6_body_probe import collect_probe
from .tt6_continuous_trial import prepare_point
from .tt8_service_trial import PointInterface
from .tt13_live_ports import effective
from .tt14_gain_model import GainMemory, Observation, collect, fit_transform


def pulse_commands(width):
    commands = np.zeros((6 * width + 4, width))
    for channel in range(width):
        start = 6 * channel
        commands[start:start+3, channel] = .66
        commands[start+3:start+6, channel] = -.66
    return commands


def measure(wires, seed, tracker_factory):
    arena = wires.arena
    probe = PointInterface(wires, prepare_point(arena, 'ordinary'), server=-1, seed=seed)
    tracker = tracker_factory()
    poses = []
    for action, frames in collect_probe(probe, seed, pulse_commands(arena.action_size)):
        tracker.update(frames)
        blade = tracker.actor
        poses.append([*blade.position, tracker.actor_pitch] if blade is not None and blade.observed else [None] * 4)
    return poses


def paired_observation(reference, current, width):
    repeat, measured = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    n = min(len(repeat), len(measured))
    positions = (measured[:n] - measured[0]) - (repeat[:n] - repeat[0])
    ticks = np.flatnonzero(np.isfinite(positions[:, :3]).all(axis=1))
    moves = np.vstack([np.zeros(width), pulse_commands(width)])
    return Observation(moves, ticks, positions[ticks])


class ProbeMemory(GainMemory):
    def __init__(self, recalled, reference):
        super().__init__(recalled.source, 'hybrid')
        self.active = recalled.active
        self.maps = recalled.maps
        self.reference = reference

    def observe(self, actions, trace, acquire):
        game = collect(actions, trace, self.source.action_size)
        current = self.maps[self.active]
        update = dict(previous=self.active, active=self.active, bank=len(self.maps))
        if min(game.train.sum(), (~game.train).sum()) < 5:
            return dict(update, reason='insufficient sightings')
        errors = current.errors(game)
        update['errors'] = errors
        if max(errors) < .065:
            return dict(update, reason='no optical surprise')

        pulse = paired_observation(self.reference, acquire(), self.source.action_size)
        if len(pulse.ticks) < 16:
            return dict(update, reason='insufficient pulse overlap')
        discrepancy = float(np.linalg.norm(pulse.poses[:, :3], axis=1).mean())
        update['pulse_difference'] = discrepancy
        if discrepancy < .012:
            self.active = 0
            return dict(update, active=0, reason='original response', candidate='original')

        base = self.source
        original_state = effective(pulse.moves, base.kernels)[pulse.ticks]
        nominal = base.pose(original_state)
        info = Observation(pulse.moves, pulse.ticks, nominal + pulse.poses)
        choices = list(self.maps)
        for evidence in ([info], [game], [info, game]):
            for kind in ('gain', 'coupling'):
                candidate = fit_transform(base, evidence, current.transform, kind)
                if candidate is not None:
                    choices.append(candidate)

        def loss(map):
            states = effective(pulse.moves @ map.transform.T, map.base.kernels)[pulse.ticks]
            predicted = map.base.pose(states)[:, :3] - nominal[:, :3]
            motion = float(np.linalg.norm(predicted - pulse.poses[:, :3], axis=1).mean())
            scene = float(np.mean(map.errors(game)))
            return motion, scene

        physical_current, scene_current = loss(current)
        options = []
        for i, candidate in enumerate(choices):
            physical, scene = loss(candidate)
            if physical + .004 < physical_current and scene < scene_current + .008:
                options.append((physical + .55 * scene, i, candidate, physical, scene))
        if not options:
            return dict(update, reason='repeated response did not support an alternative')
        _, index, candidate, physical, scene = min(options, key=lambda row: row[0])
        if index >= len(self.maps):
            self.maps.append(candidate)
            index = len(self.maps) - 1
        self.active = index
        return dict(update, active=index, bank=len(self.maps), candidate=candidate.kind,
                    reason='paired movement supported a correction', pulse_error=physical, game_error=scene)
