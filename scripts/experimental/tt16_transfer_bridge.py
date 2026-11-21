"""Acquire held-out motion evidence and replay the frozen TT15 handoff fits."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _samples
from .tt6_body_probe import collect_probe
from .tt6_render import cache_hall_stereo
from .tt13_live_audit import assignment
from .tt14_signal_search import SignalArena
from .tt15_context_trial import play, schedule
from .tt15_discovery import DiscoveryPlayer
from .tt15_motion_transfer import Recorder, match_motion
from .tt15_pitch_probe import motion_test, one_port_fit, select
from .tt16_transfer_protocol import FIXTURES, load_optical_player, make_arena, setup_path


class Playback:
    def __init__(self, actions):
        self.actions = iter(actions)

    def act(self, rgb, reward):
        return next(self.actions)


def as_json(value):
    return json.loads(json.dumps(value))


def costs(actions):
    commands = np.asarray(actions)
    absolute = np.abs(commands)
    return dict(frames=len(actions), active_port_commands=int(np.count_nonzero(absolute > 1e-12)),
                rate_l1_frames=float(absolute.sum()),
                per_port_rate_l1_frames=absolute.sum(axis=0).tolist())


def bridge_path(folder, fixture):
    return folder / f'transfer-{fixture}-bridge.json.gz'


def harvest(folder, fixture):
    setup = setup_path(folder, fixture)
    spec = _read(setup)
    arena = make_arena(spec)
    try:
        cache_hall_stereo(arena, reflections=True)
        _, _, player, memory = load_optical_player(spec, arena)
        source = memory.source
        reference = source.jacobian(np.zeros(source.action_size))[:3]
        seed = spec['search_seed']
        search = SignalArena(**dict(spec['arena'], render_mode='projected'))
        try:
            blocks, metric = schedule(search, seed)
            end = next(i + 1 for i, b in enumerate(blocks) if b['motor'] == 1)
            blocks = blocks[:end]
            wiring = [TT6Wiring(search, value) for value in (spec['original_wiring'], spec['wiring'])]
            actor = Recorder(DiscoveryPlayer(reference))
            args = search, wiring, seed, blocks, metric, bool(spec['replica'] % 2)
            experiment = play(*args, player=actor)
            replay = Recorder(Playback(experiment['actions']))
            if play(*args, player=replay) != experiment or not np.array_equal(actor.positions, replay.positions):
                raise ValueError('search physics, RGB positions or scalar readings diverged')
            key, loss = match_motion(reference, experiment['actions'], actor.positions, actor.player.motor.curve)
            repeat_key, repeat_loss = match_motion(reference, experiment['actions'], replay.positions,
                                                   actor.player.motor.curve)
            if (key, loss) != (repeat_key, repeat_loss):
                raise ValueError('search motion fit diverged on replay')
        finally:
            search.close()

        axis, port, _, commands = motion_test(source, key[0])
        pitch_seed = int(np.random.default_rng([seed, 27318]).integers(0, 2**31-1))
        wires = TT6Wiring(arena, spec['wiring'])
        history = collect_probe(wires, pitch_seed, commands)
        if len(history) != 13:
            raise ValueError('held-out pitch query failed to finish its 12-frame budget')
        samples = _samples([history], player.tracker_factory)
        selected_key, evidence = select(source, key, port, samples)
        evidence = dict(evidence, actions=commands.tolist())
        only = one_port_fit(source, evidence)
        replayed = _samples([collect_probe(wires, pitch_seed, commands)], player.tracker_factory)
        if not (np.array_equal(samples[0], replayed[0]) and
                np.array_equal(samples[1][1], replayed[1][1]) and np.array_equal(samples[2], replayed[2], equal_nan=True)):
            raise ValueError('pitch query observations diverged on replay')
        evaluator = assignment(source.action_size, spec['original_wiring'], spec['wiring'])
    finally:
        arena.close()
    out = dict(schema='tt16-frozen-transfer-evidence', fixture=fixture, setup_sha256=_hash(setup),
               keys=dict(search=key, search_pitch=selected_key, probe_only=only['key']), truth=evaluator,
               search=dict(experiment, seed=seed, metric=metric.tolist(), schedule=blocks,
                           positions=np.asarray(actor.positions).tolist(), error_m=loss,
                           represented_frames=len(experiment['actions']),
                           changed_frames=sum(b['length'] for b in blocks if b['motor'] == 1),
                           cost=costs(experiment['actions']), replayed=True),
               pitch=dict(evidence, seed=pitch_seed, axis=axis, cost=costs(commands), replayed=True),
               one_port_fit=only, source_sha256=_hash(Path(__file__)))
    _write(bridge_path(folder, fixture), out)
    print(fixture, 'search frames', out['search']['represented_frames'], 'changed', out['search']['changed_frames'],
          'position matches', sum(a == b for a, b in zip(key[0], evaluator[0], strict=True)),
          'full search+pitch', as_json(selected_key) == evaluator, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('fixture', choices=FIXTURES)
    args = p.parse_args()
    harvest(args.folder, args.fixture)
