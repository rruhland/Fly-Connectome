"""Carry a cursor learner through hidden motor rewiring and restore it mid-game."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import BLOCK, BLOCKS_PER_TARGET, PULSE, CursorPlayer, TargetArena
from .tt10_target_trial import SOURCES, targets


def cursor_state(agent):
    return dict(prior=agent.prior.tolist(), update=agent.update, exploration=agent.exploration,
                covariance=agent.covariance.tolist(), cross=agent.cross.tolist(),
                count=agent.count, probes=agent.probes, conflicts=agent.conflicts, discarded=agent.discarded,
                command=agent.command.tolist(), requested=agent.requested.tolist(),
                previous=None if agent.previous is None else agent.previous.tolist())


def restore_cursor(state):
    agent = CursorPlayer(np.array(state['prior']), update=state['update'], exploration=state['exploration'])
    for key in ('covariance', 'cross', 'command', 'requested'):
        setattr(agent, key, np.asarray(state[key]).copy())
    for key in ('count', 'probes', 'conflicts', 'discarded'):
        setattr(agent, key, state[key])
    agent.previous = None if state['previous'] is None else np.asarray(state['previous']).copy()
    return agent


def play(arena, native, seed, goals, agent, *, restore=None):
    second = seed
    wiring = DelayedWiring(TT6Wiring(arena, native), Disturbance(delays=((0, 0), (90, 2), (240, 1)),
                                                                noise=((0, 0.), (90, .025))), seed=seed)
    wiring.reset(seed=seed, render=False)
    rows = []
    saved = None
    for i, goal in enumerate(goals):
        if i == 4 or i == 10:
            # The physical arm, pending commands, controller and probe budget
            # keep their current state as the cable mapping changes.
            wiring.wiring = TT6Wiring(arena, second if i == 4 else native)
        if i == 8:
            saved = cursor_state(agent)
            if restore is not None:
                if saved != restore:
                    raise ValueError('reconstructed checkpoint prefix diverged')
                agent = restore_cursor(restore)
        arena.set_target(goal)
        score = 0
        actions = []
        for _ in range(BLOCKS_PER_TARGET):
            cmd = agent.act(arena.frame())
            for frame in range(BLOCK):
                act = cmd if frame < PULSE else np.zeros_like(cmd)
                _, reward, _, _, _ = wiring.step(act, render=False)
                actions.append(act.tolist())
                score += reward
        rows.append(dict(score=score, actions=actions))
    return rows, saved, cursor_state(agent)


def run(site, seed, sessions, output, checkpoint):
    name, game = SOURCES[site]
    source_path = ROOT/'artifacts/tt9'/f'{name}.json.gz'
    start = _read(ROOT/'artifacts/tt9'/f'{game}.json.gz')
    model = load(source_path)[0].models[-1]
    prior = model.jacobian(np.zeros(model.action_size))[:3]
    arena = TargetArena(**dict(start['arena'], render_mode='projected'))
    entries = []
    try:
        for s in np.random.default_rng(seed).integers(0, 2**31 - 1, size=sessions).tolist():
            goals = targets(arena, s, 16)
            policies = dict(carried=CursorPlayer(prior, exploration='evidence'),
                            cold_probes=CursorPlayer(np.zeros_like(prior), exploration='probe'),
                            frozen=CursorPlayer(prior, update=False, exploration='evidence'))
            runs = {}
            for name, policy in policies.items():
                tape, state, final = play(arena, start['installation_seed'], s, goals, policy)
                runs[name] = dict(points=tape, final=final)
                if name == 'carried' and not entries:
                    full = _read(source_path)
                    full['target_cursor'] = state
                    _write(checkpoint, full)
                    fresh = CursorPlayer(prior, exploration='evidence')
                    repeated, _, end = play(arena, start['installation_seed'], s, goals, fresh,
                                            restore=_read(checkpoint)['target_cursor'])
                    if repeated != tape or end != final:
                        raise ValueError('saved cursor learner changed its physical continuation')
            entries.append(dict(seed=s, runs=runs, goals=[z.tolist() for z in goals]))
    finally:
        arena.close()
    _write(output, dict(schema='tt10-midgame-wiring-v1', site=site, seed=seed, source=_hash(source_path),
                        checkpoint=_hash(checkpoint), entries=entries, switch_targets=[4, 10],
                        implementation=_hash(Path(__file__))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site', choices=SOURCES)
    p.add_argument('output', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--sessions', type=int, default=8)
    a = p.parse_args()
    run(a.site, a.seed, a.sessions, a.output, a.checkpoint)
