"""Transfer a camera-fitted arm map to scored cursor targets under hidden wiring changes."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import BLOCK, BLOCKS_PER_TARGET, PULSE, CursorPlayer, TargetArena


SOURCES = {
    'offset_a': ('precontact-match-a-player', 'precontact-match-a'),
    'offset_b': ('precontact-match-b-player', 'precontact-match-b'),
    'rotary': ('rotary-lift-a-player', 'rotary-lift-carried-a'),
}


def targets(arena, seed, count):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(50 * count):
        arena.reset(seed=seed, render=False)
        pulse = rng.uniform(-1., 1., arena.action_size)
        for tick in range(16):
            arena.step(pulse if tick < 3 else np.zeros_like(pulse), render=False)
        point = arena.paddle.copy()
        if .09 < np.linalg.norm(point - arena.origin) < .48:
            result.append(point)
        if len(result) == count:
            return result
    raise RuntimeError('insufficient reachable target poses')


def play(arena, goals, controller, wiring_seed, session_seed, *, tape=None):
    schedule = Disturbance(delays=((0, 0), (48, 2), (120, 1)), noise=((0, .0), (48, .025)))
    interface = DelayedWiring(TT6Wiring(arena, wiring_seed), schedule, seed=session_seed)
    interface.reset(seed=session_seed, render=False)
    outcomes = []
    actions = []
    tick = 0
    for goal in goals:
        arena.set_target(goal)
        score = 0
        for _ in range(BLOCKS_PER_TARGET):
            if tape is None:
                pulse = controller.act(arena.frame())
            for frame in range(BLOCK):
                action = np.asarray(tape[tick]) if tape is not None else (pulse if frame < PULSE else np.zeros_like(pulse))
                if not np.isfinite(action).all() or np.max(abs(action)) > 1.:
                    raise ValueError('invalid command')
                _, reward, _, _, _ = interface.step(action, render=False)
                actions.append(action.tolist())
                score += reward
                tick += 1
        outcomes.append(score)
    if tape is not None and tick != len(tape):
        raise ValueError('incorrect command tape length')
    return dict(scores=outcomes, actions=actions)


def run(seed, sessions, rounds):
    rows = []
    donors = []
    for index, (site, (name, game)) in enumerate(SOURCES.items()):
        checkpoint = ROOT / 'artifacts/tt9' / f'{name}.json.gz'
        receipt = _read(ROOT / 'artifacts/tt9' / f'{game}.json.gz')
        library, _, _, _ = load(checkpoint)
        # Carry the latest fitted model, without looking up the installation.
        model = library.models[-1]
        jac = model.jacobian(np.zeros(model.action_size))[:3]
        donors.append(dict(site=site, checkpoint=_hash(checkpoint), matrix=jac.tolist()))
        settings = dict(receipt['arena'], render_mode='projected')
        arena = TargetArena(**settings)
        try:
            for session in range(sessions):
                course = int(np.random.SeedSequence([seed, index, session]).generate_state(1)[0])
                goals = targets(arena, course, rounds)
                native = receipt['installation_seed']
                for changed, wiring in [(False, native), (True, course)]:
                    policies = dict(evidence_carry=CursorPlayer(jac, exploration='evidence'),
                                    evidence_cold=CursorPlayer(np.zeros_like(jac), exploration='evidence'),
                                    evidence_fixed=CursorPlayer(jac, update=False, exploration='evidence'),
                                    evidence_reversed=CursorPlayer(-jac, exploration='evidence'),
                                    carried=CursorPlayer(jac), fixed=CursorPlayer(jac, update=False),
                                    cold=CursorPlayer(np.zeros_like(jac)), probe_first=CursorPlayer(np.zeros_like(jac), exploration='probe'),
                                    reversed=CursorPlayer(-jac))
                    runs = {key: play(arena, goals, agent, wiring, course) for key, agent in policies.items()}
                    for key, result in runs.items():
                        result.update(probes=policies[key].probes, discarded=policies[key].discarded)
                        replay = play(arena, goals, None, wiring, course, tape=result['actions'])
                        if result['scores'] != replay['scores']:
                            raise ValueError('physical target score diverged')
                    rows.append(dict(site=site, session=course, rewired=changed, wiring=wiring,
                                     goals=[g.tolist() for g in goals], runs=runs))
        finally:
            arena.close()
    return dict(schema='tt10-physical-arm-cursor-v1', seed=seed, donors=donors, sessions=sessions,
                rounds=rounds, rows=rows, implementation={f: _hash(Path(__file__).with_name(f))
                                                         for f in ['tt10_target.py','tt10_target_trial.py']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--seed', type=int, default=68371107)
    p.add_argument('--sessions', type=int, default=6)
    p.add_argument('--rounds', type=int, default=8)
    args = p.parse_args()
    if min(args.sessions, args.rounds) < 1:
        p.error('sessions and rounds must be positive')
    receipt = run(args.seed, args.sessions, args.rounds)
    _write(args.output, receipt)
    for rewired in [False, True]:
        selected = [r['runs'] for r in receipt['rows'] if r['rewired'] == rewired]
        print(rewired, {key: int(sum(sum(r[key]['scores']) for r in selected)) for key in selected[0]})
