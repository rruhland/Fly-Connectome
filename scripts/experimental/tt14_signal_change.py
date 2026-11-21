"""Relocate a hidden signal source without resetting the arm or announcing the change."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _write
from .tt6_body_arena import TT6Wiring
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import TOLERANCE
from .tt10_target_trial import targets
from .tt14_signal_search import SearchPlayer, SignalArena, marker, port_maps


class ChangePlayer(SearchPlayer):
    def __init__(self, prior, *, detect, probe):
        super().__init__(prior, probe=probe)
        self.detect = detect
        self.errors = []
        self.suspect = 0
        self.elapsed = 0
        self.switches = []

    def act(self, rgb, reward):
        x, y, z = marker(rgb) / self.scale
        phi = np.array([1., x, y, z, x*x, y*y, z*z, x*y, x*z, y*z])
        residual = float(reward - phi @ np.linalg.solve(self.gram, self.cross))
        if self.detect and len(self.errors) >= 40:
            history = self.errors[-32:]
            median = np.median(history)
            noise = np.median(np.abs(history - median))
            unexpected = abs(residual - median) > .13 + 6*noise
            self.suspect = self.suspect + 1 if unexpected else 0
            if self.suspect >= 3:
                # Relearn the reward landscape. The physical servo and its fitted
                # map, accumulated motor commands and pose history all continue.
                self.gram = np.eye(10) * .025
                self.cross = np.zeros(10)
                self.cross[4:7] = -.35 * .025
                self.tick = 0
                self.suspect = 0
                self.errors.clear()
                self.switches.append(self.elapsed)
        self.errors.append(residual)
        self.elapsed += 1
        return super().act(rgb, reward)


def trajectory(arena, wiring, seed, goals, metric, switch_tick, offset, compressed, *, player=None, tape=None):
    noise = Disturbance(delays=((0,0),(40,2),(115,1),(180,3)), noise=((0,0.),(40,.025)))
    interface = DelayedWiring(wiring, noise, seed=seed)
    interface.reset(seed=seed, render=False)
    rng = np.random.default_rng([seed,3471])
    frames, actions, rewards = [], [], []
    for tick in range(264):
        goal = goals[int(tick >= switch_tick)]
        delta = (arena.paddle - goal)/.20
        penalty = delta @ metric @ delta
        strength = round(float(offset - (np.log1p(penalty) if compressed else penalty) + rng.normal(0,.006)), 2)
        action = tape[tick] if tape is not None else player.act(arena.frame(), strength)
        interface.step(action, render=False)
        frames.append(int(np.linalg.norm(arena.paddle-goal) <= TOLERANCE))
        actions.append(np.asarray(action).tolist())
        rewards.append(strength)
    after = frames[switch_tick:]
    held = np.flatnonzero(np.convolve(after, np.ones(4),mode='valid') == 4)
    return dict(actions=actions,rewards=rewards, before=sum(frames[:switch_tick]), after=sum(after),
                early=sum(after[:48]),late=sum(after[-48:]),acquired=int(held[0]+4) if held.size else None)


def run(site, output, seed, count):
    priors, parents, settings, ports = port_maps(Path('artifacts/tt13'),site)
    arena = SignalArena(**dict(settings,render_mode='projected'))
    rows = []
    try:
        wiring = TT6Wiring(arena,ports)
        for i, s in enumerate(np.random.default_rng(seed).integers(0,2**31-1,count).tolist()):
            rng = np.random.default_rng([s,571])
            candidates = targets(arena,s,12)
            first = candidates[0]
            second = next(p for p in candidates if np.linalg.norm(p-first) > .14)
            phase = int(rng.integers(102,135))
            axes = Rotation.random(random_state=rng).as_matrix()
            metric = (axes*rng.uniform(.17,.65,3)) @ axes.T
            offset = rng.uniform(1.,3.)
            compressed = bool(i%2)
            cases = {}
            for changed in (True,False):
                goals = (first,second if changed else first)
                for name, prior, detect, probe in (
                    ('carry',priors['remap'],False,False),
                    ('detect',priors['remap'],True,False),
                    ('cold',np.zeros_like(priors['remap']),True,True)):
                    player = ChangePlayer(prior,detect=detect,probe=probe)
                    args = arena,wiring,s,goals,metric,phase,offset,compressed
                    row = trajectory(*args,player=player)
                    if row != trajectory(*args,tape=row['actions']):
                        raise ValueError('replayed signal observations or task outcomes differ')
                    cases[(name if changed else name+'_stable')] = dict(row,switches=player.switches)
            for name in ('carry','detect','cold'):
                if cases[name]['actions'][:phase] != cases[name+'_stable']['actions'][:phase]:
                    raise ValueError('policy moved before observing a change')
            rows.append(dict(seed=s,phase=phase,first=first.tolist(),second=second.tolist(),compressed=compressed,
                             offset=offset,metric=metric.tolist(),cases=cases))
    finally:
        arena.close()
    _write(output,dict(rows=rows,seed=seed,parents=parents,implementation=_hash(Path(__file__)),
                       base=_hash(Path(__file__).with_name('tt14_signal_search.py'))))
    print({k:sum(r['cases'][k]['after'] for r in rows) for k in rows[0]['cases']},flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site',choices=('gantry','five'))
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=8)
    a = p.parse_args()
    run(a.site,a.output,a.seed,a.count)
