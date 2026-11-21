"""Find an invisible signal source with an arm and a scalar strength reading."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import SCALE, SIZE, TOLERANCE
from .tt10_target_trial import targets
from .tt11_pursuit_learning import LocalServo
from .tt11_tracking import TrackingArena


class SignalArena(TrackingArena):
    def frame(self):
        # The task renders just the tool. The source has no visible marker.
        rgb = super().frame()
        rgb[..., 1] = 0
        return rgb


def marker(rgb):
    xy = np.nonzero(rgb[0, :, :, 0])
    xz = np.nonzero(rgb[1, :, :, 0])
    return (np.array([xy[1].mean(), xy[0].mean(), xz[0].mean()]) - SIZE//2) / SCALE


class SearchPlayer:
    """Fit a smooth local reward surface from RGB positions and public readings."""
    def __init__(self, prior, probe):
        self.motor = LocalServo(prior, probe=probe)
        self.gram = np.eye(10) * .025
        self.cross = np.zeros(10)
        self.cross[4:7] = -.35 * .025
        self.tick = 0
        self.goal = np.zeros(3)
        self.scale = .20
        self.probes = .12 * np.array([[1,1,1], [-1,1,-1], [1,-1,-1], [-1,-1,1]])

    def act(self, rgb, reward):
        position = marker(rgb) / self.scale
        x, y, z = position
        features = np.r_[1., position, position**2, x*y, x*z, y*z]
        self.gram += np.outer(features, features)
        self.cross += features * reward
        if self.tick % 4 == 0:
            if self.tick < 36:
                self.goal = self.probes[self.tick//9]
            else:
                weights = np.linalg.solve(self.gram, self.cross)
                hessian = np.diag(2 * weights[4:7])
                hessian[np.triu_indices(3, 1)] = weights[7:]
                hessian[np.tril_indices(3, -1)] = weights[7:]
                eigen, axes = np.linalg.eigh(-hessian)
                # Keep uncertain directions locally concave while more data arrives.
                curvature = (axes * np.maximum(eigen, .25)) @ axes.T
                optimum = np.linalg.solve(curvature, weights[1:4])
                self.goal = self.scale * np.clip(optimum, -2.3, 2.3)
        # A virtual waypoint invokes the same RGB motor servo as in pursuit.
        # Only this policy writes the green channel.
        image = rgb.copy()
        x, y, z = np.clip(np.rint(self.goal*SCALE).astype(int), -SIZE//2+2, SIZE//2-3) + SIZE//2
        for view, row in enumerate((y,z)):
            image[view,row-1:row+2,x-1:x+2,1] = 240
        self.tick += 1
        return self.motor.act(image)


def play(arena, wiring, seed, target, field, *, player=None, tape=None, flat=False):
    noise = Disturbance(delays=((0,0),(42,2),(96,1)), noise=((0,0.),(42,.025)))
    interface = DelayedWiring(wiring, noise, seed=seed)
    interface.reset(seed=seed, render=False)
    arena.set_target(target)
    rng = np.random.default_rng([seed,587])
    distances, readings, actions = [], [], []
    for tick in range(144):
        # This analog reward encodes field strength, not coordinates. Its
        # amplitude, offsets and three principal directions vary per source.
        delta = (arena.paddle-target)/.20
        penalty = delta @ field['metric'] @ delta
        if field['profile'] == 'compressed':
            penalty = np.log1p(penalty)
        strength = round(float(field['offset'] - penalty + rng.normal(0,.006)), 2)
        action = tape[tick] if tape is not None else player.act(arena.frame(), 0. if flat else strength)
        if np.max(np.abs(action)) > 1. or not np.isfinite(action).all():
            raise ValueError('invalid action')
        interface.step(action, render=False)
        distances.append(float(np.linalg.norm(arena.paddle-target)))
        readings.append(strength)
        actions.append(np.asarray(action).tolist())
    on = np.asarray(distances) <= TOLERANCE
    held = np.flatnonzero(np.convolve(on.astype(int), np.ones(4), mode='valid') == 4)
    return dict(actions=actions, readings=readings, frames=int(on.sum()),
                quarters=on.reshape(4,-1).sum(axis=1).tolist(),
                acquired_at=int(held[0]+4) if held.size else None,
                final_distance=float(np.median(distances[-16:])))


def port_maps(root, site):
    priors, parents = {}, {}
    for mode in ('fixed', 'remap'):
        receipt = root/f'live-{site}-{mode}.json.gz'
        checkpoint = root/f'live-{site}-{mode}-player.json.gz'
        data = _read(receipt)
        archive = _read(checkpoint)['port_memory']
        active = data['games'][0]['points'][-1]['choice']['active']
        model = load(checkpoint)[0].models[archive['indices'][active]]
        priors[mode] = model.jacobian(np.zeros(model.action_size))[:3]
        parents[mode] = dict(receipt=_hash(receipt), checkpoint=_hash(checkpoint), active=active)
    return priors, parents, data['arena'], data['new_ports']


def trial(priors, parents, settings, wiring_factory, output, seed, count):
    arena = SignalArena(**dict(settings,render_mode='projected'))
    rows = []
    try:
        wiring = wiring_factory(arena)
        for i, s in enumerate(np.random.default_rng(seed).integers(0,2**31-1,count).tolist()):
            rng = np.random.default_rng([s,773])
            target = targets(arena,s,1)[0]
            axes = Rotation.random(random_state=rng).as_matrix()
            metric = (axes * rng.uniform(.17,.65,3)) @ axes.T
            field = dict(offset=rng.uniform(1.,3.), metric=metric, profile=('quadratic','compressed')[i % 2])
            reference = next(iter(priors.values()))
            last = list(priors)[-1]
            cases = dict(priors, cold=np.zeros_like(reference), direct=priors[last], flat=priors[last])
            results = {}
            for key, prior in cases.items():
                player = SearchPlayer(prior, probe=(key!='direct'))
                result = play(arena,wiring,s,target,field,player=player,flat=key=='flat')
                if result != play(arena,wiring,s,target,field,tape=result['actions']):
                    raise ValueError('signal-task replay diverged')
                results[key] = result
            prefix = 4 * reference.shape[1]
            for name in (*priors, 'flat'):
                if results[name]['actions'][:prefix] != results['cold']['actions'][:prefix]:
                    raise ValueError('equal motor probe controls diverged')
            rows.append(dict(seed=s,target=target.tolist(),metric=metric.tolist(),offset=field['offset'],
                             profile=field['profile'],cases=results))
    finally:
        arena.close()
    _write(output, dict(implementation=_hash(Path(__file__)), seed=seed, parents=parents, rows=rows))
    print({k:(sum(r['cases'][k]['frames'] for r in rows),
              sum(r['cases'][k]['acquired_at'] is not None for r in rows)) for k in rows[0]['cases']},flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site',choices=('gantry','five'))
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=12)
    args = p.parse_args()
    priors, parents, settings, ports = port_maps(Path('artifacts/tt13'),args.site)
    trial(priors,parents,settings,lambda arena: TT6Wiring(arena,ports),args.output,args.seed,args.count)
