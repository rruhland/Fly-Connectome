"""Identify a five-to-six action-effect transfer with unlabelled RGB probes."""

import argparse
from contextlib import ExitStack, closing
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from .tt24_bounce_ident import WristPupil, _INC
from .tt24_bounce_trial import _write, seeds, verify_replay
from .tt24_redundant import MarkerBounce, RedundantBounce


ROOT = Path('artifacts/tt24')
POLICY = ROOT / 'bounce-ident-pupil.npz'
OUTPUT = ROOT / 'redundant-game.json.gz'
DRIVE = 6
SETTLE = 9


def _centroid(mask):
    regions, n = label(mask)
    if not n:
        return None
    histogram = np.bincount(regions.ravel())
    histogram[0] = 0
    y, x = np.nonzero(regions == histogram.argmax())
    return np.array((x.mean(), y.mean()))


def markers(arena):
    """Measure three contrasting paddle sites in two ordinary rendered images."""
    outputs = []
    for rgb in arena.probe_images():
        r, g, b = np.moveaxis(rgb, -1, 0)
        masks = ((g > 123) & (b > 123) & (r < .40 * g),
                 (r > 115) & (b > 115) & (g < .40 * r),
                 (g > 118) & (r < .42 * g) & (b < .42 * g))
        for mask in masks:
            found = _centroid(mask)
            if found is None:
                return None
            outputs.append(found / rgb.shape[0])
    return np.concatenate(outputs)


def conduct(arena, action):
    emitted = np.zeros(arena.action_size)
    emitted[:len(action)] = action
    for tick in range(DRIVE + SETTLE):
        obs, _, done = arena.step(emitted if tick < DRIVE else np.zeros_like(emitted), render=False)
        if done:
            raise ValueError('probe session exceeded its charged duration')
    return obs


def _fit(x, y, ridge):
    return np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)


def inverse(rows, ridge):
    return rows.T @ np.linalg.inv(rows @ rows.T + ridge * np.eye(len(rows)))


def measure(arena, seed, *, active, repetitions):
    """Observe commands, joint changes and common camera effects of paired probes."""
    rng = np.random.default_rng(seed + 362)
    observed = []
    obs = arena.reset(seed=seed, split='practice', render=False)
    sensed = markers(arena)
    commands = []
    for _ in range(repetitions):
        directions = np.eye(active)[rng.permutation(active)]
        commands.extend(.24 * direction * sign for direction in directions for sign in (1., -1.))
    for command in commands:
        initial = obs['joints'][:active].copy()
        new = conduct(arena, command)
        final = markers(arena)
        if sensed is not None and final is not None:
            observed.append((command, new['joints'][:active] - initial, final - sensed))
        obs, sensed = new, final
    if len(observed) < active + 1:
        raise ValueError(f'only {len(observed)} usable probe blocks for {active} dimensions')
    u, q, optical = (np.asarray(items) for items in zip(*observed, strict=True))
    optical_joint = _fit(q, optical, .00018)
    optical_drive = _fit(u, optical, .00085)
    joint_drive = _fit(u, q, .00085)
    return dict(command=optical_drive, encoder=optical_joint, actuator=joint_drive,
                evidence=dict(total_blocks=len(commands), observed_blocks=len(u), frames=len(commands) * (DRIVE + SETTLE),
                              resets=1, seed=seed, calibration_views=2,
                              rank=int(np.linalg.matrix_rank(optical_drive)),
                              command_singular_values=np.linalg.svd(optical_drive, compute_uv=False).tolist(),
                              commands=u.tolist(), encoders=q.tolist(), optical=optical.tolist()))


def grounding(source, target):
    axes = len(target['encoder'])
    legacy = np.eye(axes, 5)
    virtual = legacy.copy()
    if axes > 5:
        virtual[5:] = target['encoder'][5:] @ inverse(source['encoder'], .0013)
    effect = target['actuator'] @ virtual
    donor = source['actuator']
    transport = donor @ inverse(effect, .00022)
    # Keep the full correspondence in the receipt, including the independent new rate column.
    direct = source['command'] @ inverse(target['command'], .00026)
    return dict(virtual=virtual, transmit=transport, optical_transmit=direct,
                residual=float(np.linalg.norm(transport @ effect - donor) / np.linalg.norm(donor)))


def play(arena, seed, policy_path, transport, virtual, *, stationary=False):
    learner = WristPupil.load(policy_path)
    obs = arena.reset(seed=seed, split='held', render=not stationary)
    home = obs['joints'].copy()
    active = len(virtual)
    actions, stock_proposals, excursions = [], [], []
    score = 0.
    done = False
    while not done:
        sensor = obs | dict(joints=home[:5] + (obs['joints'][:active] - home[:active]) @ virtual,
                            joint_velocity=obs['joint_velocity'][:active] @ virtual)
        proposal = np.zeros(5) if stationary else learner.command(sensor)
        rate = proposal @ transport
        scale = max(1., float(np.max(np.abs(rate))))
        extra = np.zeros(arena.action_size - len(rate))
        action = np.r_[rate / scale, extra]
        if scale > 1.:
            learner.issued -= _INC * proposal * (1. - 1. / scale)
        obs, reward, done = arena.step(action, render=not stationary)
        score += reward
        actions.append(action.tolist())
        stock_proposals.append(proposal.tolist())
        excursions.append((obs['joints'] - home).tolist())
    metrics = arena.judge_metrics()
    displacement = np.asarray(excursions)
    extra_peak = float(np.max(np.abs(displacement[:, 5]))) if arena.action_size == 6 else None
    near_stop = (float(np.mean(np.abs(displacement[:, 5] + home[5]) >= .47))
                 if arena.action_size == 6 else None)
    return metrics | dict(score=score, actions=actions, source_commands=stock_proposals,
                          max_encoder_travel=np.max(np.abs(displacement), axis=0).tolist(), sixth_travel_rad=extra_peak,
                          sixth_stop_fraction=near_stop)


def aggregate(rows):
    counts = [row['rebounds'] for row in rows]
    longest = [row['longest_consecutive_rebounds'] for row in rows]
    travel = [row['sixth_travel_rad'] for row in rows if row['sixth_travel_rad'] is not None]
    cmd = [action[5] for row in rows if row['sixth_travel_rad'] is not None for action in row['actions']]
    return dict(mean=float(np.mean(counts)), total=sum(counts), longest=float(np.mean(longest)), max=max(longest),
                counts=counts, runs=longest, sixth_peak=float(np.max(travel)) if travel else None,
                sixth_median_travel=float(np.median(travel)) if travel else None,
                sixth_rate_rms=float(np.sqrt(np.mean(np.square(cmd)))) if cmd else None)


def run(options):
    marker_options = dict(view='none', width=192, height=144, calibration=True, max_steps=400)
    main_options = dict(view='wrist', width=192, height=144, max_steps=300)
    starts = seeds(options.seed + 1, options.held)
    report = dict(schema='tt24-redundant-game-1', source_policy=str(options.policy),
                  source_sha256=sha256(options.policy.read_bytes()).hexdigest(), seed=options.seed, launch_seeds=starts,
                  calibration=dict(), outcome=dict(), stated_encoder_correspondence='legacy reset angles and order')
    with ExitStack() as stack:
        open_arena = lambda typ, **kw: stack.enter_context(closing(typ(**kw)))
        calibration = {name: open_arena(typ, **marker_options, **extra) for name, typ, extra in
                       (('stock', MarkerBounce, {}), ('six', RedundantBounce, {}), ('pinned', RedundantBounce, {'pinned': True}))}
        data = {name: measure(arena, options.seed, active=6 if name == 'six' else 5,
                              repetitions=options.repetitions) for name, arena in calibration.items()}
        for name, information in data.items():
            report['calibration'][name] = information['evidence'] | {k: information[k].tolist()
                                                                    for k in ('command', 'encoder', 'actuator')}
        ground = {name: grounding(data['stock'], data[name]) for name in ('six', 'pinned')}
        report['grounding'] = {n: {k: x.tolist() if isinstance(x, np.ndarray) else x for k, x in value.items()}
                               for n, value in ground.items()}
        main = {name: open_arena(typ, **main_options, **extra) for name, typ, extra in
                (('stock', MarkerBounce, {}), ('six', RedundantBounce, {}), ('pinned', RedundantBounce, {'pinned': True}))}
        conditions = dict(stock=('stock', np.eye(5), np.eye(5), False),
                          stationary=('six', np.zeros((5, 6)), np.eye(6, 5), True),
                          padded=('six', np.eye(5, 6), np.eye(6, 5), False),
                          learned=('six', ground['six']['transmit'], ground['six']['virtual'], False),
                          pinned_learned=('pinned', ground['pinned']['transmit'], ground['pinned']['virtual'], False))
        for condition, (name, transport, virtual, stationary) in conditions.items():
            episodes = [play(main[name], seed, options.policy, transport, virtual, stationary=stationary) for seed in starts]
            for item in episodes:
                verify_replay(main[name], item)
            summary = aggregate(episodes)
            report['outcome'][condition] = dict(summary=summary, drops=episodes)
            print(condition, summary, flush=True)
            _write(options.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=62231803)
    parser.add_argument('--held', type=int, default=12)
    parser.add_argument('--repetitions', type=int, choices=(1, 2), default=1)
    parser.add_argument('--policy', type=Path, default=POLICY)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    options = parser.parse_args()
    if options.held < 1:
        parser.error('held sample count must be positive')
    run(options)


if __name__ == '__main__':
    main()
