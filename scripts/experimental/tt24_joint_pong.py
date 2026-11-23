"""Train one lateral readout with both actual Pong and hinged-donor reward."""

import argparse
from time import perf_counter

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_donor_trial import DESIGNS, ROOT, Native, World, _exploration


TAG = 'tt24_joint_pong'
PLAN = ROOT / f'{TAG}-plan.json'
WARM = ROOT / f'{TAG}-warm.json.gz'
OUTPUT = ROOT / f'{TAG}.json.gz'
DONOR = ROOT / 'tt24_donor-refined.npz'
DONOR_LINEAGE = ROOT / 'tt24_donor-refined.json.gz'
WARM_SEED = 219894453
REPLICATES = (65898032, 272587403, 869080873)
WARM_ITERS = 9
UPDATES = 8
DIRECTIONS = 5
HEIGHT, WIDTH = 48, 32
STRIDE = 4
CAP = 6 * 120
ARM_CAP = 240
ARM_GAIN = .25
MODES = ('joint', 'shuffled')
MODE_COLUMNS = ((1, 2), (4, 5), (7, 8), (15, 16))
ROWS = (np.arange(HEIGHT) + .5) / HEIGHT
COLUMNS = (np.arange(3, WIDTH - 3) + .5) / WIDTH


def field(frame, old, velocity, visible):
    paddle = frame[:, :, 1]
    actors = (paddle @ ROWS) / np.maximum(1, paddle.sum(axis=1))
    interior = frame[:, :, 3:-3]
    counts = interior.sum(axis=(1, 2))
    measured = np.column_stack((interior.sum(axis=1) @ COLUMNS,
                                interior.sum(axis=2) @ ROWS)) / np.maximum(1, counts[:, None])
    seen = counts > 0
    both = seen & visible
    velocity[both] = .69 * velocity[both] + .31 * (measured[both] - old[both]) * (120 / STRIDE)
    velocity[~both] = 0.
    old[seen] = measured[seen]
    visible[:] = seen
    return actors, seen


@torch.no_grad()
def arcade(weights, seeds, *, cap=CAP, opponent=.45, witness=None, paddle_factory=None):
    """Run the installed batched Pong native physics and its actual raster."""
    weights = np.repeat(np.asarray(weights).reshape(-1, len(MODE_COLUMNS)), len(seeds), axis=0)
    game = Pong(list(seeds) * (len(weights) // len(seeds)), Physics(opponent_speed=opponent))
    if paddle_factory is not None:
        game.body = paddle_factory(game.config)
    current = np.ones(len(weights), dtype=bool)
    old = np.zeros((len(weights), 2))
    velocity = np.zeros_like(old)
    seen_previous = np.zeros(len(weights), dtype=bool)
    score, hits, age = [np.zeros(len(weights)) for _ in range(3)]
    action = np.zeros(len(weights))
    previous_paddle = np.zeros(len(weights))
    paddle_speed = np.zeros(len(weights))
    for step in range(cap):
        if step % STRIDE == 0:
            frame = game.render(HEIGHT, WIDTH).numpy()
            paddle, seen = field(frame, old, velocity, seen_previous)
            if step == 0:
                anchor = paddle.copy()
                previous_paddle = paddle.copy()
            gap = np.clip((old[:, 1] - paddle) * .76 / .065, -3., 3.)
            tangent = np.clip(velocity[:, 1] * 3. / .38, -3., 3.)
            pose = np.clip((paddle - anchor) * .76 / .04, -3., 3.)
            normal = np.clip(velocity[:, 0] * 3. / 1.5, -3., 3.)
            observation = np.stack((gap, tangent, pose, np.clip(gap * normal, -3., 3.)), axis=1)
            offset = .36 * np.tanh((observation * weights).sum(axis=1))
            paddle_speed = .70 * paddle_speed + .30 * (paddle - previous_paddle) * (120 / STRIDE)
            if witness is not None:
                witness.sample(paddle, paddle_speed, old, velocity, seen & current)
            request = np.clip(-11. * (anchor + offset - paddle) + 1.35 * paddle_speed, -1., 1.)
            # Native motor convention: positive analog drive moves the image upward.
            # A missing mover leaves the last issued request in place until another camera return.
            action[seen] = request[seen]
            previous_paddle = paddle
        result = game.step(torch.from_numpy(action.astype(np.float32)), auto_reset=False)
        scored = result.scores.numpy()
        age += current
        contacts = result.hits.numpy()
        hits += contacts * current
        if witness is not None:
            witness.outcome(contacts, scored, current)
        score += scored * current
        current &= scored == 0
        if not current.any():
            break
    return np.column_stack((score, hits, age)).reshape(-1, len(seeds), 3)


def native_reward(result):
    return (result[..., 0] + .43 * result[..., 2] / CAP).mean(axis=1)


def population(weights, directions, amplitude):
    shifts = amplitude * directions
    return np.clip(np.r_[weights[None], weights + shifts, weights - shifts], -3., 3.)


def preference(values):
    good = values[1:1+DIRECTIONS]
    bad = values[1+DIRECTIONS:]
    scale = max(.12, float(np.std(np.r_[good, bad])))
    return np.clip((good - bad) / scale, -2.5, 2.5)


def warm(plan):
    if WARM.exists():
        receipt = _read(WARM)
        if receipt['plan'] != _hash(PLAN):
            raise ValueError('the warmed readout has different lineage')
        return receipt
    rng = np.random.default_rng(WARM_SEED)
    readout = np.zeros(len(MODE_COLUMNS))
    history = []
    start = perf_counter()
    for iteration, starts in enumerate(plan['warm_train']):
        noise = rng.normal(size=(DIRECTIONS, len(MODE_COLUMNS)))
        knobs = population(readout, noise, .38 / (1. + .035 * iteration))
        result = arcade(knobs, starts)
        fitness = native_reward(result)
        best = knobs[int(np.argmax(fitness))]
        readout = np.clip(.65 * best + .35 * readout + .16 * np.mean(noise * preference(fitness)[:, None], axis=0), -3., 3.)
        history.append(dict(state=readout.tolist(), fitness=fitness.tolist(), returns=int(result[..., 1].sum())))
        print('warm native', iteration + 1, np.round(readout, 3), round(perf_counter() - start, 1), flush=True)
    held = arcade([np.zeros(len(MODE_COLUMNS)), readout], plan['warm_held'])
    paused, playing = (int(np.sum(outcomes[:, 1] >= 1)) for outcomes in held)
    passes = playing >= 15 and playing >= paused + 4
    result = dict(plan=_hash(PLAN), warm_trace=history, weights=readout.tolist(), held=held.tolist(), gate=passes,
                  held_with_return=dict(shared=playing, stationary=paused), seconds=perf_counter() - start)
    _write(WARM, result)
    print('native competence', result['held_with_return'], 'gate', passes, flush=True)
    return result


def register():
    rng = np.random.default_rng(1729152086)

    def roster(count):
        return rng.integers(1, 2**31 - 1, size=count).tolist()

    trial_rows = [dict(seed=seed, native=[roster(6) for _ in range(UPDATES)], donor=[roster((3, 2)) for _ in range(UPDATES)],
                       held_native=roster(24), held_donor=roster((3, 6))) for seed in REPLICATES]
    donor = _read(DONOR_LINEAGE)
    if not donor['donor_gate_passed'] or donor['frozen_sha256'] != _hash(DONOR):
        raise ValueError('donor motor adapter requires the complete qualified donor-only checkpoint')
    plan = dict(schema='tt24-joint-pong-1', donor_sha256=_hash(DONOR), warm_train=[roster(10) for _ in range(WARM_ITERS)],
                warm_held=roster(24), streams=trial_rows, modes=list(MODES), native_cap_frames=CAP,
                arm_cap_frames=ARM_CAP, directions=DIRECTIONS, shared_columns=[list(columns) for columns in MODE_COLUMNS],
                residual_gain=ARM_GAIN,
                warm_gate='15 of 24 with a native return and at least four more than stopped',
                fitness_native='terminal point + .43 of elapsed native play fraction',
                fitness_arm='consecutive physical rebound count',
                primary='mean held donor consecutive rebounds after updates 2, 4 and 8',
                retention='count fresh seeds with an actual return before/after shared updates')
    if PLAN.exists():
        if _read(PLAN) != plan:
            raise ValueError('joint training schedule changed')
    else:
        _write(PLAN, plan)
    return plan


def arm_weights(knobs, baseline, basis):
    result = np.repeat(baseline[None], len(knobs), axis=0)
    for index, (first, second) in enumerate(MODE_COLUMNS):
        result[:, :, first] += ARM_GAIN * knobs[:, index, None] * basis[:, 0]
        result[:, :, second] += ARM_GAIN * knobs[:, index, None] * basis[:, 1]
    return result


def trial(worlds, basis, baseline, state, descriptor):
    rng = np.random.default_rng(descriptor['seed'])
    modes = np.repeat(state[None], len(MODES), axis=0)
    held_batches = [world.prepare(seeds, 'held') for world, seeds in zip(worlds, descriptor['held_donor'], strict=True)]

    def arms(readout, batches):
        weights = arm_weights(readout, baseline, basis)
        return np.array([world.play(weights, rosters, ARM_CAP) for world, rosters in zip(worlds, batches, strict=True)])

    before = arcade([state], descriptor['held_native'])[0]
    rows = {'0': arms(modes, held_batches).tolist()}
    updates = []
    for iteration, (native_seeds, arm_seeds) in enumerate(zip(descriptor['native'], descriptor['donor'], strict=True), 1):
        noise = rng.normal(size=(DIRECTIONS, len(MODE_COLUMNS)))
        knobs = np.stack([population(model, noise, .23) for model in modes])
        flat = knobs.reshape(-1, len(MODE_COLUMNS))
        physical = arms(flat, [w.prepare(s, 'held') for w, s in zip(worlds, arm_seeds, strict=True)])
        pong = arcade(flat, native_seeds)
        physical_score = physical[..., 1].mean(axis=(0, 2)).reshape(len(MODES), -1)
        pong_score = native_reward(pong).reshape(len(MODES), -1)
        # Both conditions play both games. Sham rewards permute which side of
        # each paired native perturbation produced its real observed outcome.
        shuffle = rng.choice((-1, 1), DIRECTIONS)
        main = np.stack([preference(values) for values in physical_score])
        auxiliary = np.stack([preference(values) for values in pong_score])
        auxiliary[1] *= shuffle
        modes = np.clip(modes + .16 * np.mean((main + .7 * auxiliary)[..., None] * noise[None], axis=1), -3., 3.)
        updates.append(dict(joint=modes.tolist(), real_pong_scores=pong_score.tolist(),
                            donor_scores=physical_score.tolist(), shuffle=shuffle.tolist(),
                            arm_frames=int(physical[..., 2].sum()), real_pong_frames=int(pong[..., 2].sum())))
        if iteration in (2, 4, 8):
            rows[str(iteration)] = arms(modes, held_batches).tolist()
            print('joint stream', descriptor['seed'], iteration,
                  np.around(np.array(rows[str(iteration)])[..., 1].mean(axis=(0, 2)), 3), flush=True)
    after = arcade(modes, descriptor['held_native'])
    return dict(seed=descriptor['seed'], parameters=modes.tolist(), donor_held=rows, updates=updates,
                native_before=before.tolist(), native_after=after.tolist())


def native_metrics(outcomes):
    score, hits, frames = np.array(outcomes).T
    return dict(points=len(score), own_return=int((hits >= 1).sum()), later=int(np.maximum(hits - 1., 0.).sum()),
                earned=int(((score > 0) & (hits >= 1)).sum()), losses=int((score < 0).sum()), frames=int(frames.sum()))


def summarize(rows):
    curves = np.array([[np.array(row['donor_held'][str(budget)])[..., 1].mean(axis=(0, 2))
                        for budget in (0, 2, 4, 8)] for row in rows])
    cosines = []
    for row in rows:
        rng = np.random.default_rng(row['seed'])
        for update in row['updates']:
            directions = rng.normal(size=(DIRECTIONS, len(MODE_COLUMNS)))
            rng.choice((-1, 1), DIRECTIONS)
            donor = preference(np.asarray(update['donor_scores'][0])) @ directions
            native = preference(np.asarray(update['real_pong_scores'][0])) @ directions
            size = np.linalg.norm(donor) * np.linalg.norm(native)
            if size > 0:
                cosines.append(float((donor @ native) / size))
    source = dict(before=native_metrics(np.concatenate([row['native_before'] for row in rows])))
    for index, mode in enumerate(MODES):
        source[mode] = native_metrics(np.concatenate([row['native_after'][index] for row in rows]))
    return dict(donor_curves=curves.mean(axis=0).T.tolist(), early_paired=(curves[:, 1:, 0] - curves[:, 1:, 1]).mean(axis=1).tolist(),
                native_retention=source, local_alignment=cosines)


def run(plan, warmed):
    native = Native()
    worlds = [World(native, design) for design in DESIGNS]
    try:
        basis, probes = _exploration(worlds)
        with np.load(DONOR, allow_pickle=False) as saved:
            baseline = saved['weights']
        receipt = (_read(OUTPUT) if OUTPUT.exists() else
                   dict(plan=_hash(PLAN), warm=_hash(WARM), basis=basis.tolist(), probes=probes, results=[]))
        if receipt['plan'] != _hash(PLAN) or receipt['warm'] != _hash(WARM):
            raise ValueError('joint study inputs changed')
        for schedule in plan['streams'][len(receipt['results']):]:
            receipt['results'].append(trial(worlds, basis, baseline, np.array(warmed['weights']), schedule))
            _write(OUTPUT, receipt)
        receipt['summary'] = summarize(receipt['results'])
        _write(OUTPUT, receipt)
        print('joint study', receipt['summary'], flush=True)
    finally:
        for world in worlds:
            world.close()
        native.working.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = register()
    result = warm(plan)
    if result['gate'] and not args.warm_only:
        run(plan, result)


if __name__ == '__main__':
    main()
