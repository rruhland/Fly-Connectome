"""Select one existing donor feedback head using live source wrist rewards."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from .tt19_so101 import WRIST_CAMERA, _GRASP_POS
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena, mujoco
from .tt24_broad_donors import NOVEL, TRAIN, FamilyArena
from .tt24_donor_assess import ARCHIVE, CHECKPOINT
from .tt24_donor_trial import DESIGNS, ROOT, DonorBounce, Native, _checkpoint
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_universal_donor_images import KnownSourceRig


TAG = 'tt24_public_donor_select'
PLAN = ROOT / f'{TAG}-plan.json'
SCREEN = ROOT / f'{TAG}-source.json.gz'
REPORT = ROOT / f'{TAG}.json.gz'
WEIGHTS = ROOT / f'{TAG}.npz'
BROAD = ROOT / 'broad-donors.npz'
SCREEN_CAP, HELD_CAP = 240, 300
SUSTAINED = 8
SOURCE_NAMES = ('light_donor', 'medium_donor', 'long_link_donor')
MODES = ('original', 'winner')


class EncoderRig(KnownSourceRig):
    """Supplied source body geometry, evaluated in a separate encoder-only buffer."""

    def __init__(self, original, arena):
        super().__init__(original)
        self.model = arena._model
        self.shadow = mujoco.MjData(self.model)
        self.joints = arena._joint_position[:5]
        self.blade = arena._blade
        self.grip = arena._grip
        self.paddle = arena._paddle

    def set_pose(self, joints):
        self.shadow.qpos[self.joints] = joints
        mujoco.mj_kinematics(self.model, self.shadow)
        grip_rotation = self.shadow.xmat[self.grip].reshape(3, 3)
        self.shadow.qpos[self.paddle:self.paddle + 3] = self.shadow.xpos[self.grip] + grip_rotation @ _GRASP_POS
        self.shadow.qpos[self.paddle + 3:self.paddle + 7] = self.shadow.xquat[self.grip]
        mujoco.mj_kinematics(self.model, self.shadow)
        mujoco.mj_camlight(self.model, self.shadow)
        camera = self.shadow.camera(WRIST_CAMERA)
        normal = self.shadow.geom_xmat[self.blade].reshape(3, 3)[:, 2]
        self.pose[:] = np.r_[camera.xpos, camera.xmat, self.shadow.geom_xpos[self.blade],
                            normal[:2] if normal[2] >= 0. else -normal[:2]]


def candidates():
    with np.load(CHECKPOINT, allow_pickle=False) as saved:
        base = saved['weights']
    with np.load(ARCHIVE, allow_pickle=False) as saved:
        archive = dict(zip(saved['generations'], saved['weights'], strict=True))
    if not np.array_equal(base, archive[36]):
        raise ValueError('latest archived original donor differs from anchor')
    with np.load(BROAD, allow_pickle=False) as archive_broad:
        broad = archive_broad['weights']
    centre = np.mean([archive[g] for g in (21, 25, 26, 34, 36)], axis=0)
    return dict(original=base, donor_34=archive[34], donor_26=archive[26], late_centroid=centre,
                late_extrapolated=base + .45 * (base - centre), broad_blend=.65 * base + .35 * broad)


def register():
    rng = np.random.default_rng(1538360350)
    draws = iter((rng.choice(2**31-2, 28, replace=False) + 1).tolist())
    source = {domain: [next(draws) for _ in range(2)] for domain in SOURCE_NAMES}
    hold = {domain: [next(draws) for _ in range(8 if domain in ('stock', 'changed') else 2)]
            for domain in ('stock', 'changed', *SOURCE_NAMES)}
    pool = candidates()
    plan = dict(schema='tt24-optical-donor-select-1', source_order=list(SOURCE_NAMES), source=source, held=hold,
                hashes={p.name: _hash(p) for p in (CHECKPOINT, ARCHIVE, BROAD, FUSED_OUTPUT)},
                candidates={name: w.tolist() for name, w in pool.items()}, source_horizon=SCREEN_CAP,
                held_horizon=HELD_CAP, sustained=SUSTAINED,
                ranking='0.50 mean sustained fraction + 0.30 worst donor sustained fraction + 0.20 mean capped streak fraction',
                source_geometry='camera and nominal welded blade reconstructed from supplied source CAD and encoders',
                held_geometry='fixed stock optical calibration on both receiving bodies', zero_stock_motor_practice=True)
    if PLAN.exists():
        if _read(PLAN) != plan:
            raise ValueError('optical source protocol changed')
    else:
        _write(PLAN, plan)
    return plan


def body(domain, cap):
    if domain == 'stock':
        return BounceArena(view='wrist', width=192, height=144, max_steps=cap)
    if domain in ('long_link_donor', 'changed'):
        arena = FamilyArena(TRAIN[4] if domain == 'long_link_donor' else NOVEL, view='wrist')
        arena.max_steps = cap
        return arena
    index = SOURCE_NAMES.index(domain)
    arena = DonorBounce(DESIGNS[index], view='wrist', width=192, height=144, max_steps=cap)
    arena._model.opt.gravity[2] *= (.88, 1.11)[index]
    return arena


def play(arena, native, weights, seed, domain):
    regime = 'standard' if domain == 'stock' else 'changed'
    reset = dict(seed=seed, split='held', regime=regime)
    obs = arena.reset(**reset)
    sensor = load_public(FUSED_OUTPUT)
    rig = None
    if domain in SOURCE_NAMES:
        rig = EncoderRig(sensor.decoder, arena)
        sensor.decoder = rig
    issued, command = np.zeros(5), np.zeros(5)
    tape = []
    while True:
        if rig is not None:
            rig.set_pose(obs['joints'])
        state = np.ascontiguousarray(sensor.observe(obs), dtype=np.float64)
        native.command(state.ctypes.data, weights.ctypes.data, issued.ctypes.data, command.ctypes.data)
        tape.append(command.tolist())
        obs, _, done = arena.step(command)
        if done:
            break
    judge = arena.judge_metrics()
    arena.reset(**reset, render=False)
    for act in tape:
        arena.step(act, render=False)
    if arena.judge_metrics() != judge:
        raise ValueError(f'original referee replay changed on {domain}, seed {seed}')
    return dict(metrics=judge, actions=tape)


def analyze(rows):
    streak = [row['metrics']['longest_consecutive_rebounds'] for row in rows]
    return dict(streaks=streak, mean=float(np.mean(streak)), sustained=sum(x >= SUSTAINED for x in streak),
                at_least_ten=sum(x >= 10 for x in streak), drops=len(streak),
                qualified=sum(row['metrics']['rebounds'] for row in rows), max=max(streak))


def evaluate(task):
    domain, seeds, cap, policies = task
    arena = body(domain, cap)
    native = Native()
    try:
        rows = {}
        played = {}
        for mode, weights in policies.items():
            array = np.ascontiguousarray(weights, dtype=np.float64)
            key = array.tobytes()
            if key not in played:
                played[key] = [play(arena, native, array, seed, domain) for seed in seeds]
            rows[mode] = played[key]
        report = dict(rows=rows, summary={name: analyze(value) for name, value in rows.items()},
                      independent_rollouts=len(played) * len(seeds))
        print(domain, {n: v['streaks'] for n, v in report['summary'].items()}, flush=True)
        return domain, report
    finally:
        arena.close()
        native.working.cleanup()


def utility(by_body):
    frame = np.asarray(by_body)
    fractions = (frame >= SUSTAINED).mean(axis=1)
    return float(.50 * fractions.mean() + .30 * fractions.min() + .20 * (np.minimum(frame, SUSTAINED)/SUSTAINED).mean())


def select(plan, workers):
    receipt = (_read(SCREEN) if SCREEN.exists() else dict(plan=_hash(PLAN), bodies={}))
    if receipt['plan'] != _hash(PLAN):
        raise ValueError('source policy selection lineage changed')
    options = plan['candidates']
    tasks = [(domain, plan['source'][domain], SCREEN_CAP, options) for domain in SOURCE_NAMES
             if domain not in receipt['bodies']]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for domain, rows in pool.map(evaluate, tasks):
            receipt['bodies'][domain] = rows
            _write(SCREEN, receipt)
    ranking = {name: utility([receipt['bodies'][body]['summary'][name]['streaks'] for body in SOURCE_NAMES])
               for name in options}
    winner = max(ranking, key=ranking.get)
    _checkpoint(WEIGHTS, np.array(options[winner]))
    receipt.update(ranking=ranking, winner=winner, winner_sha256=_hash(WEIGHTS))
    _write(SCREEN, receipt)
    print('fixed source outcome', ranking, winner, flush=True)
    return receipt


def held(plan, source, workers):
    output = (_read(REPORT) if REPORT.exists() else dict(plan=_hash(PLAN), source=_hash(SCREEN), held={}))
    if output['plan'] != _hash(PLAN) or output['source'] != _hash(SCREEN):
        raise ValueError('held outcome does not match locked source winner')
    if _hash(WEIGHTS) != source['winner_sha256']:
        raise ValueError('selected weights changed before held runs')
    with np.load(WEIGHTS, allow_pickle=False) as saved:
        frozen = saved['weights'].tolist()
    policies = dict(original=plan['candidates']['original'], winner=frozen)
    tasks = [(domain, seeds, HELD_CAP, policies) for domain, seeds in plan['held'].items() if domain not in output['held']]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for domain, rows in pool.map(evaluate, tasks):
            output['held'][domain] = rows
            _write(REPORT, output)
    if _hash(WEIGHTS) != source['winner_sha256']:
        raise ValueError('held play changed the selected weights')
    output['held_summary'] = {domain: row['summary'] for domain, row in output['held'].items()}
    _write(REPORT, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    plan = register()
    source = select(plan, args.workers)
    if not args.source_only:
        held(plan, source, args.workers)


if __name__ == '__main__':
    main()
