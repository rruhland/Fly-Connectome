"""Carry local motor feedback from reward-trained solo bouncing into a live match."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_factory import FactoryModel
from .tt19_view_full_game import digest_json
from .tt23_sequence import course, verify
from .tt24_bounce import BounceArena
from .tt24_donor_trial import Native, ROOT, _live
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_rally_bank import HOLDOUT
from .tt24_rally_improve_optical import CACHE, DIMS, ExtentHistory, OpticalPolicy
from .tt24_rally_improve_trial import SCHEDULES
from .tt24_rally_trial import COACHED, STEP_SIZE, Station, returns, summary


SOURCE = ROOT/'tt24_donor-refined.npz'
SOURCE_REPORT = ROOT/'tt24_donor-refined.json.gz'
PLAN = ROOT/'solo-match-plan.json'
OUTPUT = ROOT/'solo-match-trial.json.gz'
MODEL = ROOT/'solo-match-stabilizer.npz'
SOURCE_STATS = ROOT/'solo-match-source.json.gz'
PROGRESS = ROOT/'solo-match-progress.json.gz'
SCALES = np.array((.04, .04, .04, .14, .14))
CAP = .30
TRIAL_GAINS = (.32, .64)
TYPES = ('solo_learned', 'shuffled', 'isotropic')
PRACTICE = {'practice_a': 1572088391, 'practice_b': 1838927104}
HELD = {'qualification_a': 826146727, 'qualification_b': 1741541821}
RETENTION = (591783781, 1590781330, 2120780137, 714139317)
HELD_WORKER = None


def axes(normal, shaft):
    shaft = shaft - normal*(normal@shaft)
    shaft /= np.linalg.norm(shaft)
    return np.column_stack((shaft, np.cross(normal, shaft), normal))


def upright(tilt):
    normal = np.r_[tilt, np.sqrt(max(.001, 1.-tilt@tilt))]
    return normal/np.linalg.norm(normal)


def source_pose(decoder, q):
    x = decoder.body(q)
    center, n = x[12:15], upright(x[15:17])
    _, grip = decoder.links.read(np.asarray(q)[None, :5])
    return center, n, axes(n, -grip[0,:,2])


def normalized_jacobian(linear, angular, basis):
    return np.vstack((basis.T@linear, basis[:,:2].T@angular))/SCALES[:, None]


def estimate_source_jacobian(decoder, q, basis):
    tangent = np.eye(5)*.002
    delta = []
    for d in tangent:
        p, n, _ = source_pose(decoder, q+d)
        a, b, _ = source_pose(decoder, q-d)
        delta.append(np.r_[basis.T@(p-a), basis[:,:2].T@(n-b)]/.004)
    return np.asarray(delta).T/SCALES[:,None]


def head_target(native, w, vector):
    """Read the native goal to convergence with the same public input fixed."""
    vector = np.ascontiguousarray(vector, dtype=float)
    request, issued = np.zeros(5), np.zeros(5)
    # The native goal is bounded at 0.245 radians, versus step cap 0.0564.
    for _ in range(5):
        native.command(vector.ctypes.data, w.ctypes.data, issued.ctypes.data, request.ctypes.data)
    return issued


def sensitivity(native, weights, state, basis):
    """Perturb only the proprioceptive pose, retaining the ball from its relative view."""
    inc = .025
    columns = []
    n = upright(state[9:11])
    for direction in range(5):
        a, b = state.copy(), state.copy()
        effect = inc * SCALES[direction]
        if direction < 3:
            position = effect * basis[:, direction]
            a[:3] += position
            a[6:9] += position
            b[:3] -= position
            b[6:9] -= position
        else:
            rotation = effect * basis[:, direction-3]
            plus, minus = n+rotation, n-rotation
            a[9:11] = (plus/np.linalg.norm(plus))[:2]
            b[9:11] = (minus/np.linalg.norm(minus))[:2]
        columns.append((head_target(native, weights, a)-head_target(native, weights, b))/(2.*inc))
    return np.asarray(columns).T


def extract(native, weights, donor, out):
    arena = BounceArena(view='wrist', width=192, height=144)
    responses = []
    clips = []
    try:
        for item in donor['held_stock'][:4]:
            seed = item['seed']
            tape = item['conditions']['public']
            obs = arena.reset(seed=seed, split='held')
            eye = load_public(FUSED_OUTPUT)
            wanted = len(responses)
            for tick, command in enumerate(tape['actions']):
                if tick < 128:
                    state = eye.observe(obs)
                    if tick % 8 == 0:
                        q = obs['joints']
                        _, _, basis = source_pose(eye.decoder,q)
                        j = estimate_source_jacobian(eye.decoder,q,basis)
                        responses.append(j@sensitivity(native, weights, state, basis))
                obs, _, done = arena.step(command, render=tick+1<128)
                if done:
                    break
            if digest_json(arena.judge_metrics()) != digest_json(tape['metrics']):
                raise ValueError('source replay changed physical outcome')
            clips.append(dict(seed=seed, frames=len(tape['actions']), sampled=len(responses)-wanted))
    finally:
        arena.close()
    fields = np.stack(responses)
    transfer = np.median(fields, axis=0)
    symmetric = .5*(transfer+transfer.T)
    eig, basis = np.linalg.eigh(symmetric)
    stable = (basis*np.minimum(eig,0.))@basis.T
    radius = np.linalg.norm(stable)
    if radius < .02:
        raise ValueError('the donor head has no resolved locally restoring response')
    # The three candidate priors share the same operator's Frobenius norm.
    stable *= min(1., 1./radius)
    orthogonal, _ = np.linalg.qr(np.random.default_rng(656623124).normal(size=(5,5)))
    shuffled = orthogonal@stable@orthogonal.T
    generic = -np.eye(5)*np.linalg.norm(stable)/np.sqrt(5)
    np.savez_compressed(out, solo_learned=stable, shuffled=shuffled, isotropic=generic)
    return dict(clips=clips, samples=len(responses), signed_eigenvalues=eig.tolist(), raw=transfer.tolist(),
                transmitted_frobenius_norm=float(np.linalg.norm(stable)), source_head_weight_sha256=_hash(SOURCE),
                publicly_calibrated_observer_sha256=_hash(FUSED_OUTPUT))


class Residual:
    """Correct a measured tool error about the match head's own issued pose."""

    def __init__(self, match, matrix, gain):
        self.match, self.matrix, self.gain = match, matrix, gain
        self.body = FactoryModel()
        self.perturbed = 0
        self.control_frames = 0
        self.norm = 0.
        self.peak = 0.

    def command(self, state, issued):
        chosen = self.match.command(state, issued)
        q = state[20:25]
        _, normal, j, jn = self.body.pose(q)
        # Link frames are public vendor geometry. The shaft is the last
        # rotation's axial direction, which remains tangent to the paddle face.
        r = self.body.basis.copy()
        for angle, (_, orient) in zip(q, self.body.transforms, strict=True):
            c, s = np.cos(angle), np.sin(angle)
            rz = np.array(((c,-s,0.), (s,c,0.), (0.,0.,1.)))
            r = r@orient@rz
        j = normalized_jacobian(j, jn, axes(normal, -r[:,2]))
        discrepancy = j @ np.clip(q-issued[:5], -.16, .16)
        correction = self.matrix @ discrepancy
        goal = np.linalg.solve(j.T@j+.35*np.eye(5), j.T@correction)
        rate = goal/STEP_SIZE[:5]
        rate *= self.gain*min(1., CAP/max(1.e-10,np.linalg.norm(rate)))
        action = chosen.copy()
        action[:5] = np.clip(action[:5]+rate, -1., 1.)
        effort = np.linalg.norm(action[:5]-chosen[:5])
        self.norm += float(effort)
        self.perturbed += int(effort>1.e-6)
        self.control_frames += 1
        self.peak = max(self.peak, float(effort))
        return action


def plan():
    practice = course(PRACTICE)[:4]
    held = course(HELD)
    launched = {row['seed'] for row in practice+held}
    known = {row['seed'] for row in course(HOLDOUT)}
    known.update(row['seed'] for g in _read(SCHEDULES)['schedules'].values() for row in g)
    known.update(data['seed'] for data in _read(SOURCE_REPORT)['held_stock'])
    with np.load(CACHE, allow_pickle=False) as archive:
        known.update(map(int, archive['seed']))
    if len(launched)!=len(practice+held) or launched&known or launched.intersection(RETENTION):
        raise ValueError('source, fit, and held schedules must be disjoint')
    return dict(schema='tt24-solo-match-plan-1', practice=practice, held=held,
                retention=list(RETENTION), gain_options=list(TRIAL_GAINS), relative_motor_bound=CAP,
                source_model_sha256=_hash(SOURCE), match_motor_data_sha256=_hash(CACHE),
                practice_cost_per_residual=8, shared_frozen_scores=4,
                primary='near legal returns with a legal far reply in the verified live two-paddle game')


def receive(station, launch, match, sphere, matrix=None, gain=0.):
    actor = Residual(match,matrix,gain) if matrix is not None and gain else match
    row = station.play(launch, 'participant', actor, lambda: ExtentHistory(sphere), live_opening=True)
    if isinstance(actor, Residual):
        motor = dict(perturbed_frames=actor.perturbed,
                     average_correction=actor.norm/max(1,actor.control_frames), peak_correction=actor.peak)
    else:
        motor = dict(perturbed_frames=0, average_correction=0., peak_correction=0.)
    row['residual'] = motor
    return row


def skill(row):
    return min(len(returns(row,1)),len(returns(row,-1)))


def utility(rows):
    return float(np.mean([min(8,skill(row)) for row in rows]))


def practice(arena, match, sphere, priors, schedule, checkpoint):
    points = checkpoint['practice']
    for scene in schedule[len(points):]:
        cases = {'frozen':receive(arena, scene, match, sphere)}
        for name in TYPES:
            for gain in TRIAL_GAINS:
                cases[f'{name}_{gain}'] = receive(arena, scene, match, sphere, priors[name], gain)
        point = dict(scene, results=cases)
        checkpoint['practice_verified'] += verify([point])
        points.append(point)
        _write(PROGRESS, checkpoint)
        print('practice seed', scene['seed'], {key:skill(row) for key,row in cases.items()}, flush=True)
    evidence = dict(frozen=[point['results']['frozen'] for point in points])
    selected = {}
    for name in TYPES:
        observations = [[point['results'][f'{name}_{value}'] for point in points] for value in TRIAL_GAINS]
        measured = [utility(a) for a in [evidence['frozen'], *observations]]
        win = int(np.argmax(measured))
        selected[name] = (0.,*TRIAL_GAINS)[win]
        evidence[name] = dict(curve=measured, gain=selected[name],
                              trajectories={str(a):b for a,b in zip(TRIAL_GAINS,observations,strict=True)})
        print('practice',name,measured,'selected',selected[name],flush=True)
    return selected, evidence


def held_point(arena, match, sphere, priors, selected, launch):
    # Even if score fits keep zero, both fixed small corrections remain in
    # the sealed held test to compare their direct physical effect.
    modes = {'frozen': (None,0.), 'solo_learned_fixed': ('solo_learned', TRIAL_GAINS[0]),
             'shuffled_fixed': ('shuffled', TRIAL_GAINS[0])}
    modes |= {name: (name,gain) for name,gain in selected.items()}
    completed = {}
    linked = {}
    for label,(name,gain) in modes.items():
        key = (name,gain) if gain else (None,0.)
        if key not in linked:
            linked[key] = receive(arena, launch, match, sphere, priors.get(name), gain)
        completed[label] = linked[key]
    if len({run['prefix_rgb'] for run in completed.values()})!=1:
        raise ValueError('paired policies saw different initial images')
    distinct = {str(key):row for key,row in linked.items()}
    return dict(launch,results=completed), verify([dict(launch,results=distinct)])


def initialize_held(opponent, priors, selected):
    global HELD_WORKER
    with np.load(CACHE, allow_pickle=False) as data:
        match = OpticalPolicy(data, 'interaction')
        sphere = data['sphere'].copy()
    HELD_WORKER = Station(opponent, render_dims=DIMS), match, sphere, priors, selected


def dispatch_held(launch):
    return held_point(*HELD_WORKER, launch)


def evaluation(arena, match, sphere, priors, schedule, selected, checkpoint, opponent, workers):
    receipt = checkpoint['held']
    chronological = {row['seed']:i for i,row in enumerate(schedule)}

    def publish(result):
        point, checked = result
        checkpoint['held_verified'] += checked
        receipt.append(point)
        receipt.sort(key=lambda row:chronological[row['seed']])
        _write(PROGRESS, checkpoint)
        print('held', point['seed'], {name:skill(row) for name,row in point['results'].items()},flush=True)

    recorded = {row['seed'] for row in receipt}
    todo = [row for row in schedule if row['seed'] not in recorded]
    if workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                                 initializer=initialize_held, initargs=(opponent,priors,selected)) as pool:
            jobs = [pool.submit(dispatch_held,launch) for launch in todo]
            for result in as_completed(jobs):
                publish(result.result())
    else:
        for launch in todo:
            publish(held_point(arena, match, sphere, priors, selected, launch))
    return receipt


def retention(native, weights):
    arena = BounceArena(view='wrist', width=192, height=144)
    rows = []
    try:
        for seed in RETENTION:
            row = _live(native, arena, weights, seed, 'standard', 'public')
            audit = _live(native, arena, weights, seed, 'standard', 'replay', tape=row['actions'])
            if row!=audit:
                raise ValueError('retained source tape changed')
            rows.append(dict(seed=seed,**row))
    finally:
        arena.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--held-workers', type=int, default=1)
    args = parser.parse_args()
    requirements = plan()
    if PLAN.exists() and _read(PLAN)!=requirements:
        raise ValueError('registration differs')
    if not PLAN.exists():
        _write(PLAN, requirements)
    donor = _read(SOURCE_REPORT)
    if _hash(SOURCE)!=donor['frozen_sha256']:
        raise ValueError('the independently scored source head changed')
    with np.load(SOURCE, allow_pickle=False) as saved:
        weights = np.ascontiguousarray(saved['weights'])
    native = Native()
    receiver = None
    try:
        if SOURCE_STATS.exists() and MODEL.exists():
            source = _read(SOURCE_STATS)
            if (source['source_head_weight_sha256']!=_hash(SOURCE)
                    or source['publicly_calibrated_observer_sha256']!=_hash(FUSED_OUTPUT)
                    or source['model_sha256']!=_hash(MODEL)):
                raise ValueError('the source extraction digest differs')
        else:
            source = extract(native, weights, donor, MODEL)
            source['model_sha256'] = _hash(MODEL)
            _write(SOURCE_STATS,source)
        print('frozen stabilizer extracted',source,flush=True)
        if args.extract_only:
            return
        with np.load(MODEL, allow_pickle=False) as f:
            candidates = {key:f[key] for key in f.files}
        with np.load(CACHE, allow_pickle=False) as d:
            match = OpticalPolicy(d,'interaction')
            sphere = d['sphere'].copy()
        opponent = _read(COACHED)['metadata']['opening']['far_profile']
        receiver = Station(opponent,render_dims=DIMS)
        marker = dict(plan_sha256=_hash(PLAN),model_sha256=_hash(MODEL))
        checkpoint = (_read(PROGRESS) if PROGRESS.exists() else
                      dict(marker=marker, practice=[], held=[], practice_verified=0, held_verified=0))
        before = [r['seed'] for r in checkpoint['practice']]
        complete = {r['seed'] for r in checkpoint['held']}
        if (checkpoint['marker']!=marker or before!=[r['seed'] for r in requirements['practice'][:len(before)]]
                or len(complete)!=len(checkpoint['held']) or complete-{r['seed'] for r in requirements['held']}):
            raise ValueError('incomplete replay data uses another schedule or controller')
        triplet, practices = practice(receiver, match, sphere, candidates, requirements['practice'],checkpoint)
        held = evaluation(receiver, match, sphere, candidates, requirements['held'],triplet,checkpoint,
                          opponent,args.held_workers)
        output = dict(schema='tt24-solo-match-receipt-1', source=source, selected=triplet, practice=practices,
                      practice_verified=checkpoint['practice_verified'], held_verified=checkpoint['held_verified'],
                      cases=held, plan_sha256=_hash(PLAN),
                      donor_sha256_after=_hash(SOURCE), model_sha256=_hash(MODEL),
                      held_summary={key:summary(held,key,0)|dict(legal_exchanges=sum(skill(row['results'][key])
                                                                                 for row in held))
                                    for key in held[0]['results']})
        _write(OUTPUT, output)
        output['source_retention'] = retention(native, weights)
        _write(OUTPUT, output)
    finally:
        if receiver is not None:
            receiver.arena.close()
        native.working.cleanup()


if __name__ == '__main__':
    main()
