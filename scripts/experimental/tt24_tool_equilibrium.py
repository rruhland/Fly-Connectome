"""Transport a familiar optical paddle pose into a new arm's equilibrium once per drop."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm, episode
from .tt24_metric_stereo import calibrated_cameras
from .tt24_public_body import PublicDecoder, stock
from .tt24_public_morph import MorphDecoder, OpticalLinks, ProbeEyes
from .tt24_ready_geometry import ball_position


OUTPUT = ROOT / 'tool-equilibrium.json.gz'
LOCAL_OUTPUT = ROOT / 'tool-equilibrium-local-workspace.json.gz'
RIG = ROOT / 'public-body-fused.json.gz'
MODES = ('bundle', 'aligned', 'reverse', 'workspace')
COURSE = (('alternate', 12), ('stock', 8))
SEED = 599374469
CAP = 360
SCALE = np.array([.03,.03,.03,.15,.15])


def paddle(decoder, posture):
    return decoder.body(posture)[12:17]


def bounded_xy(query, center, offset, limit):
    delta = query[:2]-center[:2]-offset[:2]
    return delta*min(1.,limit/max(1.e-9,np.linalg.norm(delta)))


def transport(decoder, posture, target, *, damping=0.):
    def residual(q):
        error = (paddle(decoder, q)-target)/SCALE
        return np.r_[error, damping*(q-posture)/.24]

    result = least_squares(residual, posture, bounds=(posture-.24, posture+.24), max_nfev=35)
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('could not register the nominal pose inside the local joint range')
    return result.x


def registration():
    arenas = (stock(), ProbeEyes())
    try:
        rigs = [calibrated_cameras(arena) for arena in arenas]
    finally:
        for arena in arenas:
            arena.close()
    comparisons = []
    for a,b in zip(*rigs, strict=True):
        aa,bb = asdict(a),asdict(b)
        errors = {k:float(np.max(abs(np.asarray(aa[k])-np.asarray(bb[k])))) for k in aa}
        if max(errors.values()) > 1e-6:
            raise ValueError('the public pose fits do not share the same static stereo registration')
        comparisons.append(errors)
    return comparisons


class ToolEquilibrium(CarriedWristPolicy):
    """Keep the learned motor readout and transport its reference only."""

    def __init__(self, path, workspace_limit):
        super().__init__(path)
        self.workspace_limit = workspace_limit
        self.mode = 'bundle'
        self.reference = None
        self.tick = 0
        self.observed = []
        self.query = None
        self.switch_tick = None
        first, second = self.calibration
        keys = ('offset','normal','registration','default')
        self.stock_decoder = PublicDecoder(*(first[k] for k in keys))
        self.altered_decoder = MorphDecoder(OpticalLinks(second['shifts'],second['tilts']),
                                            *(second[k] for k in keys))
        self.world_pose = paddle(self.stock_decoder, self.equilibrium)
        self.aligned = transport(self.altered_decoder, self.equilibrium, self.world_pose)

    def reference_at(self, candidate):
        self.issued += self.reference-candidate
        self.reference = candidate.copy()
        self.sensor.home = paddle(self.sensor.decoder, candidate)[:3].copy()

    def begin_release(self, obs):
        super().begin_release(obs)
        q = np.asarray(obs['joints'])
        self.reference = q+self.reuse*(self.equilibrium-q)
        self.tick = 0
        self.observed.clear()
        self.query = None
        self.switch_tick = None
        if self.calibration_index and self.mode in ('aligned','reverse','workspace'):
            signed = -1. if self.mode == 'reverse' else 1.
            self.reference_at(self.reference+signed*self.reuse*(self.aligned-self.equilibrium))

    def translation(self, center):
        return bounded_xy(self.query,center,self.stock_decoder.default,self.workspace_limit)

    def act(self, obs):
        if self.mode in ('workspace','workspace_local') and self.tick < 10 and self.query is None:
            found = ball_position(self.sensor.decoder, obs)
            if found is not None:
                if self.observed and np.linalg.norm(self.observed[-1]-found) > .05:
                    self.observed.clear()
                self.observed.append(found)
                if len(self.observed) == 3:
                    self.query = np.median(self.observed, axis=0)
                    self.switch_tick = self.tick
                    desired = (paddle(self.sensor.decoder,self.reference) if self.mode == 'workspace_local'
                               else self.world_pose.copy())
                    desired[:2] += self.translation(desired)
                    fitted = transport(self.sensor.decoder, self.reference, desired, damping=.075)
                    self.reference_at(fitted)
        self.tick += 1
        return super().act(obs)

    def receipt(self):
        return dict(q=self.reference.tolist(), pose=paddle(self.sensor.decoder,self.reference).tolist(),
                    visible=None if self.query is None else self.query.tolist(), switch=self.switch_tick)


def groups(cases, modes=MODES):
    scores = {mode:np.array([row['modes'][mode]['metrics']['longest_consecutive_rebounds'] for row in cases])
              for mode in modes}
    return {mode:dict(mean=float(values.mean()), max=int(values.max()), ten=int((values>=10).sum()),
                      twenty=int((values>=20).sum()), wins_vs_bundle=int((values>scores['bundle']).sum()),
                      losses_vs_bundle=int((values<scores['bundle']).sum()), streaks=values.tolist())
            for mode,values in scores.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--bundle',type=Path,default=BUNDLE)
    parser.add_argument('--local-workspace',action='store_true',
                        help='keep the recalled body height and normal on a fresh altered cohort')
    args = parser.parse_args()
    course = (('alternate',12),) if args.local_workspace else COURSE
    modes = ('bundle','workspace_local') if args.local_workspace else MODES
    seed = SEED+101 if args.local_workspace else SEED
    output = args.output or (LOCAL_OUTPUT if args.local_workspace else OUTPUT)
    start = perf_counter()
    camera_agreement = registration()
    limit = .75*_read(RIG)['body']['radius_m']
    controller = ToolEquilibrium(args.bundle,limit)
    q, p = controller.equilibrium, controller.world_pose
    other = paddle(controller.altered_decoder,q)
    aligned = paddle(controller.altered_decoder,controller.aligned)
    calibration = dict(rig_errors=camera_agreement, known_stock_q=q.tolist(), known_stock_pose=p.tolist(),
                       alternate_same_q_pose=other.tolist(), transported_q=controller.aligned.tolist(),
                       transported_pose=aligned.tolist(), workspace_cap_m=limit,
                       expected_launcher_xyz={name:(pose[:3]+decoder.default).tolist() for name,pose,decoder in
                                              [('stock',p,controller.stock_decoder),
                                               ('alternate',other,controller.altered_decoder)]})
    print('public pose check', calibration,flush=True)
    rng = np.random.default_rng(seed)
    plans = [dict(body=body,learner_seed=int(rng.integers(1,2**31-1)),
                  launcher_seed=int(rng.integers(1,2**31-1)),prep_ticks=int(rng.integers(14,39)),
                  changes=[],mass_ratio=1.,drag_ratio=1.) for body,size in course for _ in range(size)]
    report = dict(schema='tt24-tool-equilibrium-1', seed=seed,bundle_sha256=_hash(args.bundle),
                  geometry=calibration,schedule=plans,cap_ticks=CAP,cases=[],verified=0,identity_controls=0,
                  modes=modes)
    output.parent.mkdir(parents=True,exist_ok=True)
    _write(output,report)
    stock_launcher = BounceArena(view='none')
    try:
        for body,_ in course:
            plant = InstalledArm(body)
            plant.arena.max_steps = CAP
            try:
                for plan in (x for x in plans if x['body']==body):
                    stock_launcher.reset(seed=plan['launcher_seed'],split='held',render=False)
                    launch = tuple(v.copy() for v in (stock_launcher._ball_position,stock_launcher._ball_velocity,
                                                      stock_launcher._ball_spin))
                    rows = {}
                    for mode in modes:
                        if mode in ('aligned','reverse') and rows['bundle']['model_index']==0:
                            # Identity transport on the same public model shares the exact verified tape.
                            rows[mode] = rows['bundle'] | dict(shared_bundle_tape=True)
                            report['identity_controls'] += 1
                            continue
                        controller.mode = mode
                        row = episode(plant,controller,plan,launch,'bundle')
                        row['reference'] = controller.receipt()
                        replay = episode(plant,controller,plan,launch,'bundle',replay=row['actions'])
                        if digest_json(row['metrics'])!=digest_json(replay['metrics']):
                            raise ValueError('changed referee outcome in exact physical replay')
                        rows[mode] = row
                        report['verified'] += 1
                    report['cases'].append(dict(plan=plan,release_xyz=launch[0].tolist(),
                                                release_velocity=launch[1].tolist(),release_spin=launch[2].tolist(),
                                                modes=rows))
                    _write(output,report)
                    print(body,plan['learner_seed'],[x['metrics']['longest_consecutive_rebounds'] for x in rows.values()],
                          flush=True)
            finally:
                plant.arena.close()
    finally:
        stock_launcher.close()
    if _hash(args.bundle)!=report['bundle_sha256']:
        raise ValueError('the carried parameters changed during the qualification')
    report['summary'] = {body:groups([row for row in report['cases'] if row['plan']['body']==body],modes)
                         for body,_ in course}
    report['wall_seconds'] = perf_counter()-start
    _write(output,report)
    print(report['summary'],flush=True)


if __name__ == '__main__':
    main()
