"""Audit a separable camera-motion term in the frozen wrist velocity observer."""

import argparse
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy, commands
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import BODIES, OUTPUT as SOURCE, InstalledArm
from .tt24_public_body import ConsensusWrist
from .tt24_ready import prepare_step


OUTPUT = ROOT / 'ego-motion-wrist-audit.json.gz'
BETAS = (-.5, 0., .5, 1., 1.5)
OBSERVED_TICKS = 180


class CameraMotion:
    """Change the reported velocity while preserving the original wrist tracker."""

    def __init__(self, sensor):
        self.sensor = sensor
        self.camera = deque(maxlen=4)
        self.terms = np.zeros((3,3))

    def observe(self, observation):
        state = self.sensor.observe(observation)
        body = self.sensor.decoder.body(observation['joints'])
        self.camera.append((body[:3],body[3:12].reshape(3,3)))
        latest_center, latest_rotation = self.camera[-1]
        points = self.sensor.history
        recent_cameras = list(self.camera)[-len(points):]
        for lag in (1,2,3):
            past = max(0,len(points)-1-lag)
            center, rotation = recent_cameras[past]
            local = rotation.T @ (points[past]-center)
            # Both world velocities are in the current axes. Only this term
            # restores the displacement attributable to camera movement.
            delta = (latest_center-center)+(latest_rotation-rotation) @ local
            self.terms[lag-1] = delta*(60./lag)
        return state

    def with_gain(self, state, beta):
        if beta == 1. or not self.terms.any():
            return state
        features = self.sensor.motion_input.copy()
        features[3:] += (beta-1.)*self.terms.ravel()
        velocity = self.sensor.motion.predict(features)
        if isinstance(self.sensor,ConsensusWrist):
            velocity = np.clip(velocity,[-1.8,-1.8,-4.5],[1.8,1.8,4.5])
        updated = state.copy()
        updated[3:6] = velocity
        return updated


def audit(installation, policy, still, row):
    arena = installation.arena
    plan = row['plan']
    past = row['modes']['bundle']
    location, velocity, spin = (np.array(row['launch'][name]) for name in ('position','velocity','spin'))
    lead = plan['prep_ticks']
    changes = dict(plan['changes'])
    obs = installation.reset(plan,location,render=True)
    policy.begin_release(obs)
    still.begin_release(obs)
    move = CameraMotion(policy.sensor)
    frozen = CameraMotion(still.sensor)
    fixed_q = obs['joints'].copy()
    active, stationary, visible_airborne = 0,0,0
    effects, errors, rates, boundaries = [],[],[],[]
    for i, command in enumerate(past['actions']):
        tick = i-lead
        if tick in changes:
            boundaries.append(installation.switch_gravity(tick,changes[tick]))
        if i < OBSERVED_TICKS:
            misses = move.sensor.gaps
            state = move.observe(obs)
            frozen_obs = {'rgb':obs['rgb'],'joints':fixed_q}
            frozen_state = frozen.observe(frozen_obs)
            for beta in BETAS:
                if not np.array_equal(frozen.with_gain(frozen_state,beta),frozen_state):
                    raise ValueError('the camera-motion term changed a fixed-camera output')
            stationary += 1
            claims = [move.with_gain(state,beta) for beta in BETAS]
            original_action = commands(state,policy.weights,policy.issued.copy())
            alternatives = np.array([commands(s,policy.weights,policy.issued.copy()) for s in claims])
            if not np.array_equal(original_action,np.array(command)):
                raise ValueError('full camera compensation changed the portable release commands')
            policy.issued += original_action*(5.2/60.)
            for estimate in claims:
                if not np.array_equal(estimate[np.r_[0:3,6:11]],state[np.r_[0:3,6:11]]):
                    raise ValueError('camera compensation changed non-velocity observation features')
            active += int(move.terms.any())
            effects.append(move.terms[0].tolist())
            rates.append(np.linalg.norm(alternatives-original_action,axis=1).tolist())
            ball_contact = any(c.efc_address >= 0 and arena._ball_geom in (c.geom1,c.geom2) for c in arena._data.contact)
            if tick>=0 and not ball_contact and move.sensor.gaps==misses:
                v = arena._ball_velocity.copy()
                errors.append((np.array([s[3:6] for s in claims])-v).tolist())
                visible_airborne += 1
        render = i+1 < OBSERVED_TICKS
        if tick < 0:
            prepare_step(arena,np.asarray(command),location)
            if tick == -1:
                arena._ball_velocity[:] = velocity
                arena._ball_spin[:] = spin
                mujoco.mj_forward(arena._model,arena._data)
            obs = arena._observation(render)
        else:
            obs,_,done = arena.step(command,render=render)
            if done != (i+1==len(past['actions'])):
                raise ValueError('archived compound trajectory ends at another tick')
    if (digest_json(arena.judge_metrics())!=digest_json(past['metrics']) or
            digest_json(boundaries)!=digest_json(past['changes_reached'])):
        raise ValueError('archived compound replay changed physical evidence')
    return dict(seed=plan['learner_seed'],body=plan['body'],stationary_exact=stationary,
                nonzero_camera_motion=active,visible_airborne=visible_airborne,ego_term=effects,
                action_diff=rates,velocity_error=errors,optics=policy.calibration_index)


def summarize(rows):
    errors = np.asarray([e for row in rows for e in row['velocity_error']])
    changes = np.asarray([v for row in rows for v in row['action_diff']])
    ego = np.asarray([v for row in rows for v in row['ego_term']])
    return dict(stationary_frames_exact=sum(row['stationary_exact'] for row in rows),
                archived_forecasts=len(changes),visible_airborne_frames=len(errors),
                ego_speed_median=float(np.median(np.linalg.norm(ego,axis=1))),
                velocity_rmse={str(b):float(np.sqrt(np.mean(errors[:,i]**2))) for i,b in enumerate(BETAS)},
                action_change_rms={str(b):float(np.sqrt(np.mean(changes[:,i]**2))) for i,b in enumerate(BETAS)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    archive=_read(SOURCE)
    if archive['bundle_sha256']!=_hash(BUNDLE):
        raise ValueError('the original carried bundle is required to audit its archived plays')
    policy,stationary=CarriedWristPolicy(BUNDLE),CarriedWristPolicy(BUNDLE)
    groups={}
    for body in BODIES:
        installation=InstalledArm(body)
        try:
            locked=[r for r in archive['cases'] if r['plan']['body']==body][:3]
            groups[body]=[audit(installation,policy,stationary,row) for row in locked]
            print(body,summarize(groups[body]),flush=True)
        finally:
            installation.arena.close()
    pooled=[row for rows in groups.values() for row in rows]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    _write(args.output,dict(schema='tt24-wrist-ego-motion-audit-1',source_sha256=_hash(SOURCE),
                            bundle_sha256=_hash(BUNDLE),betas=BETAS,exact_archive_replays=len(pooled),
                            results={body:summarize(rows) for body,rows in groups.items()},
                            pooled=summarize(pooled),cases=groups))


if __name__=='__main__':
    main()
