"""Qualify a wrist-only scene calibration with recovered physical rebounds."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt24_bounce import BounceArena, WRIST_CAMERA, mujoco
from .tt24_bounce_ident import WristPupil
from .tt24_public_body import load_public
from .tt24_world_body_play import advance_body
from .tt24_wrist_egomotion import (CALIBRATION, FIT, HEIGHT, HELD, MAP, OUT, PUPIL,
                                 TRIAL, WIDTH, arm_motion, camera_pose, discover_edges,
                                 line_residual, link_pose, reprojected_lines)
from .tt4_transfer_trial import _hash, _read, _write

STOCK_HELD=tuple(range(368201,368217))
DIAGNOSTIC=(368171,368172,368173,368174)
MODES=('unmodified','mount_links','mount','hold','stock_original')


class TransferDecoder:
    """Change coordinates around the unchanged stock optical and motor fits."""

    def __init__(self, optical, decoder, original, fitted):
        self.optical=optical
        self.decoder=decoder
        self.original=original
        self.fitted=fitted
        self.default=optical.default
        self.image_height=getattr(optical,'image_height',HEIGHT)

    def body(self,joints):
        q=np.asarray(joints,dtype=float)[None,:5]
        initial_p,initial_r=link_pose(q,self.decoder.links,self.original[6:])
        fitted_p,fitted_r=link_pose(q,self.decoder.links,self.fitted[6:])
        initial_rotation=initial_r[0]@Rotation.from_rotvec(self.original[:3]).as_matrix()
        fitted_rotation=fitted_r[0]@Rotation.from_rotvec(self.fitted[:3]).as_matrix()
        correction=fitted_rotation@initial_rotation.T
        body=self.optical.body(joints)
        body[:3]+=fitted_p[0]+fitted_r[0]@self.fitted[3:6]-initial_p[0]-initial_r[0]@self.original[3:6]
        body[3:12]=(correction@body[3:12].reshape(3,3)).ravel()
        body[12:15]+=fitted_p[0]-initial_p[0]+(fitted_r[0]-initial_r[0])@self.decoder.offset[3:]
        norm=body[15:17]
        world_normal=np.r_[norm,np.sqrt(max(.02,1.-norm@norm))]
        body[15:17]=(fitted_r[0]@initial_r[0].T@world_normal)[:2]
        return body

    def ball(self,ellipse,body):
        return self.optical.ball(ellipse,body)


def qualification(arena,seed,mode,prior,geometry,models):
    arena_mode='standard' if mode=='stock_original' else 'changed'
    obs=arena.reset(seed=seed,split='held',regime=arena_mode,render=mode!='hold')
    if mode!='hold':
        pupil=WristPupil.load(prior)
        if mode in ('mount','mount_links'):
            pupil.sensor.decoder=TransferDecoder(pupil.sensor.decoder,geometry,models['source_theta'],models[mode])
    actions=[]
    while True:
        action=np.zeros(5) if mode=='hold' else pupil.command(obs)
        actions.append(action.tolist())
        obs,_,done=arena.step(action,render=mode!='hold')
        if done:
            break
    judge=arena.judge_metrics()
    arena.reset(seed=seed,split='held',regime=arena_mode,render=False)
    for action in actions:
        arena.step(action,render=False)
    if arena.judge_metrics()!=judge:
        raise ValueError('replayed joints differ from online rebound judge')
    return dict(seed=seed,judge=judge,actions=actions)


def held_bounces(args):
    geometry=load_public(args.sensor).decoder
    with np.load(args.map,allow_pickle=False) as source:
        models={name:source[name] for name in ('source_theta','mount','mount_links')}
    doc=dict(schema='tt24-wrist-egomotion-bounce-1',model_sha256=_hash(args.map),source_sha256=_hash(args.pupil),
             prior_calibration_sha256=_hash(args.sensor),fit_receipt_sha256=_hash(FIT),
             arena_horizon=args.horizon,alternate_seeds=list(HELD),stock_seeds=list(STOCK_HELD),
             exact_physical_judge_replay=True,conditions={})
    if args.resume and args.out.exists():
        old=_read(args.out)
        if any(old[key]!=value for key,value in doc.items() if key!='conditions'):
            raise ValueError('existing qualification uses different frozen assets or held conditions')
        doc['conditions']=old['conditions']
    for mode in args.modes:
        arm='stock' if mode=='stock_original' else 'alternate'
        arena=BounceArena(view='wrist',width=WIDTH,height=HEIGHT,arm=arm,max_steps=args.horizon)
        try:
            chosen=STOCK_HELD if mode=='stock_original' else HELD
            rows=doc['conditions'].get(mode,[])
            if [r['seed'] for r in rows]!=list(chosen[:len(rows)]):
                raise ValueError('existing runs differ from the registered seed order')
            for seed in chosen[len(rows):]:
                row=qualification(arena,seed,mode,args.pupil,geometry,models)
                rows.append(row)
                print(mode,seed,row['judge']['longest_consecutive_rebounds'],flush=True)
                doc['conditions'][mode]=rows
                _write(args.out,doc)
        finally:
            arena.close()


def geometry_diagnostic(args):
    """Measure the frozen map against hidden poses on four additional test motions."""
    geometry=load_public(args.sensor).decoder
    with np.load(args.map,allow_pickle=False) as source:
        models={name:source[name] for name in ('source_theta','mount','mount_links')}
    arena=BounceArena(view='none',width=WIDTH,height=HEIGHT,arm='alternate')
    cam=arena._data.camera(WRIST_CAMERA)
    rows=[]
    try:
        for seed in DIAGNOSTIC:
            obs=arena.reset(seed=seed,split='held',regime='changed',render=False)
            home=obs['joints'].copy()
            encoders,positions,rotations=[],[],[]
            for tick in range(96):
                if tick%2==0:
                    mujoco.mj_kinematics(arena._model,arena._data)
                    encoders.append(obs['joints'].copy())
                    positions.append(cam.xpos.copy())
                    rotations.append(cam.xmat.reshape(3,3).copy())
                obs=advance_body(arena,arm_motion(seed,tick,home,obs['joints']))
            for mode,model in models.items():
                position,rotation=camera_pose(np.asarray(encoders),geometry,model)
                angle=Rotation.from_matrix(np.swapaxes(rotation,1,2)@rotations).magnitude()
                error=np.linalg.norm(position-positions,axis=1)
                rows.append(dict(seed=seed,mode=mode,position_mm=float(np.sqrt(np.mean(error**2))*1000.),
                                 attitude_deg=float(np.rad2deg(np.sqrt(np.mean(angle**2))))))
    finally:
        arena.close()
    _write(args.out,dict(model_sha256=_hash(args.map),observations_per_clip=48,rows=rows,
                         labels='hidden camera pose diagnostic after freezing fit',seeds=list(DIAGNOSTIC)))
    print(rows,flush=True)


def choose_view(obs, geometry, lines, models):
    edges=discover_edges(obs['rgb'])[None,:,:]
    q=np.asarray(obs['joints'],dtype=float)[None,:]
    scores={}
    for mode in ('source_theta','mount_links'):
        eye,attitude=camera_pose(q,geometry,models[mode])
        delta=line_residual(edges,reprojected_lines(lines,eye,attitude))
        scores[mode]=float(np.mean(delta**2)) if len(delta) else 0.
    return min(scores,key=scores.get),scores


def view_recall(args):
    geometry=load_public(args.sensor).decoder
    with np.load(args.map,allow_pickle=False) as source:
        models={name:source[name] for name in ('source_theta','mount_links')}
        lines=source['points']
    rows=[]
    for arm,regime,sample in [('stock','standard',STOCK_HELD),('alternate','changed',HELD)]:
        arena=BounceArena(view='wrist',width=WIDTH,height=HEIGHT,arm=arm)
        try:
            for seed in sample:
                obs=arena.reset(seed=seed,split='held',regime=regime)
                choice,scores=choose_view(obs,geometry,lines,models)
                rows.append(dict(seed=seed,arm=arm,choice=choice,mean_squared_projection=scores))
        finally:
            arena.close()
    _write(args.out,dict(model_sha256=_hash(args.map),observations_per_drop=1,rows=rows,
                         input='one reset wrist image plus encoder packet'))
    print({arm:sum(r['choice']==expected for r in rows if r['arm']==arm)
           for arm,expected in [('stock','source_theta'),('alternate','mount_links')]},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('bounce','diagnostic','recall'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--horizon',type=int,default=600)
    parser.add_argument('--map',type=Path,default=MAP)
    parser.add_argument('--sensor',type=Path,default=CALIBRATION)
    parser.add_argument('--pupil',type=Path,default=PUPIL)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.out is None:
        args.out=TRIAL if args.task=='bounce' else OUT/f'wrist-egomotion-{args.task}.json.gz'
    {'bounce':held_bounces,'diagnostic':geometry_diagnostic,'recall':view_recall}[args.task](args)
