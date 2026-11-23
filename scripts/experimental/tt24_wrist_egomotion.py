"""Calibrate a relocated wrist eye from its view of stationary scene edges."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt24_bounce import BounceArena
from .tt24_public_body import FOCAL, load_public
from .tt24_world_body_play import advance_body
from .tt4_transfer_trial import _hash, _write

OUT = Path('artifacts/tt24')
CALIBRATION = OUT/'public-body-fused.npz'
PUPIL = OUT/'bounce-ident-pupil.npz'
MAP = OUT/'wrist-egomotion-map.npz'
CLIPS = OUT/'wrist-egomotion-clips.npz'
FIT = OUT/'wrist-egomotion-fit.json.gz'
TRIAL = OUT/'wrist-egomotion-trial.json.gz'
PROBES = (368061, 368062, 368063)
HELD = tuple(range(368101, 368117))
WIDTH, HEIGHT = 192,144
SAMPLE = 2
CLIP_TICKS = 96
EDGES = ('table_left', 'back_left', 'back_right', 'table_right')


def robust_line(x, y, tolerance=.8, minimum=6):
    if len(x)<minimum:
        return None
    first, last = np.triu_indices(len(x), k=1)
    spread = x[last]-x[first]
    good = spread >= max(2.,.20*np.ptp(x))
    if not np.any(good):
        return None
    first,last,spread = first[good],last[good],spread[good]
    slope = (y[last]-y[first])/spread
    offset = y[first]-slope*x[first]
    distance = np.abs(y[None,:]-slope[:,None]*x-offset[:,None])/np.hypot(slope[:,None],1.)
    counts = (distance<=tolerance).sum(axis=1)
    select = counts.argmax()
    include = distance[select]<=tolerance
    if counts[select]<minimum:
        return None
    return np.linalg.lstsq(np.column_stack((x[include],np.ones(include.sum()))),y[include],rcond=None)[0]


def pixel_line(fit):
    # Image x and y coordinates, y facing up and origin at the pixel centre.
    a,b = fit
    line = np.array((a,1.,a*(WIDTH-1)/2+b-(HEIGHT-1)/2))
    # y_image=(H-1)/2-y_pixel, hence a*x_image+y_image+c=0.
    return line/np.hypot(*line[:2])


def discover_edges(rgb):
    r,g,b = np.asarray(rgb,dtype=float).transpose(2,0,1)
    blue = (g>1.9*r)&(g>34.)&(b>.99*g)&(b<1.40*g)
    dark = (r<20.)&(g<26.)&(b<31.)
    x = np.arange(rgb.shape[1],dtype=float)
    y_blue,y_dark = np.argmax(blue,axis=0),np.argmax(dark,axis=0)
    valid = blue.any(axis=0)
    table_left = robust_line(x[valid&(x<53)],y_blue[valid&(x<53)])
    back_left = robust_line(x[(x<81)&(y_dark>6)],y_dark[(x<81)&(y_dark>6)])
    lines = np.full((4,3),np.nan)
    if table_left is not None:
        lines[0] = pixel_line(table_left)
        below = y_blue-np.polyval(table_left,x)
        support = valid & (x>52) & (x<132) & (below>1.4) & (below<24.)
        table_right = robust_line(x[support],y_blue[support],minimum=4)
        if table_right is not None and table_right[0]-table_left[0]>.23:
            cross = (table_left[1]-table_right[1])/(table_right[0]-table_left[0])
            if 51.<cross<126.:
                lines[3] = pixel_line(table_right)
    if back_left is not None:
        lines[1] = pixel_line(back_left)
        under = y_dark-np.polyval(back_left,x)
        support = (x>91)&(x<160)&(under>1.6)&(under<21.)
        back_right = robust_line(x[support],y_dark[support],minimum=9)
        if back_right is not None and back_right[0]-back_left[0]>.08:
            cross = (back_left[1]-back_right[1])/(back_right[0]-back_left[0])
            if 86.<cross<154.:
                lines[2] = pixel_line(back_right)
    return lines


def arm_motion(seed,tick,home,joints):
    rng = np.random.default_rng(seed)
    direction = rng.choice((-1.,1.),size=5)
    phase = rng.uniform(-np.pi,np.pi,5)
    frequency = rng.uniform(.55,1.30,5)
    magnitude = np.array((.22,.18,.19,.19,.15))
    q = home + np.minimum(1.,tick/14.)*magnitude*direction*np.sin(2*np.pi*frequency*tick/60.+phase)
    return np.clip(4.0*(q-joints),-.4,.4)


def acquire(path):
    outputs={}
    for arm in ('stock','alternate'):
        arena=BounceArena(view='wrist',width=WIDTH,height=HEIGHT,arm=arm)
        arena._model.geom_contype[arena._ball_geom]=0
        arena._model.geom_conaffinity[arena._ball_geom]=0
        try:
            regime='standard' if arm=='stock' else 'changed'
            encoder,features,tapes,views = [],[],[],[]
            for seed in PROBES:
                obs=arena.reset(seed=seed,split='practice',regime=regime,render=False)
                home=obs['joints'].copy()
                row=[]
                for tick in range(CLIP_TICKS):
                    if tick%SAMPLE==0:
                        image=arena._observation(True)['rgb']
                        row.append((obs['joints'].copy(),discover_edges(image),image.copy()))
                    action=arm_motion(seed,tick,home,obs['joints'])
                    obs=advance_body(arena,action)
                    tapes.append(action)
                q,feature,image=zip(*row,strict=True)
                encoder.append(q)
                features.append(feature)
                views.append(image)
                print('wrist edge clips',arm,seed,np.isfinite(feature).all(axis=2).sum(0),flush=True)
            outputs.update({f'{arm}_{key}':np.asarray(value) for key,value in
                            (('q',encoder),('edges',features),('commands',tapes),('images',views))})
        finally:
            arena.close()
    np.savez_compressed(path,**outputs)


def infer_line(cameras,attitudes,image_lines):
    """Intersect stock-view ray planes to reconstruct one stationary edge."""
    valid=np.isfinite(image_lines).all(axis=1)
    local = image_lines[valid]*[FOCAL,FOCAL,-1.]
    planes = np.einsum('nij,nj->ni',attitudes[valid],local)
    planes /= np.linalg.norm(planes,axis=1,keepdims=True)
    value = np.sum(planes*cameras[valid],axis=1)
    _,_,vectors = np.linalg.svd(planes,full_matrices=False)
    direction = vectors[-1]
    position=np.linalg.lstsq(np.vstack((planes,direction)),np.r_[value,0.],rcond=None)[0]
    return np.stack((position-.25*direction,position+.25*direction))


def reprojected_lines(points,camera,attitude):
    relative=points[None,:,:,:]-camera[:,None,None,:]
    plane=np.cross(relative[:,:,0],relative[:,:,1])
    line=np.einsum('nji,nmj->nmi',attitude,plane)*[1.,1.,-FOCAL]
    line/=np.linalg.norm(line[:,:,:2],axis=-1,keepdims=True)
    return line*np.where(line[:,:,1:2]>=0,1.,-1.)


def line_residual(observed,predicted):
    coordinates=np.array(((-81.,-43.),(-75.,-30.),(25.,48.),(-12.,-4.)))
    usable=np.isfinite(observed).all(axis=2)
    measured=np.where(usable[:,:,None],observed,0.)
    predicted_y=np.maximum(predicted[:,:,1,None],.0001)
    measured_y=np.maximum(measured[:,:,1,None],.0001)
    use_x=measured_y>.45
    ys=-(measured[:,:,0,None]*coordinates+measured[:,:,2,None])/measured_y
    # The nearly vertical table border is compared at visible image rows.
    rows=np.array((-29.,-43.))
    measured_x=np.where(np.abs(measured[:,:,0,None])>.45,measured[:,:,0,None],1.)
    xs=-(measured[:,:,1,None]*rows+measured[:,:,2,None])/measured_x
    at_cross_sections=np.where(use_x,
        predicted[:,:,0,None]*coordinates+predicted_y*ys+predicted[:,:,2,None],
        predicted[:,:,0,None]*xs+predicted_y*rows+predicted[:,:,2,None])
    # A sign invariant angle remains continuous when a line is exactly vertical.
    angle=30.*(measured[:,:,0]*predicted[:,:,1]-measured[:,:,1]*predicted[:,:,0])
    return np.concatenate((at_cross_sections,angle[...,None]),axis=-1)[usable].ravel()


def link_pose(q,links,shape):
    """Apply small generic offsets and joint-axis tilts to the vendor chain."""
    stretches=shape[:3]
    axes=shape[3:].reshape(3,2)
    position=np.zeros((len(q),3))
    orientation=np.broadcast_to(links.basis,(len(q),3,3)).copy()
    for i,(offset,frame) in enumerate(links.links):
        if i in (1,2,3):
            offset=offset.copy()
            along=offset/np.linalg.norm(offset)
            offset+=stretches[i-1]*along
        position+=orientation@offset
        orientation=orientation@frame
        direction=np.r_[axes[i-1],1.] if i in (1,2,3) else np.array((0.,0.,1.))
        spin=Rotation.from_rotvec(q[:,i,None]*direction/np.linalg.norm(direction)).as_matrix()
        orientation=orientation@spin
    return position, orientation


def camera_pose(q,decoder,theta):
    position,orientation=link_pose(q,decoder.links,theta[6:])
    eye=position+decoder.offset[:3]+orientation@theta[3:6]
    rotation=orientation@Rotation.from_rotvec(theta[:3]).as_matrix()
    return eye,rotation


def pose_at(decoder,q):
    body=np.asarray([decoder.body(row) for row in q])
    return body[:,:3],body[:,3:12].reshape(-1,3,3)


def fit_model(clips,model,receipt):
    decoder=load_public(CALIBRATION).decoder
    home=np.r_[Rotation.from_matrix(decoder.wrist_attitude).as_rotvec(),decoder.translation,np.zeros(9)]
    with np.load(clips,allow_pickle=False) as observed:
        qs,qt=(observed[f'{arm}_q'] for arm in ('stock','alternate'))
        ss,st=(observed[f'{arm}_edges'] for arm in ('stock','alternate'))
    fit_q=qs.reshape(-1,5)
    fit_s=ss.reshape(-1,4,3)
    ci,ri=pose_at(decoder,fit_q)
    points=np.asarray([infer_line(ci,ri,fit_s[:,edge]) for edge in range(len(EDGES))])
    fit_s=st[:2].reshape(-1,4,3)
    fit_q=qt[:2].reshape(-1,5)
    # Local corrections around carried stock geometry preserve metric scale.
    weights=np.r_[np.repeat(1.1,3),np.repeat(17.,3),np.repeat(55.,3),np.repeat(16.,6)]
    bounds=np.r_[np.repeat(.50,3),np.repeat(.18,3),np.repeat(.070,3),np.repeat(.35,6)]

    def residual(theta,q=fit_q,key=fit_s):
        camera,attitude=camera_pose(q,decoder,theta)
        projection=reprojected_lines(points,camera,attitude)
        return line_residual(key,projection)

    def loss(theta):
        return np.r_[residual(theta),weights*(theta-home)]

    rigid=least_squares(lambda u: loss(np.r_[u,home[6:]]),home[:6],
                        bounds=(home[:6]-bounds[:6],home[:6]+bounds[:6]),loss='soft_l1',f_scale=.8)
    rigid_theta=np.r_[rigid.x,home[6:]]
    combined=least_squares(loss,rigid_theta,bounds=(home-bounds,home+bounds),loss='soft_l1',f_scale=.8)
    held_q,held_s=qt[2],st[2]
    all_models={'unmodified':home,'mount':rigid_theta,'mount_links':combined.x}
    metric={}
    for name,theta in all_models.items():
        train=residual(theta)
        test=residual(theta,held_q,held_s)
        metric[name]=dict(training_px=float(np.sqrt(np.mean(train**2))),withheld_px=float(np.sqrt(np.mean(test**2))),
                          withheld_median_px=float(np.median(np.abs(test))),parameters=theta.tolist())
    np.savez_compressed(model,points=points,source_theta=home,mount=rigid_theta,mount_links=combined.x)
    public=dict(schema='tt24-wrist-static-edges-1',fit_clips=list(PROBES[:2]),withheld_clip=PROBES[2],
                source_clips=list(PROBES),target_control_ticks=2*CLIP_TICKS,
                target_rgb_frames=2*CLIP_TICKS//SAMPLE,held_rgb_frames=CLIP_TICKS//SAMPLE,
                source_rgb_frames=3*CLIP_TICKS//SAMPLE,source_hash=_hash(CALIBRATION),archive_hash=_hash(clips),
                checkpoint_hash=_hash(model),features=list(EDGES),metrics=metric)
    _write(receipt,public)
    print(metric,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('collect','fit'))
    parser.add_argument('--clips',type=Path,default=CLIPS)
    parser.add_argument('--map',type=Path,default=MAP)
    parser.add_argument('--fit',type=Path,default=FIT)
    args=parser.parse_args()
    if args.task=='collect':
        acquire(args.clips)
    else:
        fit_model(args.clips,args.map,args.fit)
