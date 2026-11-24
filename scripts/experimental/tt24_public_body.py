"""Identify the stock wrist and paddle sensor from ordinary stereo RGB and encoders."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _write
from .tt19_factory import FactoryModel
from .tt20_fixture import ARCHIVED_FIXTURE, TABLE_EDGE
from .tt24_bounce_ident import WristState, _ellipse, _unit_camera_ray
from .tt24_metric_stereo import Camera, triangulate
from .tt24_predict_reference import HEIGHT, TrainingEyes


OUTPUT = Path('artifacts/tt24/public-body.npz')
RECEIPT = Path('artifacts/tt24/public-body.json.gz')
SHARED_OUTPUT = Path('artifacts/tt24/public-body-shared.npz')
SHARED_RECEIPT = Path('artifacts/tt24/public-body-shared.json.gz')
WIDTH_OUTPUT = Path('artifacts/tt24/public-body-width.npz')
WIDTH_RECEIPT = Path('artifacts/tt24/public-body-width.json.gz')
IMPULSE_RECEIPT = Path('artifacts/tt24/public-body-impulse.json.gz')
FUSED_OUTPUT = Path('artifacts/tt24/public-body-fused.npz')
FUSED_RECEIPT = Path('artifacts/tt24/public-body-fused.json.gz')
PROBES = tuple(range(365001, 365009))
HELD = tuple(range(365101, 365105))
SHARED_HELD = tuple(range(365201, 365205))
WIDTH_HELD = tuple(range(365301, 365305))
FUSED_HELD = tuple(range(365401, 365405))
ANGLES = 2*np.pi*np.arange(16)/16
DIRECTIONS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
RIM = np.column_stack((np.cos(2*np.pi*np.arange(44)/44), np.sin(2*np.pi*np.arange(44)/44)))
FOCAL = HEIGHT/(2*np.tan(np.deg2rad(54.)))


def stock(*, fixture=ARCHIVED_FIXTURE):
    arena = TrainingEyes(fixture=fixture)
    arena.max_steps = 72
    arena._model.vis.global_.fovy = 23.
    for eye, yaw, tilt in zip(arena.eyes, (12.,102.), (-36.,-33.), strict=True):
        eye.lookat[:] = (0., -.574, 1.1844) if fixture == TABLE_EDGE else (0., -.94, .93)
        eye.distance = 1.48
        eye.azimuth = yaw
        eye.elevation = tilt
    return arena


def stationary(arena):
    _ = arena.fixed_images()
    cameras = []
    for eye in arena.eyes:
        arena._renderer.update_scene(arena._data, camera=eye, scene_option=arena._scene_option)
        cameras.append(Camera.from_fixed_renderer(arena._renderer.scene))
    return tuple(cameras)


def blade_profile(rgb):
    r,g,b = rgb.transpose(2,0,1)
    mask = (r > 49) & (g < .44*r) & (b < .53*r)
    labels, n = label(mask)
    if not n:
        return np.full(2+len(DIRECTIONS),np.nan)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    component = counts.argmax()
    if counts[component] < 60:
        return np.full(2+len(DIRECTIONS),np.nan)
    y,x = np.nonzero(labels==component)
    pixels = np.column_stack((x-(rgb.shape[1]-1)/2, (rgb.shape[0]-1)/2-y))
    return np.r_[pixels.mean(0), (pixels@DIRECTIONS.T).max(axis=0)+.32]


def sight(rgb):
    found = _ellipse(rgb)
    return np.full(3,np.nan) if found is None else found


def probe(seed, tick, home, joints):
    """Balanced motor babble in joint coordinates, independent of the ball."""
    rng = np.random.default_rng(seed)
    sign = rng.choice((-1.,1.),size=5)
    mask = np.roll(np.array((1., .45, .65, .52, .72)), seed%5)
    take = 1. if tick//14%2 == 0 else -1.
    rate = .24*take*sign*mask + .095*np.sin(.28*tick+np.arange(5)*1.9)
    return np.clip(rate - 1.25*(joints-home), -.43, .43)


def collect(seeds, tapes=None):
    arena = stock()
    records = []
    try:
        for seed in seeds:
            obs = arena.reset(seed=seed)
            home = obs['joints'].copy()
            if not records:
                eyes = stationary(arena)
            online = []
            done = False
            while not done:
                command = (probe(seed, len(online), home, obs['joints']) if tapes is None
                           else np.asarray(tapes[seed]['actions'][len(online)]))
                frames = arena.fixed_images()
                online.append((obs['joints'].copy(), sight(obs['rgb']), np.array([sight(im) for im in frames]),
                               np.array([blade_profile(im) for im in frames]), command))
                obs, _, done = arena.step(command)
            enc, wrist, fixed, blade, actions = tuple(np.asarray(x) for x in zip(*online, strict=True))
            ending = arena.judge_metrics()
            if tapes is not None:
                replay_equal = json.dumps(ending, sort_keys=True) == json.dumps(tapes[seed]['ending'], sort_keys=True)
                if len(online) != tapes[seed]['ticks'] or not replay_equal:
                    raise ValueError('recorded public-probe episode changed')
            arena.reset(seed=seed,render=False)
            for command in actions:
                arena.step(command,render=False)
            if arena.judge_metrics()!=ending:
                raise ValueError('probe replay differs')
            records.append(dict(seed=seed, enc=enc, wrist=wrist, fixed=fixed, blade=blade, actions=actions,
                                terminal=ending))
            print('public probe',seed,len(actions),flush=True)
    finally:
        arena.close()
    return eyes, records


def circle_silhouette(center, slope, radius, thickness, camera):
    nx,ny = slope
    n = np.r_[nx,ny,1.]
    n /= np.linalg.norm(n)
    return silhouette(center,n,radius,thickness,camera)


def silhouette(center,n,radius,thickness,camera):
    e = np.cross(n,(0.,1.,0.))
    e /= np.linalg.norm(e)
    b = np.cross(n,e)
    disk = center + radius*(RIM@np.stack((e,b)))
    points = np.concatenate((disk+thickness*n,disk-thickness*n))
    image,_ = camera.project(points)
    return (image@DIRECTIONS.T).max(axis=0)


def disk_pose(eyes, profile, seed=None, fixed_size=None):
    center, _, _ = triangulate(eyes, profile[None, :, :2], tolerance=.07)
    if not np.isfinite(center).all() or not np.isfinite(profile).all():
        return None
    center = center[0]
    tilt = np.zeros(2) if seed is None else seed[3:5]
    spec = np.array((.073,.004)) if fixed_size is None else fixed_size

    def move(theta):
        nominal, slope, size = theta[:3], theta[3:5], theta[5:] if fixed_size is None else fixed_size
        images = np.asarray([circle_silhouette(nominal,slope,*size,eye) for eye in eyes])
        return np.r_[(images-profile[:,2:]).ravel(), (nominal-center)*48.]

    init = np.r_[center,tilt,spec] if fixed_size is None else np.r_[center,tilt]
    low, high = np.r_[center-.035, -.85,-.85], np.r_[center+.035,.85,.85]
    if fixed_size is None:
        low,high = np.r_[low,.045,0.], np.r_[high,.11,.014]
    fit = least_squares(move,np.clip(init,low,high),bounds=(low,high),loss='soft_l1',f_scale=.65,max_nfev=20)
    error = float(np.sqrt(np.mean(move(fit.x)[:2*len(DIRECTIONS)]**2)))
    return (fit.x,error) if error < 2.25 else None


class VendorLinks:
    """Use public link frames as an extrapolating basis, without a scene-state query."""
    def __init__(self):
        factory = FactoryModel()
        self.basis = factory.basis
        self.links = factory.transforms

    def read(self, q):
        r = np.broadcast_to(self.basis,(len(q),3,3)).copy()
        p = np.zeros((len(q),3))
        for angle,(offset,frame) in zip(q.T, self.links, strict=True):
            p += r@offset
            r = r@frame
            c,s = np.cos(angle),np.sin(angle)
            local = np.zeros_like(r)
            local[:,0,0] = local[:,1,1] = c
            local[:,1,0],local[:,0,1],local[:,2,2] = s,-s,1.
            r = r@local
        return p,r


def body_fit(records, eyes):
    raw_sizes = []
    for record in records:
        for profile in record['blade'][::12]:
            result = disk_pose(eyes,profile)
            if result is not None:
                raw_sizes.append(result[0][5:])
    if not raw_sizes:
        raise ValueError('no reliable paired RGB blade profiles')
    size = np.median(raw_sizes,axis=0)
    last = None
    errors, encoders, positions, normals = [], [], [], []
    for record in records:
        for index in range(0,len(record['enc']),3):
            result = disk_pose(eyes,record['blade'][index],last,fixed_size=size)
            if result is None:
                continue
            last,error = result
            normal = np.r_[last[3:5],1.]
            normals.append(normal/np.linalg.norm(normal))
            positions.append(last[:3])
            encoders.append(record['enc'][index])
            errors.append(error)
    if len(errors)<24:
        raise ValueError('insufficient confirmed disk silhouettes')
    poses,rotations = VendorLinks().read(np.asarray(encoders))
    weights = np.sqrt(1/(1+np.square(errors)))
    blocks = np.concatenate((np.broadcast_to(np.eye(3),rotations.shape),rotations),axis=2)
    response = np.asarray(positions)-poses
    shape = (len(weights)*3,6)
    offset = np.linalg.lstsq((blocks*weights[:,None,None]).reshape(shape),
                             (response*weights[:,None]).ravel(),rcond=.001)[0]
    pred = poses+offset[:3]+rotations@offset[3:]
    local_n = np.einsum('nij,ni->nj',rotations,np.asarray(normals))
    orient = np.average(local_n,axis=0,weights=weights)
    orient /= np.linalg.norm(orient)
    return offset, orient, dict(radius_m=float(size[0]),half_thickness_m=float(size[1]),sightings=len(errors),
                               silhouette_rmse_px=float(np.sqrt(np.mean(np.square(errors)))),
                               position_rmse_m=float(np.sqrt(np.mean(np.sum((pred-positions)**2,axis=1)))))


def shared_normal(records, eyes, offset, original, *, widths=False):
    """Fit the normal of one rigid pictured face across all actuated poses."""
    q = np.concatenate([row['enc'][::2] for row in records])
    pixels = np.concatenate([row['blade'][::2, :, 2:] for row in records])
    valid = np.isfinite(pixels).all(axis=(1,2))
    p,r = VendorLinks().read(q[valid])
    center = p+offset[:3]+r@offset[3:]
    reference = pixels[valid]
    phase = np.arctan2(original[1],original[0])

    def loss(params):
        angle,radius,half_thickness = params
        n = r@np.array((np.cos(angle),np.sin(angle),0.))
        e = np.cross(n,(0.,1.,0.))
        e /= np.linalg.norm(e,axis=1,keepdims=True)
        g = np.cross(n,e)
        plane = np.stack((e,g),axis=1)
        rim = center[:,None,:]+radius*(RIM@plane)
        shift = half_thickness*n[:,None,:]
        rim = np.concatenate((rim+shift,rim-shift),axis=1)
        projected = []
        for eye in eyes:
            image,_=eye.project(rim.reshape(-1,3))
            maxima = (image.reshape(len(rim),-1,2) @ DIRECTIONS.T).max(axis=1)
            projected.append(maxima)
        residual = np.stack(projected,axis=1)-reference
        if widths:
            half = len(DIRECTIONS)//2
            residual = .5*(residual[...,:half]+residual[...,half:])
        return residual.ravel()

    initial = np.array((phase,.075,.006))
    result = least_squares(loss,initial,bounds=([phase-1.,.055,.0],[phase+1.,.092,.014]),
                           loss='soft_l1',f_scale=.72,max_nfev=35)
    angle,radius,half_thickness = result.x
    local = np.r_[np.cos(angle),np.sin(angle),0.]
    return local,dict(visual_support_rmse_px=float(np.sqrt(np.mean(loss(result.x)**2))),
                      preceding_support_rmse_px=float(np.sqrt(np.mean(loss(initial)**2))),
                      fitted_normal=local.tolist(),radius_m=float(radius),half_thickness_m=float(half_thickness),
                      paired_pose_labels=int(valid.sum()), silhouette_widths=widths)


def ball_fit(records, eyes, body_offset):
    enc = np.concatenate([r['enc'] for r in records])
    stereo = np.concatenate([r['fixed'] for r in records])
    ball,gap,_ = triangulate(eyes,stereo[:,:,:2],tolerance=.028)
    wrist = np.concatenate([r['wrist'] for r in records])
    valid = np.isfinite(wrist).all(axis=1)&np.isfinite(ball).all(axis=1)
    if valid.sum()<24:
        raise ValueError('insufficient paired wrist ball')
    pos,r = VendorLinks().read(enc[valid])
    known = np.einsum('nij,ni->nj',r,ball[valid]-pos-body_offset[:3])
    scale_free = _unit_camera_ray(wrist[valid],FOCAL)
    a,b = scale_free-scale_free.mean(0),known-known.mean(0)
    u,s,vt = np.linalg.svd(a.T@b)
    signs = np.array([1.,1.,np.linalg.det(vt.T@u.T)])
    initial = vt.T@np.diag(signs)@u.T
    size = np.clip((s@signs)/np.sum(a*a),.010,.033)
    translation = known.mean(0)-size*(initial@scale_free.mean(0))
    strength = np.sqrt(1./(1.+(gap[valid]/.008)**2))
    def residual(params):
        angles,offset,rad = params[:3],params[3:6],params[6]
        estimate = Rotation.from_rotvec(angles).apply(scale_free*rad)+offset
        return ((estimate-known)*strength[:,None]).ravel()
    initial = np.r_[Rotation.from_matrix(initial).as_rotvec(),translation,size]
    solved = least_squares(residual,initial,loss='soft_l1',f_scale=.009,max_nfev=75)
    err = np.linalg.norm(residual(solved.x).reshape(-1,3),axis=1)
    return solved.x,dict(paired_ball_frames=int(valid.sum()),ball_radius_m=float(solved.x[-1]),
                         ball_registration_median_m=float(np.median(err))), ball


class PublicDecoder:
    def __init__(self,offset,normal,registration,default):
        self.links=VendorLinks()
        self.offset=offset
        self.normal=normal
        self.translation=registration[3:6]
        self.wrist_attitude=Rotation.from_rotvec(registration[:3]).as_matrix()
        self.radius=float(registration[6])
        self.default=default
        self.image_height=HEIGHT

    def body(self, joints):
        pos,r = self.links.read(np.asarray(joints)[None,:5])
        p,R=pos[0]+self.offset[:3],r[0]
        camera,blade = p+R@self.translation,p+R@self.offset[3:]
        normal=R@self.normal
        if normal[2]<0:
            normal=-normal
        return np.r_[camera,(R@self.wrist_attitude).ravel(),blade,normal[:2]]

    def ball(self,ellipse,body):
        if ellipse is None:
            return None
        ray=_unit_camera_ray(ellipse,FOCAL)*self.radius
        return body[:3]+body[3:12].reshape(3,3)@ray


class PublicMotion:
    def __init__(self,mean,scale,weights):
        self.mean,self.scale,self.weights=mean,scale,weights

    def predict(self,raw):
        x=np.clip((raw-self.mean)/self.scale,-5,5)
        return np.r_[1.,x]@self.weights


class GatedWrist(WristState):
    """Reject abrupt range outliers and propagate the last observed motion."""
    def __init__(self,decoder,motion=None):
        super().__init__(decoder,motion)
        self.missing = 0
        self.rejections = 0

    def read(self,ellipse,joints):
        if ellipse is not None and self.missing <= 3:
            body = self.decoder.body(joints)
            candidate = self.decoder.ball(ellipse,body)
            prior = body[12:15]+self.decoder.default if self.previous is None else self.previous+self.velocity/60.
            if self.previous is None:
                bad = np.any(np.abs(candidate-prior)>[.13,.13,.24])
            else:
                bad = np.linalg.norm(candidate-prior)>.065+.022*self.missing
            if bad:
                ellipse = None
                self.rejections += 1
        self.missing = self.missing+1 if ellipse is None else 0
        return super().read(ellipse,joints)


class ConsensusWrist(WristState):
    """Use short optical agreement to reacquire the ball after a bad mask."""
    def __init__(self,decoder,motion=None):
        super().__init__(decoder,motion)
        self.candidate = None
        self.rejections = 0

    def read(self,ellipse,joints):
        if ellipse is not None:
            body = self.decoder.body(joints)
            measured = self.decoder.ball(ellipse,body)
            home = body[12:15] if self.home is None else self.home
            relative = measured-home
            supported = np.max(np.abs(relative[:2]))<=.18 and -.07<=relative[2]<=.80
            prediction = home+self.decoder.default if self.previous is None else self.previous+self.velocity/60.
            threshold = .11 if self.previous is None else .066
            distant = np.linalg.norm(measured-prediction)>threshold
            agrees = self.candidate is not None and np.linalg.norm(measured-self.candidate)<.055
            if not supported or distant and not agrees:
                self.candidate = measured if supported else None
                self.rejections += 1
                ellipse = None
            else:
                self.candidate = None
                if distant:
                    self.history.clear()
                    self.previous = measured
                    self.velocity[:] = 0.
        if ellipse is None:
            self.velocity *= .78
        output=super().read(ellipse,joints)
        self.velocity=np.clip(self.velocity,[-1.8,-1.8,-4.5],[1.8,1.8,4.5])
        output[3:6]=self.velocity
        return output


def motion_fit(decoder,records,points,observer=0):
    xx,yy=[],[]
    offset=0
    for clip in records:
        wrist=clip['wrist']
        target=points[offset:offset+len(wrist)]
        offset+=len(wrist)
        model=(WristState,GatedWrist,ConsensusWrist)[observer](decoder)
        for t,(q,px) in enumerate(zip(clip['enc'],wrist,strict=True)):
            model.read(px if np.isfinite(px).all() else None, q)
            if t>=2 and t+1<len(target) and np.isfinite(target[t-1:t+2]).all() and np.isfinite(px).all():
                if np.linalg.norm(target[t+1]-2*target[t]+target[t-1])<.065:
                    xx.append(model.motion_input)
                    yy.append((target[t+1]-target[t-1])*30.)
    x,y=np.asarray(xx),np.asarray(yy)
    mean=x.mean(0)
    scale=np.maximum(x.std(0), np.r_[[.035]*3,[.2]*9])
    features=np.column_stack((np.ones(len(x)),np.clip((x-mean)/scale,-5,5)))
    weights=np.linalg.solve(features.T@features+3*np.eye(features.shape[1]),features.T@y)
    return PublicMotion(mean,scale,weights),int(len(x))


def load_public(path):
    """Load a wrist-only eleven-channel optical and proprioceptive observer."""
    with np.load(path,allow_pickle=False) as s:
        decoder=PublicDecoder(s['offset'],s['normal'],s['registration'],s['default'])
        motion=PublicMotion(s['motion_mean'],s['motion_scale'],s['motion_weights'])
        observer=int(s['observer']) if 'observer' in s else int('shared' in s and bool(s['shared']))
    return (WristState,GatedWrist,ConsensusWrist)[observer](decoder,motion)


def build(path, receipt, count=8, *, reuse=None, widths=False):
    tapes = None
    if reuse is not None:
        from .tt4_transfer_trial import _read
        tapes = {item['seed']:item for item in _read(reuse)['data']}
    eyes,records=collect(PROBES[:count],tapes)
    offset,normal,body=body_fit(records,eyes)
    if reuse is not None:
        normal,body['shared_projection']=shared_normal(records,eyes,offset,normal,widths=widths)
    registration,optics,points=ball_fit(records,eyes,offset)
    decoder=PublicDecoder(offset,normal,registration,np.zeros(3))
    starts=np.r_[0,np.cumsum([len(x['enc']) for x in records[:-1]])]
    origin=np.asarray([decoder.body(x['enc'][0])[12:15] for x in records])
    default=np.nanmedian(points[starts]-origin,axis=0)
    decoder.default=default
    observer=2 if widths else int(reuse is not None)
    motion,observations=motion_fit(decoder,records,points,observer)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,offset=offset,normal=normal,registration=registration,default=default,
                        motion_mean=motion.mean,motion_scale=motion.scale,motion_weights=motion.weights,
                        observer=observer)
    out=dict(schema='tt24-public-body-optical-fit-1', probes=count, frames=sum(len(r['enc']) for r in records),
             body=body,optics=optics,motion_samples=observations,calibration_sha256=_hash(path),
             data=[dict(seed=r['seed'],ticks=len(r['enc']),actions=r['actions'].tolist(),ending=r['terminal']) for r in records],
             label_sources=['paired fixed RGB disk silhouettes', 'paired fixed RGB ball', 'wrist RGB', 'encoders'])
    if reuse is not None:
        out['same_probes_reprojected_sha256']=_hash(reuse)
    _write(receipt,out)
    return out


def diagnose(path,seeds=HELD):
    from .tt24_bounce_ident import CleanState
    arena=stock()
    try:
        traces=[]
        for seed in seeds:
            sensor=load_public(path)
            obs=arena.reset(seed=seed,split='held')
            oracle=CleanState(arena)
            home=obs['joints'].copy()
            truth,pred,visible=[],[],[]
            done=False
            while not done:
                truth.append(oracle.observe(obs))
                pred.append(sensor.observe(obs))
                visible.append(_ellipse(obs['rgb']) is not None)
                obs,_,done=arena.step(probe(seed,len(truth)-1,home,obs['joints']))
            x,y=np.asarray(truth),np.asarray(pred)
            valid=np.asarray(visible)
            delta=y-x
            gap=lambda cols, mask: float(np.sqrt(np.mean(np.sum(delta[mask,cols]**2,axis=1))))
            traces.append(dict(seed=seed,frames=len(truth),visible=int(valid.sum()),ball_rmse=gap(slice(0,3),valid),
                               motion_rmse=gap(slice(3,6),valid),blade_rmse=gap(slice(6,9),slice(None)),
                               normal_xy_rmse=gap(slice(9,11),slice(None)),
                               rejected=getattr(sensor,'rejections',0)))
        return traces
    finally:
        arena.close()


def impulse_audit(path,receipt):
    """Compare the independent stereo flight change with the painted disk plane."""
    from .tt4_transfer_trial import _read

    probes=_read(RECEIPT)
    tapes={item['seed']:item for item in probes['data']}
    eyes,records=collect(tuple(tapes),tapes)
    decoder=load_public(path).decoder
    examples=[]
    acceleration=[]
    for row in records:
        ball,_,_=triangulate(eyes,row['fixed'][:,:,:2],tolerance=.028)
        examples.append(ball)
        a,b=np.diff(ball,axis=0)[:-1],np.diff(ball,axis=0)[1:]
        free=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)&(a[:,2]*b[:,2]>0)
        difference=(b-a)*3600
        free &= np.linalg.norm(difference,axis=1)<28.
        acceleration.extend(difference[free])
    drift=np.median(acceleration,axis=0)
    events=[]
    for row,ball in zip(records,examples,strict=True):
        blade=np.array([decoder.body(q)[12:15] for q in row['enc']])
        _,rotations=decoder.links.read(row['enc'])
        candidates=[]
        for t in range(3,len(ball)-3):
            previous=(ball[t-1]-ball[t-3])*30
            next_=(ball[t+3]-ball[t+1])*30
            change=next_-previous-drift*(4./60.)
            if not np.isfinite(change).all() or not previous[2]<-.37 or not next_[2]>.30 or change[2]<1.05:
                continue
            if not np.isfinite(ball[t]).all() or np.linalg.norm(ball[t]-blade[t])>.12:
                continue
            speed=float(np.linalg.norm(blade[t+1]-blade[t-1])*30.)
            if np.linalg.norm(change[:2])>.60*change[2]:
                continue
            direction=change/np.linalg.norm(change)
            local=rotations[t].T@direction
            if local@decoder.normal<0:
                local=-local
            candidates.append(dict(seed=row['seed'],tick=t,paddle_speed_mps=speed,impulse=change.tolist(),
                                   direction=direction.tolist(),local=local.tolist()))
        selected=[]
        for event in sorted(candidates,key=lambda event:-event['impulse'][2]):
            if all(abs(event['tick']-previous['tick'])>=7 for previous in selected):
                selected.append(event)
        events.extend(sorted(selected,key=lambda event:event['tick']))
    def aggregate(samples):
        if not samples:
            return dict(count=0)
        local=np.median([item['local'] for item in samples],axis=0)
        local/=np.linalg.norm(local)
        angle=float(np.rad2deg(np.arccos(np.clip(abs(local@decoder.normal),-1,1))))
        return dict(count=len(samples),median_local_normal=local.tolist(),angle_to_silhouette_deg=angle)
    result=dict(probe_tapes_sha256=_hash(RECEIPT),decoder_sha256=_hash(path),frames=sum(len(r['enc']) for r in records),
                stereo_acceleration_samples=len(acceleration),free_acceleration_mps2=drift.tolist(),events=events,
                all=aggregate(events),slow_paddle=aggregate([item for item in events if item['paddle_speed_mps']<=.14]),
                assumption='The largest near-face velocity reversal approximates the normal impulse.',
                used_to_train=False)
    _write(receipt,result)
    return result


def fuse(path,receipt):
    """Refine the visually fitted disk normal with observed contact impulses."""
    from .tt4_transfer_trial import _read

    source=_read(WIDTH_RECEIPT)
    impulse=_read(IMPULSE_RECEIPT)
    if impulse['decoder_sha256']!=_hash(WIDTH_OUTPUT):
        raise ValueError('the optical impulse audit uses a different disk fit')
    directions=np.asarray([item['local'] for item in impulse['events']])
    if len(directions)<3:
        raise ValueError('too few observable contacts for a residual estimate')
    local=np.median(directions,axis=0)
    local[2]=0.
    direction=local/np.linalg.norm(local)
    with np.load(WIDTH_OUTPUT,allow_pickle=False) as fit:
        fields={key:fit[key] for key in fit.files}
    # Keep a two-sample photographic prior against sliding or spinning impacts.
    normal=len(directions)*direction+2*fields['normal']
    fields['normal']=normal/np.linalg.norm(normal)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**fields)
    result={key:value for key,value in source.items() if key!='held_oracle_diagnostic'}
    result.update(calibration_sha256=_hash(path),impulse_evidence_sha256=_hash(IMPULSE_RECEIPT),
                  previous_checkpoint_sha256=_hash(WIDTH_OUTPUT),fused_local_normal=fields['normal'].tolist(),
                  contact_count=len(directions),source_of_contact_labels='paired RGB changes in ball velocity',
                  contact_assumption='The robust velocity jump is dominated by the face-normal impulse.')
    _write(receipt,result)
    return result


if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--output',type=Path)
    args.add_argument('--receipt',type=Path)
    args.add_argument('--drops',type=int,choices=range(1,9),default=8)
    args.add_argument('--diagnose',action='store_true')
    args.add_argument('--shared',action='store_true')
    args.add_argument('--width',action='store_true')
    args.add_argument('--impulses',action='store_true')
    args.add_argument('--fuse',action='store_true')
    cfg=args.parse_args()
    path=cfg.output or (FUSED_OUTPUT if cfg.fuse else WIDTH_OUTPUT if cfg.width else SHARED_OUTPUT if cfg.shared else OUTPUT)
    receipt=cfg.receipt or (FUSED_RECEIPT if cfg.fuse else WIDTH_RECEIPT if cfg.width else
                           SHARED_RECEIPT if cfg.shared else RECEIPT)
    if cfg.impulses:
        result=impulse_audit(path,cfg.receipt or IMPULSE_RECEIPT)
    elif cfg.diagnose:
        from .tt4_transfer_trial import _read
        result=_read(receipt)
        seeds=FUSED_HELD if cfg.fuse else WIDTH_HELD if cfg.width else SHARED_HELD if cfg.shared else HELD
        result['held_oracle_diagnostic']=diagnose(path,seeds)
        _write(receipt,result)
    elif cfg.fuse:
        result=fuse(path,receipt)
    else:
        result=build(path,receipt,cfg.drops,reuse=RECEIPT if cfg.shared or cfg.width else None,widths=cfg.width)
    print({k:v for k,v in result.items() if k!='data'},flush=True)
