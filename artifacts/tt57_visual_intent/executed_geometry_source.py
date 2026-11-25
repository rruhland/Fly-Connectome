"""Camera-only stereo disk geometry, then release-disjoint future-RGB prediction.

No opponent pose/state, goal, contact time, seed or named stroke is a predictor
input. Contact time is retained solely as evaluator lead-time metadata. Labels
are the ball later observed in RGB at y=-1.10 in the existing physical tapes.
"""
from pathlib import Path
from time import monotonic,process_time
import json
import numpy as np
from scipy.ndimage import label as components
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from . import tt50_far_readout as old
from .tt51_flight_risk import sensor_future,Kernel as OldKernel
from .tt4_transfer_trial import _hash,_write

ROOT=Path('artifacts/tt57_visual_intent')
CALIBRATION=ROOT/'camera-calibration.json'
_c=json.loads(CALIBRATION.read_text())
EYES=np.array([_c[k]['eye'] for k in ('left','right')]);FOCAL=float(_c['left']['focal'])
OFFSETS=np.array([[108,64],[171,64]])
ROT=np.array([_c[k]['rotation'] for k in ('left','right')])
PP=np.array([(_c['left']['width']-1)/2,(_c['left']['height']-1)/2])-OFFSETS


def silhouette(im,bg):
    a=im.astype(float);b=bg.astype(float);r,g,blue=np.moveaxis(a,-1,0)
    mask=(abs(a-b).max(-1)>=5)&(r<97)&(blue>=r)&(blue-r<19)&(g-r<10)
    lab,n=components(mask,np.ones((3,3)));count=np.bincount(lab.ravel(),minlength=2);count[0]=0
    which=int(count.argmax())
    if not n or count[which]<7:return None
    yy,xx=np.nonzero(lab==which);top=np.argsort(a[yy,xx].mean(-1))[:max(3,len(xx)//3)]
    fg=np.median(a[yy[top],xx[top]],axis=0)
    x0,x1=max(0,int(xx.min())-2),min(im.shape[1],int(xx.max())+3)
    y0,y1=max(0,int(yy.min())-2),min(im.shape[0],int(yy.max())+3)
    pixels=a[y0:y1,x0:x1];back=b[y0:y1,x0:x1];direction=back-fg
    alpha=np.clip(np.sum((back-pixels)*direction,axis=-1)/np.maximum(np.sum(direction**2,axis=-1),16.),0,1)
    alpha[(pixels[:,:,0]>110)|(pixels[:,:,0]-pixels[:,:,2]>12)]=0
    y,x=np.mgrid[y0:y1,x0:x1];mass=alpha.sum()
    if mass<5:return None
    c=np.array([(x*alpha).sum(),(y*alpha).sum()])/mass
    u=np.stack((x-c[0],y-c[1]),-1)
    flat=u.reshape(-1,2)
    cov=(flat.T*alpha.ravel())@flat/mass+np.eye(2)/12
    return c,cov,alpha,u,mass


def geometry(frame,bg,previous=None):
    found=[silhouette(im,b) for im,b in zip(frame,bg,strict=True)]
    if any(v is None for v in found):return None,None
    rays=[]
    for i,s in enumerate(found):
        ray=np.r_[(s[0]-PP[i])/FOCAL,1.]@ROT[i];ray/=np.linalg.norm(ray);rays.append(ray)
    A=sum(np.eye(3)-np.outer(r,r) for r in rays)
    center=np.linalg.solve(A,sum((np.eye(3)-np.outer(r,r))@e for r,e in zip(rays,EYES,strict=True)))
    if not (abs(center[0])<1.2 and .3<center[1]<2.4 and .4<center[2]<1.9):return None,None
    J=[]
    for i in range(2):
        p=ROT[i]@(center-EYES[i]);J.append(FOCAL*(ROT[i,:2]*p[2]-p[:2,None]*ROT[i,2])/p[2]**2)
    def matrices(v):
        normal=np.array([v[0],-1.,v[1]]);normal/=np.linalg.norm(normal)
        disk=np.eye(3)-np.outer(normal,normal);r=np.exp(v[2])
        return normal,[r*r/4*j@disk@j.T for j in J]
    def residual(v):
        _,pred=matrices(v)
        return np.concatenate([(m-s[1])[np.triu_indices(2)]/max(1.,np.trace(s[1])) for m,s in zip(pred,found,strict=True)])
    starts=[[0,0,np.log(.08)],[.7,0,np.log(.08)],[-.7,0,np.log(.08)],[0,.7,np.log(.08)],[0,-.7,np.log(.08)]]
    if previous is not None:starts=[previous,starts[0]]
    fits=[least_squares(residual,s,bounds=([-3,-3,np.log(.04)],[3,3,np.log(.14)]),max_nfev=18) for s in starts]
    fit=min(fits,key=lambda r:float(r.fun@r.fun));normal,pred=matrices(fit.x)
    reconstruction=[]
    for cov,s in zip(pred,found,strict=True):
        Q=4*cov;Q+=np.eye(2)*.03;u=s[3]
        rho=np.sqrt(np.maximum(0,np.einsum('...i,ij,...j->...',u,np.linalg.inv(Q),u)))
        scale=np.linalg.det(Q)**.25;alpha=np.clip(.5-(rho-1)*scale,0,1)
        reconstruction.append(float(np.mean((alpha-s[2])**2)))
    feature=np.r_[center,np.outer(normal,normal)[np.triu_indices(3)],np.exp(fit.x[2]),
        fit.fun@fit.fun,reconstruction,1.]
    return feature,fit.x


class Kernel:
    @staticmethod
    def gram(x,y):
        a=cdist(x[:,:6],y[:,:6],'sqeuclidean')/6
        b=cdist(x[:,6:],y[:,6:],'sqeuclidean')/(x.shape[1]-6)
        return .35*np.exp(-a/.95)+.30*np.exp(-b/1.3)+.35*np.exp(-a/2.8-b/2.2)
    def __init__(self,x,y):
        self.mean=np.median(x,axis=0);self.scale=np.maximum(np.percentile(abs(x-self.mean),75,axis=0),.001)
        self.x=np.clip((x-self.mean)/self.scale,-5,5);self.bias=y.mean(0)
        self.weights=solve(self.gram(self.x,self.x)+.22*np.eye(len(x)),y-self.bias,assume_a='pos')
    def predict(self,x):return self.gram(np.clip((x-self.mean)/self.scale,-5,5),self.x)@self.weights+self.bias


def extract():
    cache=ROOT/'geometry-features.npz'
    if cache.exists():
        with np.load(cache) as z:return {k:z[k] for k in z.files}
    clock=monotonic();cpu=process_time();rows=[];valid=0;frames=0
    with np.load(old.ROOT/'model.npz') as z:bg=z['background']
    paths=sorted((old.ROOT/'observations').glob('case-*.npz'))
    for file_index,path in enumerate(paths):
        with np.load(path) as z:data={k:z[k] for k in z.files}
        with np.load(old.ROOT/'cached'/'observations'/path.name) as z:
            baseline={(float(g),int(s)):(x,y) for g,s,x,y in zip(z['gate'],z['stream'],z['x'],z['y'],strict=True)}
        for stream,y in enumerate(data['label']):
            if not np.isfinite(y).all():continue
            previous=None;history=[];used=set()
            for k in np.flatnonzero(data['stream']==stream):
                feature,previous_new=geometry(data['rgb'][k],bg,previous);frames+=1
                if feature is None:feature=np.zeros(14) if not history else history[-1].copy();feature[-1]=0
                else:previous=previous_new;valid+=1
                history.append(feature);x=data['ball'][k]
                if len(history)<4 or not np.isfinite(x).all() or x[4]<.9:continue
                for gate in old.GATES:
                    if gate in used or x[1]<gate:continue
                    used.add(gate)
                    if (gate,stream) not in baseline:continue
                    recent=np.array(history[-4:]);descriptor=np.r_[x,recent[-1],(recent[-1]-recent[0])/3,recent.mean(0)]
                    bx,label=baseline[(gate,stream)]
                    rows.append((gate,int(data['id']),stream,file_index>=16,descriptor,bx,label,
                        (data['contact'][stream]-data['tick'][k])/60))
        print('geometry',file_index,int(data['id']),'rows',len(rows),'valid',valid,'/',frames,flush=True)
    d={name:np.array([r[i] for r in rows]) for i,name in enumerate(('gate','release','stream','held','geometry','baseline','target','lead'))}
    np.savez_compressed(cache,**d)
    _write(ROOT/'extraction.json',dict(source_sha256=_hash(Path(__file__)),frames=frames,valid_frames=valid,
        wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu,camera_calibration_sha256=_hash(CALIBRATION),
        scope='Camera-calibrated stereo circular-disk fitting. Photometric subpixel edge coverage, visible centroid triangulation, two-view conic normal/radius fit; only past images. No native pose or target used.'))
    return d


def metrics(pred,y,lead):
    delta=pred[:,5:7]-y[:,5:7];low=y[:,-1]>.5
    trigger=(pred[:,-1]>.24)&(pred[:,5]>-.29)&(pred[:,5]<-.045)&(pred[:,6]<1.035)
    order=np.argsort(-pred[:,-1]);top=order[:max(1,int(np.ceil(len(y)/4)))]
    return dict(n=len(y),actual_low_left=int(low.sum()),lead_s=np.percentile(lead,[10,50,90]).tolist(),
        deep_mae=np.mean(abs(delta),axis=0).tolist(),deep_p90_error=np.percentile(abs(delta),90,axis=0).tolist(),
        large_lateral_errors=int((abs(delta[:,0])>.15).sum()),triggered=int(trigger.sum()),
        true_captured=int((trigger&low).sum()),false_positive=int((trigger&~low).sum()),
        dangerous_false_positive=int((trigger&((y[:,6]>1.04)|(y[:,5]<-.32)|(y[:,5]>.025))).sum()),
        low_in_top_quarter=int(low[top].sum()),brier=float(np.mean((np.clip(pred[:,-1],0,1)-low)**2)))


def main():
    ROOT.mkdir(parents=True,exist_ok=True);d=extract();report=dict(source_sha256=_hash(Path(__file__)),gates={},
        split='Existing ordinary physical corpus only: first16 complete releases train, remaining8 hold. No hard-source inputs. No tuning on held results.',
        supervision='Actual later stereo ball observation at receiving line; native contact times evaluator-only for lead metadata.',
        fitting='Analytic stereo geometry + ridge/kernel solves, no backpropagation. Equal-data TT50 kernel comparator and frozen deployed TT50 comparator.')
    frozen={}
    with np.load(old.ROOT/'risk-model.npz') as z:
        for i,g in enumerate(old.GATES):frozen[g]=OldKernel(tuple(z[f'{i}_{k}'] for k in range(6)))
    from .tt50_group_forecast import columns
    predictions=[]
    for gate in old.GATES:
        train=(d['gate']==gate)&~d['held'];held=(d['gate']==gate)&d['held'];y=sensor_future(d['target'])
        outputs={}
        for name,features in (('tt50_equal_data',d['baseline'][:,columns('position_face')]),('stereo_geometry',d['geometry'])):
            model=Kernel(features[train],y[train]);outputs[name]=model.predict(features[held])
        outputs['tt50_frozen']=frozen[gate].predict(d['baseline'][held])
        report['gates'][str(gate)]={name:metrics(pred,y[held],d['lead'][held]) for name,pred in outputs.items()}
        for name,pred in outputs.items():
            predictions.extend(dict(gate=gate,model=name,release=int(release),stream=int(stream),prediction=p.tolist(),target=t.tolist())
                for release,stream,p,t in zip(d['release'][held],d['stream'][held],pred,y[held],strict=True))
        print('RESULT',gate,report['gates'][str(gate)],flush=True)
    _write(ROOT/'comparison.json',report);_write(ROOT/'held-predictions.json',predictions)


if __name__=='__main__':main()
