"""Train-only spectral encoding of causal aligned opponent RGB shape histories."""
from pathlib import Path
from time import monotonic,process_time
import json
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.linalg import eigh
from . import tt57_visual_intent as study

ROOT=study.ROOT
HISTORY=12
RANK=24


def frame_vector(frame,bg):
    parts=[];okay=[]
    for image,background in zip(frame,bg,strict=True):
        shape=study.silhouette(image,background)
        if shape is None:
            parts.append(np.zeros((2,14,14)));okay.append(0.);continue
        center=shape[0];axis=np.arange(14)-6.5
        y,x=np.meshgrid(center[1]+axis,center[0]+axis,indexing='ij')
        # Full local photometry and silhouette, without a fitted face/stroke.
        delta=(image.astype(float)-background.astype(float)).mean(-1)/64
        photo=map_coordinates(delta,[y,x],order=1,mode='constant',cval=0)
        u=shape[3];x0=center[0]+u[0,0,0];y0=center[1]+u[0,0,1]
        alpha=map_coordinates(shape[2],[y-y0,x-x0],order=1,mode='constant',cval=0)
        parts.append(np.stack((np.clip(photo,-2,2),alpha)));okay.append(1.)
    return np.r_[np.array(parts).ravel(),okay].astype(np.float32)


def extract(d):
    path=ROOT/'movie-features.npz'
    if path.exists():
        with np.load(path) as z:return z['movie']
    clock=monotonic();cpu=process_time();found={}
    with np.load(study.old.ROOT/'model.npz') as z:bg=z['background']
    for p in sorted((study.old.ROOT/'observations').glob('case-*.npz')):
        with np.load(p) as z:raw={k:z[k] for k in z.files}
        release=int(raw['id'])
        for stream in range(len(raw['label'])):
            history=[];used=set()
            for k in np.flatnonzero(raw['stream']==stream):
                history.append(frame_vector(raw['rgb'][k],bg));x=raw['ball'][k]
                if not np.isfinite(x).all() or x[4]<.9:continue
                for gate in study.old.GATES:
                    if x[1]<gate or gate in used:continue
                    used.add(gate);recent=history[-HISTORY:];count=len(recent)
                    stacked=[recent[0]]*(HISTORY-count)+recent
                    found[(gate,release,stream)]=np.r_[np.array(stacked).ravel(),np.r_[np.zeros(HISTORY-count),np.ones(count)]]
        print('movie',p.name,flush=True)
    movie=np.array([found[(float(g),int(r),int(s))] for g,r,s in zip(d['gate'],d['release'],d['stream'],strict=True)],dtype=np.float32)
    np.savez_compressed(path,movie=movie)
    study._write(ROOT/'movie-extraction.json',dict(source_sha256=study._hash(Path(__file__)),shape=movie.shape,
        history_frames=HISTORY,maximum_history_s=(HISTORY-1)*2/60,
        wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu,
        scope='Past RGB only, subpixel centered14x14 photometry plus opacity per view. No orientation class, future frame or contact metadata. Padding flagged.'))
    return movie


def main():
    d=study.extract();movie=extract(d);y=study.sensor_future(d['target']);out={};predictions=[]
    from .tt50_group_forecast import columns
    for gate in study.old.GATES:
        train=(d['gate']==gate)&~d['held'];held=(d['gate']==gate)&d['held']
        # Basis sees only training-release pixels; analytic PCA, no backprop.
        mean=movie[train].mean(0);A=movie[train].astype(float)-mean
        val,vec=eigh(A@A.T,subset_by_index=(max(0,len(A)-RANK),len(A)-1))
        keep=val>1e-8;basis=(vec[:,keep].T@A)/np.sqrt(val[keep])[:,None]
        spectral=(movie.astype(float)-mean)@basis.T
        features=np.column_stack((d['baseline'][:,columns('position_face')],spectral))
        model=study.Kernel(features[train],y[train]);pred=model.predict(features[held])
        out[str(gate)]=study.metrics(pred,y[held],d['lead'][held])
        for release,stream,p,t in zip(d['release'][held],d['stream'][held],pred,y[held],strict=True):
            predictions.append(dict(gate=gate,release=int(release),stream=int(stream),prediction=p.tolist(),target=t.tolist()))
        print('MOVIE RESULT',gate,out[str(gate)],flush=True)
    study._write(ROOT/'movie-comparison.json',dict(source_sha256=study._hash(Path(__file__)),gates=out,
        rank=RANK,history=HISTORY,split='Same first16 training releases and8 held; PCA fitted only on training pixels; fixed rank24 and same .22 kernel ridge. No held tuning.'))
    study._write(ROOT/'movie-held-predictions.json',predictions)


if __name__=='__main__':main()
