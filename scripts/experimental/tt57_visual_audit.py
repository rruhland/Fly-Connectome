"""Audit whole cached RGB histories without training or native-state inputs."""
from pathlib import Path
from hashlib import sha256
import json
import argparse
import numpy as np

ROOT=Path('artifacts/tt57_visual_intent')


def run():
    paths=sorted(Path('artifacts/tt50_opponent_vision/observations').glob('case-*.npz'))
    if len(paths)!=24:raise ValueError('Expected the fixed24-release corpus')
    data=[]
    for p in paths:
        with np.load(p) as z:d={k:z[k] for k in z.files}
        if len(d['label'])!=6:raise ValueError('Expected six actual broad replies')
        data.append((p,d))
    results={};collisions=[]
    for history in (4,12):
      for gate in (-.45,0.,.45,.85):
        leads=[];n=0;low=0;pairs=0;identical=0;spreads=[]
        for path,d in data:
            rows=[]
            for stream,label in enumerate(d['label']):
                if not np.isfinite(label).all():continue
                indices=np.flatnonzero(d['stream']==stream)
                at=next((j for j,k in enumerate(indices) if np.isfinite(d['ball'][k]).all()
                    and d['ball'][k,4]>.9 and d['ball'][k,1]>=gate),None)
                if at is None or at<history-1:continue
                k=indices[at];pixels=d['rgb'][indices[at-history+1:at+1]]
                x,y,z,vx,vy,vz=label;dt=.43/max(1.35,-.98*vy)
                target=np.array([x+dt*vx,z+dt*vz-4.905*dt*dt]);lead=float((d['contact'][stream]-d['tick'][k])/60)
                rows.append((stream,sha256(pixels.tobytes()).hexdigest(),target,lead))
                leads.append(lead);n+=1;low+=int(target[1]<1.)
            for i,a in enumerate(rows):
                for b in rows[i+1:]:
                    pairs+=1
                    if a[1]==b[1]:
                        identical+=1;spreads.append(abs(a[2]-b[2]))
                        collisions.append(dict(file=str(path),release=int(d['id']),gate=gate,history=history,
                            streams=[a[0],b[0]],lead_s=[a[3],b[3]],future_observed_spread=abs(a[2]-b[2]).tolist()))
        results[f'{history}/{gate}']=dict(n=n,low=low,pairs=pairs,identical_pairs=identical,
            lead_s=np.percentile(leads,[10,50,90]).tolist() if leads else None,
            max_future_spread=np.max(spreads,axis=0).tolist() if spreads else None)
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'history-ambiguity-audit.json').write_text(json.dumps(dict(
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),raw_releases=len(data),actual_streams=144,
        gates=results,collisions=collisions,
        limitation='Equality applies to stored fixed crops and past-history length only. Contact time is lead metadata, never predictor input. It cannot establish whole-scene impossibility.'),indent=2)+'\n')
    print(results,flush=True)


def baseline_parity():
    from . import tt50_far_readout as old
    with np.load(old.ROOT/'model.npz') as z:bg=z['background']
    rows=[]
    for path in sorted((old.ROOT/'observations').glob('case-*.npz')):
        fresh={(r[0],r[1]):(r[3],r[4]) for r in old.examples(path,bg)}
        cached=old.ROOT/'cached'/'observations'/path.name
        if not cached.exists():continue
        with np.load(cached) as z:
            saved={(float(g),int(s)):(x,y) for g,s,x,y in zip(z['gate'],z['stream'],z['x'],z['y'],strict=True)}
        exact=fresh.keys()==saved.keys() and all(np.array_equal(fresh[k][0],saved[k][0])
            and np.array_equal(fresh[k][1],saved[k][1]) for k in fresh)
        if not exact:raise ValueError(f'Old baseline cache differs from actual pixels: {path}')
        rows.append(dict(file=str(path),rows=len(fresh),exact_raw_to_cached_features_and_labels=exact))
    (ROOT/'raw-cache-parity.json').write_text(json.dumps(dict(files=rows,total_rows=sum(r['rows'] for r in rows),
        all_exact=True,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('baseline parity',len(rows),'files',sum(r['rows'] for r in rows),'rows',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-parity',action='store_true');args=p.parse_args()
    run()
    if args.baseline_parity:baseline_parity()
