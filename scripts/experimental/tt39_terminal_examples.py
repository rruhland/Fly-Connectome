"""Use observed legal moving endpoints, not prerecorded commands, as shooting targets."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as mpc
from . import tt39_terminal_early as early
from . import tt39_terminal_fork as forks
from .tt4_transfer_trial import _hash, _read, _write
from .tt38_body_relative import Body, groups
from .tt39_terminal_servo import OUT

OBSERVATIONS = Path("artifacts/tt37/self-combined.npz")
CALIBRATION = Path("artifacts/tt38-body-relative/pose.npz")
SCALE = np.array([.12,.16,.085,.75,.82,1.12])


def collect(bank_path, calibration, out):
    meta = _read(bank_path.with_suffix(".json"))
    if meta["bank_sha256"] != _hash(bank_path):
        raise ValueError("Observed bank integrity mismatch")
    with np.load(bank_path) as data, np.load(calibration) as calibrated:
        s, mark = data["states"], data["marks"]
        body = Body(calibrated["weights"])
    point,releases=groups(meta)
    t=mark[:,1]
    candidate=np.flatnonzero((s[:,4]<-.9)&(t<0.)&(t>=-mpc.DT)&(abs(s[:,0])<.235)&(s[:,2]<1.10))
    correct=(candidate>0)&(point[candidate]==point[candidate-1])&(mark[candidate,0]-mark[candidate-1,0]==1)
    rows, label, receipt = [], [], []
    sources = {}
    seen=set()
    for i in candidate[correct]:
        context=meta["practice"][point[i]]
        name=context["source"]
        if name not in sources:
            path=bank_path.parent/name
            if _hash(path)!=meta["sources"][name]:
                raise ValueError("Changed legal warmup archive")
            sources[name]=_read(path)
        prior=sources[name]
        row=prior["rows"][context["source_row"]]
        time=mark[i,0]*mpc.DT-t[i]
        matches=[e for e in row["score"]["shots"] if e["side"]==1 and abs(e["hit"]-time)<1.e-7]
        if len(matches)!=1:
            continue
        key=(name,context["source_row"],time)
        if key in seen:
            continue
        seen.add(key)
        hits=[e for e in row["events"] if e["kind"]=="contact" and e["side"]==1 and abs(e["time"]-time)<1.e-7]
        if len(hits)!=1:
            continue
        hit=hits[0]
        ball=np.array(hit["ball"])
        q=s[i,6:11]
        q_rate=(q-s[i-1,6:11])/mpc.DT
        moved=q+(-t[i])*q_rate
        pose=body.position(moved)
        n=pose[3:6]
        n*=1. if n[1]>=0. else -1.
        dx=ball-pose[:3]
        # Actual encoder pose was sampled within 1 tick of legal contact.
        # The ball truth is only a training label in this file, never a live observation.
        normal=float(dx@n)
        rim=np.linalg.norm(dx-normal*n)
        if normal<-.015 or normal>.060 or rim>.080 or pose[2]>1.105 or abs(ball[0])>.235:
            continue
        v=(body.jacobian(moved)[:3]@q_rate)
        normal_speed=float(np.clip(v@n,-.10,2.5))
        rows.append(np.r_[ball,hit["incoming"]])
        label.append(np.r_[dx,n,normal_speed,moved])
        receipt.append(dict(id=int(releases[i]), source=name,source_row=context["source_row"],
                            impact_time_s=float(time),body_center=pose[:3].tolist(),normal_speed=normal_speed))
    if len(rows)<8:
        raise ValueError("Too few registered moving endpoints")
    np.savez_compressed(out,features=np.asarray(rows),endpoints=np.asarray(label))
    low=[e for e in receipt if e["body_center"][2]<1.015]
    _write(out.with_suffix(".json"),dict(endpoints=len(rows),low_endpoints=len(low),train=receipt,
           bank_sha256=_hash(bank_path),pose_sha256=_hash(calibration),endpoints_sha256=_hash(out),
           note="Real legally rewarded contact locations as training labels paired with precontact camera/encoder samples. No action chunk is stored."))
    print("legal observed terminal states",len(rows),"low",len(low),flush=True)


class EmpiricalServo(early.EarlyServo):
    """Infer a moving terminal target from successful endpoints; recompute all motion online."""

    def __init__(self,*args,examples,**kw):
        super().__init__(*args,**kw)
        with np.load(examples) as z:
            self.x,self.label=z["features"],z["endpoints"]
        self.endpoint_index=cKDTree(self.x/SCALE)

    def early_targets(self,q):
        twin=self.engine.twin
        path=self.engine.nominal.flight.path(twin.ball,twin.velocity,twin.spin,1,
                                                bounced=self.engine.bounced,horizon=.49)
        if not len(path):
            return []
        times,p=path[:,0],path[:,1:4]
        low=np.any((p[:,1]>-1.72)&(p[:,1]<-1.43)&(p[:,2]<1.015)&(abs(p[:,0])<.145))
        possible=(times>=.100)&(times<early.HORIZON*mpc.DT)&(p[:,1]<-1.04)&(p[:,1]>-1.79)
        possible&=(p[:,2]>.83)&(p[:,2]<1.075)&(abs(p[:,0])<.16)
        select=np.flatnonzero(possible)
        if not len(select) or not low:
            return []
        rows=[]
        for k in np.unique(np.rint(np.linspace(select[0],select[-1],5)).astype(int)):
            frames=int(round(times[k]/mpc.DT))
            if not 6<=frames<=early.HORIZON or not possible[k]:
                continue
            j=int(np.argmin(abs(times-frames*mpc.DT)))
            ball,pvel=path[j,1:4],path[j,4:7]
            d,ix=self.endpoint_index.query(np.r_[ball,pvel]/SCALE,12)
            for dist,key in zip(d,ix,strict=True):
                if dist>2.25:
                    break
                dx,n,speed,prior=self.label[key,:3],self.label[key,3:6],self.label[key,6],self.label[key,7:]
                if np.max(abs(prior-q[:5]))>4.5*frames*mpc.DT+.18:
                    continue
                goal_center=ball-dx
                angle,fit=self.geometry.fit(goal_center,n,prior)
                travel=np.max(abs(angle-q[:5]))
                mismatch=np.linalg.norm(fit[0]-goal_center)
                theta=np.linalg.norm(fit[1]-n)
                if mismatch>.043 or theta>.29 or travel>4.9*frames*mpc.DT+.16:
                    continue
                p_goal,n_goal=fit[0],fit[1]
                r=ball-p_goal
                axial=r@n_goal
                tangent=np.linalg.norm(r-axial*n_goal)
                if tangent>.073 or not -.011<axial<.065:
                    continue
                rank=float(dist+4*mismatch+.13*travel+.18*theta)
                rows.append((rank,frames,np.r_[p_goal,n_goal,speed],angle,fit))
        keep=[]
        for row in sorted(rows,key=lambda r:r[0]):
            if any(abs(row[1]-a[1])<3 and np.linalg.norm(row[2][3:6]-a[2][3:6])<.20 for a in keep):
                continue
            keep.append(row)
            if len(keep)>=4:
                break
        return keep


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("collect","forks","live"))
    p.add_argument("output",type=Path)
    p.add_argument("--examples",type=Path,default=OUT/"empirical-endpoints.npz")
    p.add_argument("--bank",type=Path,default=mpc.BANK)
    p.add_argument("--optics",type=Path,default=Path("artifacts/tt37/rebound.npz"))
    p.add_argument("--limit",type=int,default=4)
    args=p.parse_args()
    if args.mode=="collect":
        collect(OBSERVATIONS,CALIBRATION,args.output)
    else:
        def create(*a,**kw):
            return EmpiricalServo(*a,examples=args.examples,**kw)
        if args.mode=="forks":
            with patch.object(forks,"eligible",early.diagnostic_times),patch.object(forks,"TerminalServo",create):
                forks.course(args.bank,args.optics,args.output,args.limit)
        else:
            with patch.object(mpc,"MPC",create):
                mpc.run(Path("/tmp/tt25-rallies.json"),args.bank,mpc.PLAN,"development","live",args.output,
                        optic=args.optics,variant="elastic",limit=args.limit)
        result=_read(args.output)
        result["empirical_endpoints_sha256"]=_hash(args.examples)
        _write(args.output,result)
