"""Mode-preserving, reward-credited preparation from the first inbound ball pixels."""
import argparse
from collections import deque
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried

ROOT=Path("artifacts")
OUT=ROOT/"tt42"
PACKETS=[ROOT/"tt37/camera-time-school-outcomes.npz"]+[
    ROOT/f"tt39/earned-0{i}-outcomes.npz" for i in (1,2,3)]
SEED_PRACTICE=ROOT/"tt37/camera-time-school.npz"
OPTIC=ROOT/"tt37/rebound.npz"
COURSE=ROOT/"tt39/endurance-plan.json"
SCALE=np.r_[.17,.28,.115,.64,.88,.90,.50,.44,.44,.47,.56]
MOTION=np.array([.21,.16,.22,.31,.35])
WINDOW=(.06,1.51)
H=6
THRESHOLD=.014


def early(x):
    return (x is not None and np.isfinite(x).all() and WINDOW[0]<x[1]<WINDOW[1]
            and x[4]<-1.17 and .77<x[2]<1.67)


def split_passages(states):
    """Distinguish approach episodes in chronologically saved inbound rows."""
    # Any new optical approach begins at a substantially more positive y.
    # Break also on occlusions with no dense support. No hidden time labels.
    y=states[:,1]
    return np.cumsum(np.r_[True,(np.diff(y)>.18)|(np.diff(y)<-.31)])


def load_packets(paths):
    x=[]; d=[]; scores=[]; runs=[]; counts=[]; references=[]; releases=[]
    group_counter=0
    for path in paths:
        receipt=_read(path.with_suffix(".json"))
        if receipt["data_sha256"]!=_hash(path) or receipt["optics_sha256"]!=_hash(OPTIC):
            raise ValueError(f"Changed scored optical practice {path}")
        with np.load(path) as f:
            s,u,y=(f[k].copy() for k in ("states","references","reward"))
        n=0; total=0
        for point in receipt["practice"]:
            j=n+point["frames"]
            obs,action=s[n:j],u[n:j]
            if len(obs):
                grp=split_passages(obs)+group_counter
                good=np.array([early(v[:6]) for v in obs])
                x.extend(obs[good])
                references.extend(action[good,:5])
                d.extend(action[good,6:11]-action[good,:5])
                scores.extend(y[n:j][good])
                runs.extend(grp[good])
                releases.extend([point["card"]["id"]]*int(good.sum()))
                group_counter=int(grp[-1])
                total+=int(good.sum())
            n=j
        if n!=len(s):
            raise ValueError("Practice source rows or capture grouping differ")
        counts.append(dict(path=str(path),sha256=_hash(path),visible_frames=total))
    return tuple(np.asarray(v) for v in (x,d,scores,runs,references,releases)),counts


class Feedback:
    """Select a coherent local preparation mode, and learn from actual completed offers."""
    def __init__(self,bank,*,online=True,threshold=THRESHOLD):
        self.state,self.delta,self.y,self.groups=bank[:4]
        self.tree=cKDTree(self.state/SCALE)
        self.past=deque(maxlen=640)
        self.clock=1+int(np.max(self.groups))
        self.online=online
        self.threshold=threshold
        self.last_key=None
        self.cpu=0.
        self.decisions=0
        self.updates=[0,0]

    def begin(self):
        self.last_key=None
        self.cpu=0.
        self.decisions=0
        self.updates=[0,0]

    def decide(self,x,baseline):
        """Return an observed/transported six-tick move, not an open-loop tape."""
        d,ix=self.tree.query(x/SCALE,min(84,len(self.state)))
        sample=self.delta[ix]
        reward=self.y[ix].astype(float)
        labels=self.groups[ix]
        if self.past:
            if not hasattr(self,"recent_tree"):
                self.recent_tree=cKDTree(np.stack([e[0] for e in self.past])/SCALE)
            dr,js=self.recent_tree.query(x/SCALE,min(50,len(self.past)))
            chosen=[self.past[i] for i in np.atleast_1d(js)]
            d=np.r_[d,np.maximum(0.,np.atleast_1d(dr)-.07)]
            sample=np.vstack((sample,[e[1] for e in chosen]))
            reward=np.r_[reward,[e[2] for e in chosen]]
            labels=np.r_[labels,[e[3] for e in chosen]]
        # Several contiguous observations from one rewarded movement are not independent evidence.
        ordered=np.argsort(d)
        _,j=np.unique(labels[ordered],return_index=True)
        use=ordered[np.sort(j)]
        use=use[d[use]<2.25][:26]
        if len(use)<4:
            return None,None
        d,sample,reward,labels=d[use],sample[use],reward[use],labels[use]
        ex=np.flatnonzero(reward)
        if len(ex)<2:
            return None,None
        chosen=ex[:10]
        candidates=np.vstack((baseline,sample[chosen]))
        sim=np.exp(-.5*np.mean(((candidates[:,None]-sample[None])/MOTION)**2,axis=2))
        w=np.exp(-.5*(d/.98)**2)
        supports=sim*w
        marginal=(1.75+w@reward)/(3.0+w.sum())
        scores=(1.8*marginal+supports@reward)/(1.8+supports.sum(axis=1))
        cost=.016*np.mean(((candidates-baseline)/MOTION)**2,axis=1)
        # The last mode biases close ties only; it cannot initiate action without local evidence.
        hysteresis=0. if self.last_key is None else .014*np.minimum(
            1.5,np.linalg.norm((candidates-self.last_key)/(.6+MOTION),axis=1))
        value=scores-cost-hysteresis
        good=np.r_[False,np.sum((sim[1:]>.64)&(reward[None]>0),axis=1)>=2]
        good=(good & (supports.sum(axis=1)>1.1))
        if not good.any():
            return None,None
        value[~good]=-np.inf
        k=int(np.argmax(value))
        delta=candidates[k]
        conf=float(scores[k]-scores[0]-cost[k])
        if conf<self.threshold:
            return None,None
        local=sim[k]>.66
        if np.any(local & (reward>0)):
            local &= reward>0
            # Mode preserving shrinkage. Never average two distant/reversed joint approaches.
            w0=supports[k,local]
            delta=.58*delta+.42*(w0@sample[local]/w0.sum())
        self.last_key=delta.copy()
        return delta,dict(estimated_gain=conf,stock_association=float(scores[0]),
                          proposal_association=float(scores[k]),independent_local=len(use),
                          nearest=float(d[0]),tag=int(labels[chosen[k-1]]))

    def heard(self,tape,reward):
        if not self.online:
            return
        if not tape:
            return
        started=process_time()
        # Use the already issued reference at k+6, only after the offer's actual score/fault.
        # A legally delivered attempt is guaranteed by the normal MPC reward/censoring hooks.
        op=[t for t,x,*_ in tape if x is not None and x[4]<-.9 and x[1]<-.77]
        if not op:
            return
        last=op[-1]
        rows={r[0]:r for r in tape}
        samples=[]
        for t,x,q,ref,sent in tape:
            if t+H not in rows or not(last-65<t<last-7 and t%3==0 and early(x)):
                continue
            future=rows[t+H]
            f=future[1]
            if f is None or f[4]>=-.9 or f[1]>=x[1]-.05 or x[1]-f[1]>.78:
                continue
            samples.append((np.r_[x,q[:5]],future[-1][:5]-sent[:5],float(reward),self.clock))
        if samples:
            self.past.extend(samples)
            self.clock+=1
            self.updates[int(reward)]+=1
            if hasattr(self,"recent_tree"):
                del self.recent_tree
        self.last_key=None
        self.cpu+=process_time()-started


class PreparationModel(mpc.MPC):
    def __init__(self,*args,prepare=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.prepare=prepare
        a=self.engine.twin
        axes=self.engine.actuators
        self.stride=a._joint_increments[axes]*self.engine.predict.micro
        self.lo=a._joint_low[axes]
        self.hi=a._joint_high[axes]
        self.early_notes=[]

    def begin(self):
        super().begin()
        self.early_notes=[]
        if self.prepare is not None:
            self.prepare.begin()

    def step(self,tick,eye,x,q,ref,base):
        original=super().step(tick,eye,x,q,ref,base)
        if self.prepare is None:
            return original
        if not early(x):
            self.prepare.last_key=None
            return original
        before=process_time()
        stock=np.clip((np.clip(original,self.lo,self.hi)-ref)/self.stride,-.8,.8)
        rollout=np.clip(original[:5]-ref[:5],-H*.8*self.stride[:5],H*.8*self.stride[:5])
        future,note=self.prepare.decide(np.r_[x,q[:5]],rollout)
        self.prepare.cpu+=process_time()-before
        if future is None:
            return original
        aim=np.clip(future/(4.1*self.stride[:5]),-.8,.8)
        # Change realized motor rates before saturation; the stock late controller is unchanged.
        w=.52*np.clip((x[1]-.06)/.28,0.,1.)
        rate=stock.copy()
        rate[:5]=(1-w)*rate[:5]+w*aim
        self.early_notes.append(dict(tick=tick,y=float(x[1]), max_rate_change=float(np.max(abs(rate-stock))),**note))
        self.prepare.decisions+=1
        return np.clip(ref+rate*self.stride,self.lo,self.hi)

    def missed(self,tape):
        if self.prepare is not None:
            self.prepare.heard(tape,False)


class Earned(Carried):
    def __init__(self,*args,prep=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.preparation=prep

    def keep(self,archive,leg):
        learned=super().keep(archive,leg)
        if self.preparation is not None:
            self.preparation.heard(archive,True)
        return learned


def assessed_bank(train,validation):
    a,meta=load_packets(train)
    b,checked=load_packets(validation)
    seen=set(a[-1])
    mask=np.array([p not in seen for p in b[-1]])
    if not mask.any():
        raise ValueError("No release-disjoint held motion")
    return a,tuple(v[mask] for v in b),meta,checked


def audit(args):
    train,test,meta,check=assessed_bank(PACKETS[:-1],PACKETS[-1:])
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,"local")
    a=mpc.arena()
    axes=a._arm_ids[1][2]
    stride=a._joint_increments[axes][:5]*a.control_dt/a.model.opt.timestep
    a.close()
    feedback=Feedback(train,online=False)
    home=np.asarray(_read(mpc.BANK.with_suffix(".json"))["ready"])[:5]
    win=[]; chosen=0
    for x,d,r,_,ref,release in zip(*test,strict=True):
        target,confidence=raw.get(x[:6],x[6:11])
        stock=home if target is None else home+confidence*(target[:5]-home)
        base=np.clip(stock-ref,-.8*H*stride,.8*H*stride)
        feedback.last_key=None
        p,note=feedback.decide(x,base)
        chosen+=p is not None
        if p is None:
            p=base
        query,ids=feedback.tree.query(x/SCALE,16)
        successes=np.asarray(ids)[feedback.y[ids]>0]
        nn=feedback.delta[successes[0]] if len(successes) else base
        # A future reference predicts the apprentice's possible motion. This is
        # descriptive off-policy fidelity, not a causal counterfactual return.
        distances=[np.linalg.norm(x-d) for x in (base,nn,p,np.zeros(5))]
        win.append([release,r,note is not None,*distances])
    mat=np.asarray(win)
    results={}
    for k,one in (("all",np.ones(len(mat),dtype=bool)),
                  ("rewarded",mat[:,1]>0),("rewarded_and_switched",(mat[:,1]>0)&(mat[:,2]>0))):
        j=mat[one]
        if not len(j):
            results[k]=dict(frames=0)
            continue
        means=np.array([j[j[:,0]==p,3:].mean(axis=0) for p in np.unique(j[:,0])])
        results[k]=dict(frames=len(j),releases=len(means),mean_by_release_rad=means.mean(axis=0).tolist(),
                        switched=int(j[:,2].sum()))
    _write(args.output,dict(train=meta,held=check,basis="release-disjoint earlier frame predicts six-tick issued displacement",
        columns=["stock_projection","nearest_rewarded","mode_preserved","stationary"],
        selected_frames=chosen,held_frames=len(mat),results=results))
    print("off_policy",results,flush=True)


def live(args):
    data,training=load_packets(PACKETS[:-1])
    deck=_read(args.plan)["assessment"][args.start:args.start+args.limit]
    held={c["id"] for c in deck}
    if held&set(data[-1]):
        raise ValueError("Early motion practice overlaps the evaluation releases")
    receipts=_read(mpc.BANK.with_suffix(".json"))
    ready=receipts["ready"]
    with np.load(mpc.BANK) as b: raw=mpc.DeepField(b,"local")
    launch=mpc.measured_launches(args.feeds)
    doc=dict(protocol="random",cap=args.cap,rows=[],cards=deck,training=training,
        base_prior=_hash(mpc.BANK),initial_motor_memory=_hash(SEED_PRACTICE),optics_sha256=_hash(OPTIC),
        setup="identical carried motor controller, learned directional rates only on opposing half",
        online=args.online,code_sha256=_hash(Path(__file__)))
    with mpc.centered(ready):
        a=mpc.arena()
        camera,rig,stride,lo,hi=mpc.resources(a)
        learner=Feedback(data,online=args.online)
        stock=None; modified=None
        try:
            for c in deck:
                variants=[]
                for key in (False,True):
                    # Controlled point start. The novel preparation is the only
                    # changed component, while reward updates remain real within the point.
                    if key and modified is not None: modified.close()
                    if not key and stock is not None: stock.close()
                    prep=learner if key else None
                    with patch.object(mpc,"Fragments",lambda f,e:Earned(
                            f,e,practice=[SEED_PRACTICE],held=held,prep=prep)):
                        controller=PreparationModel(mpc.BANK,raw,variant="elastic",prepare=prep)
                    if key: modified=controller
                    else: stock=controller
                    row=mpc.play(a,camera,rig,stride,lo,hi,ready,c,launch[c["id"]],raw,controller,
                                 OPTIC,"live",args.cap)
                    row["preparations"]=controller.early_notes
                    if prep is not None:
                        row["preparation_cpu"]=prep.cpu
                        row["new_early_outcomes"]=prep.updates.copy()
                        row["early_experience_rows"]=len(prep.past)
                    variants.append(row)
                left,right=variants
                if left["memory"]["before"]!=right["memory"]["before"]:
                    raise RuntimeError("Shared initial carried memory differs")
                fork=next((k for k,(p,q) in enumerate(zip(left["actions"],right["actions"])) if p!=q),None)
                if fork is not None:
                    true_prefix=[digest_json([e for e in r["events"] if e["step"]<=fork]) for r in (left,right)]
                    if true_prefix[0]!=true_prefix[1]:
                        raise RuntimeError("Matched points diverged before their first differing issued action")
                elif left["actions"]!=right["actions"] or digest_json(left["events"])!=digest_json(right["events"]):
                    raise RuntimeError("Unchanged action tape produced different physical events")
                doc["rows"].append(dict(card=c,stock=left,early=right,first_change=fork))
                _write(args.output,doc)
                print(c["id"],left["score"]["learner"],right["score"]["learner"],fork,
                      len(right["preparations"]),right["new_early_outcomes"],flush=True)
        finally:
            if stock is not None: stock.close()
            if modified is not None: modified.close()
            camera.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("audit","live"))
    p.add_argument("output",type=Path)
    p.add_argument("--plan",type=Path,default=COURSE)
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--start",type=int,default=9)
    p.add_argument("--limit",type=int,default=4)
    p.add_argument("--cap",type=int,default=24)
    p.add_argument("--online",action="store_true")
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    audit(a) if a.mode=="audit" else live(a)
