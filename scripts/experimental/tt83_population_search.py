"""Population search over smooth direct joint trajectories in the public optical twin."""
from collections import Counter

import numpy as np

from . import tt57_motor_school as school
from . import tt59_fewshot_joints as few
from . import tt74_sensitivity_bank as sensitivity
from .tt4_transfer_trial import _read

mpc = school.mpc
H = school.H
POP = (36,36,32)
RANK = 13
FEATS = H*5
SEARCH_LIMIT = 2
RNG = 830913


class MotorPrior:
    """Observed successful joint-time correlations, not outcome gradients."""
    def __init__(self, motor=sensitivity.MODEL, lessons=few.MODEL):
        readout=_read(motor)
        roots=[np.asarray(x["entry"]["shape"])[:,:5] for x in readout["observations"] if "entry" in x]
        with np.load(lessons) as f:
            earned=(f["weights"] @ f["basis"]).reshape(-1,H,6)[:,:,:5]
        wave=np.array(roots)
        sample=np.concatenate((wave-wave.mean(0),earned))
        # Normalize per motion so nearly stationary habits don't disappear.
        flat=sample.reshape(-1,FEATS)
        flat=flat/np.maximum(.15,np.linalg.norm(flat,axis=1,keepdims=True))
        _,values,v=np.linalg.svd(flat,full_matrices=False)
        elapsed=(np.arange(H)+1)/H
        linear=np.minimum(1.,elapsed*3.5)
        smooth=[linear,linear*(2*elapsed-1), np.exp(-.5*((elapsed-.42)/.17)**2)*linear]
        fallback=np.asarray([g[:,None]*a for g in smooth for a in np.eye(5)]).reshape(-1,FEATS)
        proposed=np.vstack((v[:9],fallback))
        # Gram-Schmidt preserves order of the learned correlations while
        # giving every joint an explicit independent exploration direction.
        basis=[]
        for x in proposed:
            q=x.copy()
            for b in basis:
                q-=b*np.dot(b,q)
            mag=np.linalg.norm(q)
            if mag>1e-7:
                basis.append(q/mag)
            if len(basis)==RANK:
                break
        self.basis=np.asarray(basis)
        self.spread=np.r_[np.clip(values[:9]/values[0],.44,1.),[.58]*(RANK-9)]
        self.learned_fraction=float(np.sum(values[:9]**2)/np.sum(values**2))

    def decode(self, z):
        p=(np.asarray(z)@self.basis).reshape(-1,H,5)
        return np.c_[p[0],np.zeros(H)] if p.shape[0]==1 else np.pad(p,((0,0),(0,0),(0,1)))

    def encode(self, shapes):
        return np.asarray(shapes)[...,:5].reshape(-1,FEATS)@self.basis.T


def _project_path(path,ref,lo,hi,stride):
    # Clamp the actual command sequence before using distances or elites.
    target=np.clip(ref+path,lo,hi)
    prev=ref.copy()
    output=[]
    for r in target:
        prev=np.clip(prev+np.clip((r-prev)/stride,-.8,.8)*stride,lo,hi)
        output.append(prev-ref)
    return np.asarray(output)


def describe(predict, paths, ref):
    g=ref+np.pad(np.asarray(paths),((0,0),(0,school.T-H),(0,0)),mode="edge")
    f=np.zeros((len(g),9))
    touch=np.full((len(g),19),np.nan)
    u,y=predict.trajectories(g,school.AIM,(H+1)*mpc.DT,feedback=f,approaches=touch)
    loss,legal=school.score(y,f)
    # Choose a smooth receiving objective for no contacts, and a playable
    # ball-plus-margin objective for separated hits. Prefer legal returns.
    # Noncontact candidates get a directional rim/plane margin, not a
    # binary zero or a loss trained on exact held native ball position.
    gap=touch[:,1:4]-touch[:,4:7]
    normal=touch[:,7:10]
    plane=np.abs(np.einsum("ni,ni->n",gap,normal))
    radial=np.sqrt(np.maximum(0.,np.sum(gap*gap,axis=1)-plane**2))
    finite=np.isfinite(plane)&np.isfinite(radial)
    reach=np.where(finite,4.0*np.minimum(radial,.45)+1.4*np.minimum(plane,.36),0.)
    score=np.where(legal,loss+5.*np.maximum(y[:,12]-.051,0.),
                   np.where(f[:,1]>.5,loss+6.8*np.maximum(f[:,8]-.048,0.),loss+reach))
    return u[:,:H],y,f,touch,score,legal


def _unique(paths, threshold=.017, n=8):
    out=[]
    for p in paths:
        if all(np.sqrt(np.mean((p-q)**2))>=threshold for q in out):
            out.append(p)
        if len(out)>=n:
            break
    return out


def stock_anchors(raw,engine,ready,x,q,ref,base,extra,*,bank=None):
    nominal=few.anchor_paths(engine,raw,ready,x,q,ref,base)
    active=[] if bank is None else bank.suggest(x,q,ref)[0]
    seeds=[*extra,*active[:3],*nominal,np.zeros((H,6))]
    return _unique(seeds, n=9)


def _elite(paths,scores,feedback,n=7):
    order=list(np.argsort(scores,kind="stable"))
    noncontact=np.flatnonzero(feedback[:,1]<.5)
    strong=np.flatnonzero(feedback[:,1]>.5)
    frontier=[]
    for subset in (strong,noncontact):
        if len(subset):
            frontier.append(int(subset[np.argmin(scores[subset])]))
    # Protect up to two genuinely different nearest approaches. This avoids
    # spending the whole population polishing one unlucky rim contact.
    keep=[]
    for j in list(dict.fromkeys(frontier+order)):
        if all(np.sqrt(np.mean((paths[j]-paths[k])**2))>=.018 for k in keep):
            keep.append(int(j))
        if len(keep)>=n:
            break
    return keep


def _end_transport(geometry, encounter, center, fraction):
    if not np.isfinite(encounter).all() or not 3*mpc.DT < encounter[0] < (H-.5)*mpc.DT:
        return None
    ball,at=encounter[1:4],encounter[10:15]
    pose,n,j,a=geometry.pose(at,jacobian=True)
    d=ball-pose+n*.017
    if np.linalg.norm(d)>.28:
        return None
    A=np.r_[j,.075*a]
    v=np.r_[d,[0.,0.,0.]]
    derivative=A.T@np.linalg.solve(A@A.T+.037**2*np.eye(6),v)
    horizon=max(3.,encounter[0]/mpc.DT*.65)
    eta=np.clip((1+np.arange(H))/horizon,0.,1.)
    smooth=(eta*eta*(3.-2*eta))[:,None]
    return center+smooth*np.r_[np.clip(derivative,-.20,.20)*fraction,0.]


class Population:
    def __init__(self,prior,lo,hi,stride,*,rounds=POP):
        self.prior=prior
        self.lo,self.hi,self.stride=lo,hi,stride
        self.rounds=rounds

    def make(self, ref, paths):
        return np.asarray([_project_path(np.clip(p,-.65,.65),ref,self.lo,self.hi,self.stride) for p in paths])

    def search(self, engine, geometry, roots, ref, rng, mode):
        roots=np.asarray(roots)
        p0=roots.copy()
        assert len(roots)
        metrics=[]
        histories=[]
        unique_legal=set()
        center=roots.copy()
        cov=None
        for i,n in enumerate(self.rounds):
            # Identical first round and exactly identical twin budgets.
            # Later independent rounds in flat mode still consume the same
            # amount of compute and all retain legal candidates.
            adapted=i>0 and mode=="iterative"
            if adapted:
                paths,scores,feedback,encounters,_=histories[-1]
                order=_elite(paths,scores,feedback)
                center=paths[order]
                candidates=list(center[:min(4,len(center))])
                for j in order[:4]:
                    for gain in (.64,1.02):
                        if (root:=_end_transport(geometry,encounters[j],paths[j],gain)) is not None:
                            candidates.append(root)
                # Use rank-weighted previous directions around the learned
                # correlated action basis. Shrink covariance to retain
                # unnurtured movements and avoid moving all the islands.
                rank=np.argsort(scores,kind="stable")[:max(6,len(paths)//3)]
                local=paths[rank]
                near=np.argmin(np.mean((local[:,None]-prior_centers[None])**2,axis=(2,3)),axis=1)
                z=self.prior.encode(local-prior_centers[near])
                ability=self.prior.spread
                scale=np.diag((.24*ability)**2)
                cov=(z.T@z/max(1,len(z)))*.41+.59*scale
                eig,axis=np.linalg.eigh(cov)
                chol=axis@np.diag(np.sqrt(np.clip(eig,.053**2,.44**2)))@axis.T
            else:
                center=roots
                candidates=list(roots) if i==0 else []
                chol=np.diag(.27*self.prior.spread)
            prior_centers=center
            fill=n-len(candidates)
            g=rng.normal(size=(fill,RANK))@chol.T
            if adapted:
                g*=np.where(np.arange(fill)[:,None]%5==0,1.48,.94)
            elif i:
                g*=np.where(np.arange(fill)[:,None]%4==0,1.85,1.05)
            noises=self.prior.decode(g)
            if noises.ndim==2:
                noises=noises[None]
            candidates.extend(center[k%len(center)]+delta for k,delta in enumerate(noises))
            proposals=np.asarray(candidates)
            closest=np.argmin(np.mean((proposals[:,None]-p0[None])**2,axis=(2,3)),axis=1)
            offset=np.clip(proposals-p0[closest],-.32,.32)
            radius=np.sqrt(np.mean(offset**2,axis=(1,2)))
            proposals=p0[closest]+offset*np.minimum(1.,.135/np.maximum(radius,1e-9))[:,None,None]
            unit=self.make(ref,proposals)
            rates,value,f,at,score,legal=describe(engine.predict,unit,ref)
            signatures=[np.round(u,6).tobytes() for u in rates]
            new={signatures[j] for j in np.flatnonzero(legal)}-unique_legal
            unique_legal.update(new)
            metrics.append(dict(candidates=len(unit),legal=int(legal.sum()),new_legal=len(new),
                 separated=int(np.sum(f[:,1]>.5)),minimum_separation_m=float(np.min(f[:,0])),best=float(np.min(score))))
            histories.append((unit,score,f,at,signatures))
        paths=np.concatenate([a[0] for a in histories])
        score=np.concatenate([a[1] for a in histories])
        feedback=np.concatenate([a[2] for a in histories])
        elite=_elite(paths,score,feedback)
        # Recheck exactly nine choices for every optimizer, including
        # independent public translation/spin uncertainty offsets.
        sign=[b for generation in histories for b in generation[4]]
        sort=list(np.argsort(score,kind="stable"))
        best_idx, seen = [],set()
        for j in list(dict.fromkeys([*sort[:6],*elite[:3],*sort])):
            if sign[j] not in seen:
                best_idx.append(j)
                seen.add(sign[j])
            if len(best_idx)>=9:
                break
        # If a fully constrained pose leaves less than nine distinct
        # tapes, duplicate evaluation fills the same matched audit budget.
        if len(best_idx)<9:
            available=[j for j in sort if j not in best_idx]
            best_idx.extend(available[:9-len(best_idx)])
        chosen=paths[best_idx]
        codes=[dict(trial=int(j),origin_round=int(np.searchsorted(np.cumsum(self.rounds),j,side="right"))) for j in best_idx]
        ranked,diag=few.rank_paths(engine,chosen,ref,codes)
        result=dict(rounds=metrics,screened=int(sum(k["candidates"] for k in metrics)),legal=sum(k["legal"] for k in metrics),
                    audit=diag,unique_legal=len(unique_legal),issued=False)
        secure=[(u,d) for u,d in ranked if d["legal"] and d["robust"]>0]
        if not secure:
            # A robust-capable planning failure is exposed for audit. Do
            # not put this unverified action on the expected live tape.
            best=None
        else:
            best,d=min(secure,key=lambda z:z[1]["score"])
            result.update(issued=True,decision=d)
        return best,[ref+paths[k] for k in elite],result


class Player(school.season.Player):
    def __init__(self,*args,population,bank,mode,**kw):
        super().__init__(*args,**kw)
        self.population=population
        self.transport_bank=bank
        self.mode=mode
        self._capture=None

    def begin(self):
        super().begin()
        self._capture=None
        self.expanded=[]
        self.warm=[]
        self.warm_tick=-1000
        self.windows=0
        self.attempted=0
        self.bookkeeping=Counter()

    def trial(self,goals,*,long=False,traces=False):
        result=super().trial(goals,long=long,traces=traces)
        if self._capture is not None and not long:
            losses,legal,*_=result
            for k in np.argsort(losses,kind="stable")[:3]:
                goal=mpc.tail(np.asarray(goals[k]),H)
                self._capture[goal.tobytes()]=(float(losses[k]),goal)
        return result

    def search(self,tick,x,q,ref,base):
        likely=self.mode!="stock" and self.attempted<SEARCH_LIMIT and -1.15<x[1]<-.53 and x[4]<-1.
        self._capture={} if likely else None
        try:
            super().search(tick,x,q,ref,base)
        finally:
            saved,self._capture=self._capture,None
        if not likely or not self.notes or self.notes[-1]["tick"]!=tick:
            return
        if self.notes[-1].get("robust",0):
            self.bookkeeping["kept-stock"]+=1
            return
        if not self.engine.sync():
            return
        self.attempted+=1
        local=[]
        if self.warm and 0<tick-self.warm_tick<9:
            local=[mpc.tail(r[tick-self.warm_tick:],H)-ref for r in self.warm[:4]]
        learned=[r[1]-ref for r in sorted((saved or {}).values(),key=lambda z:z[0])[:5]]
        roots=stock_anchors(self.raw,self.engine,self.priors.home,x,q,ref,base,[*learned,*local],
                            bank=self.transport_bank)
        salt=int(tick+1000*self.attempted)
        rng=np.random.default_rng([RNG,salt])
        action,warm,note=self.population.search(self.engine,self.geometry,roots,ref,rng,self.mode)
        self.warm,self.warm_tick=warm,tick
        self.expanded.append(dict(tick=tick,x=x.tolist(),roots=len(roots),warm_roots=len(local),**note))
        if action is not None:
            self.best=school.atlas.reference_path(ref,action,self.population.stride,self.population.lo,self.population.hi)[1:]
            self.tick=tick
            self.lease=mpc.LEASE+3

    def step(self,tick,eye,x,q,ref,base):
        if x is not None and x[4]>.9:
            self.attempted=0
            self.warm=[]
        return super().step(tick,eye,x,q,ref,base)
