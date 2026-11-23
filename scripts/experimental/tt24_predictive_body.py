"""Fit predictive motor response and close the loop on public blade-pose effects."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .tt4_transfer_trial import _hash, _write
from .tt24_bounce_ident import _INC
from .tt24_world_arm import Grounding


OUT = Path('artifacts/tt24')
PLAY = OUT/'world-arm-public-play.npz'
MODEL = OUT/'predictive-body-fit.npz'
RECEIPT = OUT/'predictive-body-fit.json.gz'
HORIZONS = (1, 2, 3, 5, 7)
FUTURE = 3
RIDGES = (.025, .12, .6, 2.5)
WEIGHTS = np.array((.22, .40, .64, .92, 1.))
FIELD_COUNT = 26


def inputs(q, issued, velocity, previous_velocity, last_rate, center):
    return np.concatenate((np.ones((*q.shape[:-1],1)), (q-center)/.30, (issued-q)/_INC,
                           velocity/_INC, (velocity-previous_velocity)/_INC, last_rate),axis=-1)


def trajectories(path):
    with np.load(path,allow_pickle=False) as archive:
        readings=np.asarray(archive['joints'],dtype=float)
        commands=np.asarray(archive['command'],dtype=float)
        fit_count=int(archive['train_runs'])
        if not np.isclose(float(archive['dt']),1./60.) or readings.shape!=commands.shape:
            raise ValueError('each public measurement must precede its recorded 60 Hz joint rate')
        if fit_count!=6 or readings.shape[1:]!=(144,5) or len(readings)<=fit_count:
            raise ValueError('expected six 144-frame fit clips and separate diagnostic clips')
    return list(zip(readings,commands,strict=True)),fit_count


def design(row, horizon, center):
    q,u=row
    t=np.arange(2,len(q)-horizon)
    phase=np.vstack((np.zeros(5),np.cumsum(u[:-1],axis=0)))
    issued=q[0]+_INC*phase
    velocity=q[t]-q[t-1]
    earlier=q[t-1]-q[t-2]
    state=inputs(q[t],issued[t],velocity,earlier,u[t-1],center)
    ahead=u[t[:,None]+np.arange(horizon)].reshape(len(t),horizon*5)
    return np.column_stack((state,ahead)),(q[t+horizon]-q[t])/_INC


def fit_response(rows,ridge):
    center=np.vstack([row[0] for row in rows]).mean(axis=0)
    weights={}
    for horizon in HORIZONS:
        matrices,targets=zip(*(design(row,horizon,center) for row in rows),strict=True)
        A,B=np.vstack(matrices),np.vstack(targets)
        penalty=np.eye(A.shape[1])*ridge
        penalty[0,0]*=.05
        weights[horizon]=np.linalg.solve(A.T@A+penalty,A.T@B)
    return center,weights


def error(model,rows):
    center,weights=model
    result={}
    for horizon,coeff in weights.items():
        X,Y=zip(*(design(row,horizon,center) for row in rows),strict=True)
        truth=np.vstack(Y)
        delta=_INC*(np.vstack(X)@coeff-truth)
        result[str(horizon)]=dict(rms_rad=float(np.sqrt(np.mean(delta**2))),
                                  reference_rms_rad=float(np.sqrt(np.mean((_INC*truth)**2))),
                                  samples=len(delta))
    return result


def identify(source,checkpoint,receipt):
    rows,count=trajectories(source)
    train,blind=rows[:count],rows[count:]
    candidates=[]
    for ridge in RIDGES:
        folds=[error(fit_response(train[:n]+train[n+1:],ridge),[row]) for n,row in enumerate(train)]
        score=np.mean([r['rms_rad'] for group in folds for r in group.values()])
        candidates.append(dict(ridge=ridge,cross_validated_rad=float(score)))
    ridge=min(candidates,key=lambda r:r['cross_validated_rad'])['ridge']
    model=fit_response(train,ridge)
    center,weights=model
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(checkpoint,center=center,**{f'h{h}':w for h,w in weights.items()})
    data=dict(schema='tt24-joint-response-multi-horizon-1',source_sha256=_hash(source),
              checkpoint_sha256=_hash(checkpoint),used_fields=['joints','command','dt','train_runs'],
              acquisition_clips=count,acquisition_frames=sum(len(q) for q,_ in train),chosen_ridge=ridge,
              candidates=candidates,withheld_clip_forecasts=error(model,blind))
    _write(receipt,data)
    return data


class PredictiveGrounding(Grounding):
    """Use a frozen empirical motor response to set each actual actuator rate."""

    def __init__(self,decoder,home,path=MODEL):
        super().__init__(decoder,home)
        with np.load(path,allow_pickle=False) as model:
            self.center=model['center']
            self.coefficients={h:model[f'h{h}'] for h in HORIZONS}
        self.previous=self.home.copy()
        self.velocity=np.zeros(5)
        self.last_rate=np.zeros(5)
        self.difference=np.eye(FUTURE*5)
        self.difference[5:,:-5]-=np.eye((FUTURE-1)*5)

    def target(self,effect,obs):
        q=np.asarray(obs['joints'],dtype=float)
        rate=q-self.previous
        state=inputs(q,self.issued,rate,self.velocity,self.last_rate,self.center)
        present=self.pose(q)
        desired=np.r_[self.origin[:3]+np.clip(effect[:3],-.085,.085),np.clip(effect[3:],-.70,.70)]
        error=(desired-present)/self.scale
        world=(self.jacobian(q)/self.scale[:,None])*_INC
        blocks,targets=[],[]
        for h,weight in zip(HORIZONS,WEIGHTS,strict=True):
            table=self.coefficients[h]
            expected=world@(state@table[:FIELD_COUNT])
            effect_matrix=np.zeros((5,FUTURE*5))
            n=min(h,FUTURE)*5
            effect_matrix[:,:n]=world@table[FIELD_COUNT:FIELD_COUNT+n].T
            blocks.append(weight*effect_matrix)
            targets.append(weight*(error-expected))
        # Rate effort and smoothness regularize all future hypothetical actions.
        size=FUTURE*5
        blocks.extend((.125*np.eye(size),.095*self.difference))
        targets.extend((np.zeros(size),.095*np.r_[self.last_rate,np.zeros(size-5)]))
        lower,upper=np.full(size,-.75),np.full(size,.75)
        lower[:5]=np.maximum(lower[:5],(self.home-.55-self.issued)/_INC)
        upper[:5]=np.minimum(upper[:5],(self.home+.55-self.issued)/_INC)
        action=lsq_linear(np.vstack(blocks),np.concatenate(targets),bounds=(lower,upper),
                          method='bvls',tol=.0004,max_iter=18).x[:5]
        self.issued+=_INC*action
        self.previous,self.velocity,self.last_rate=q.copy(),rate,action.copy()
        return action


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=PLAY)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--receipt',type=Path,default=RECEIPT)
    args=parser.parse_args()
    print(identify(args.source,args.model,args.receipt),flush=True)
