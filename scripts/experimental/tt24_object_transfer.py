"""Learn which moving proposal predicts continuation of a native Pong point."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .tt4_transfer_trial import _hash, _read, _write

OUT=Path('artifacts/tt24')
MODEL=OUT/'object-transfer-source.npz'
SOURCE_RECEIPT=OUT/'object-transfer-source.json.gz'
PLAN=OUT/'object-transfer-plan.json'
SOURCE_SEED=73036125
SOURCE_SPLIT=(48,16)
WINDOW=9
CONTACT_RISE=.055


def encounter(ball,blade,normal,span,radius):
    """Use dimensionless motion and support in the controlled object's frame."""
    if len(ball)<WINDOW:
        return None
    ball=np.asarray(ball[-WINDOW:]).copy()
    blade=np.asarray(blade[-WINDOW:])
    valid=np.isfinite(ball).all(axis=1)
    if valid.sum()<7 or not np.isfinite(blade).all():
        return None
    t=np.arange(WINDOW,dtype=float)
    for axis in range(ball.shape[1]):
        ball[:,axis]=np.interp(t,t[valid],ball[valid,axis])
    rel=(ball-blade)/span
    signed=rel@normal-radius/span
    tangent=np.linalg.norm(rel-(rel@normal)[:,None]*normal,axis=1)
    near=int(np.argmin(np.abs(signed[2:7])))+2
    if np.min(np.abs(signed[2:7]))>1.55 or signed[0] < -.55:
        return None
    incoming=-np.polyfit(t[:4],signed[:4],1)[0]
    outgoing=np.polyfit(t[-4:],signed[-4:],1)[0]
    if incoming<=.025:
        return None
    gain=np.clip((outgoing+incoming)/max(incoming,.05),-2.,4.)
    reversal=np.clip(outgoing/max(incoming,.05),-2.5,3.5)
    alignment=np.exp(-.5*(tangent[near]/1.05)**2)
    depth=np.exp(-.5*(signed[near]/.35)**2)
    own_projected=((blade[-1]-blade[0])@normal)/span
    shift=np.linalg.norm(blade[-1]-blade[0])/span
    x=np.r_[reversal,gain,alignment,depth,reversal*alignment,reversal*depth,
            reversal*alignment*depth,min(tangent[near],3.5),np.clip(signed[-1],-2.,2.),
            np.clip(own_projected,-1.,1.),min(shift,2.)]
    return x


def read_features(agent,outcome):
    blade,ball=agent.path()
    if len(blade)<27 or agent.motor.effects is None:
        return []
    from .tt24_source_native import DT, NATIVE_HZ
    span=agent.scene.extended[agent.motor.identity][1]
    tangent=agent.motor.axis
    normal=np.array((-tangent[1],tangent[0]))
    if np.sum((.5-np.nanmedian(blade,axis=0))*normal)<0:
        normal=-normal
    duration=outcome['ticks']/NATIVE_HZ
    rows=[]
    for end in range(20,len(blade),2):
        x=encounter(ball[end-WINDOW+1:end+1],blade[end-WINDOW+1:end+1],normal,span,.013)
        if x is None:
            continue
        # Only elapsed public point duration provides the source supervision.
        survives=duration-end*DT>=.28
        rows.append(dict(frame=end,features=x.tolist(),continued=bool(survives)))
    return rows


def sigmoid_fit(X,y,ridge):
    Z=np.column_stack((np.ones(len(X)),X))
    counts=np.bincount(y.astype(int),minlength=2)
    if min(counts)<4:
        raise ValueError('native duration supplied too few outcome events')
    balanced=len(y)/(2*counts[y])

    def objective(w):
        pred=Z@w
        likelihood=np.sum(balanced*(np.logaddexp(0.,pred)-y*pred))
        grad=Z.T@(balanced*(expit(pred)-y))
        return likelihood+.5*ridge*(w@w),grad+ridge*w

    return minimize(objective,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B').x


def observe_source(args):
    import torch
    from .tt3_pong import cases_from_seeds,_native_rallies
    from .tt24_source_native import Collector,VisualConsequence,CHECKPOINT,HELD,OPPONENT_SPEED,CAP,batches

    torch.set_num_threads(1)
    archive=_read(CHECKPOINT)
    horizon=_read(HELD)['horizon']
    actor=VisualConsequence.from_checkpoint('pong',archive)
    seeds=np.random.default_rng(SOURCE_SEED).choice(100_000_000,sum(SOURCE_SPLIT),replace=False).tolist()
    rows=[]
    for group in batches(cases_from_seeds(seeds),16):
        agents=[Collector(case.exploration,actor,archive['plant'],horizon) for case in group]
        native=_native_rallies(group,agents,max_ticks=CAP,opponent_speed=OPPONENT_SPEED)
        for case,agent,outcome in zip(group,agents,native,strict=True):
            rows.append(dict(seed=case.serve,ticks=outcome['ticks'],rgb_frames=len(agent.views),
                             motor_sha256=outcome['action_sha256'],optical_sha256=agent.scene.signature.hexdigest(),
                             windows=read_features(agent,outcome)))
        print('native Pong object windows',len(rows),sum(len(row['windows']) for row in rows),flush=True)

    def xy(subset):
        selected=[w for row in subset for w in row['windows']]
        return (np.asarray([w['features'] for w in selected]),np.asarray([int(w['continued']) for w in selected]))

    X,y=xy(rows[:SOURCE_SPLIT[0]])
    H,k=xy(rows[SOURCE_SPLIT[0]:])
    center=X.mean(0)
    scale=np.maximum(X.std(0),.14)
    Z=np.clip((X-center)/scale,-5.,5.)
    shuffled=np.random.default_rng(73076155).permutation(y)
    fits={mode:sigmoid_fit(Z,target,6.) for mode,target in [('pong',y),('shuffled',shuffled)]}
    archive_test=np.column_stack((np.ones(len(H)),np.clip((H-center)/scale,-5.,5.)))
    held={mode:{'balanced_accuracy':float(np.mean([np.mean(((archive_test@w)>=0)[k==label]==bool(label))
                                                  for label in (0,1)])),
                'brier':float(np.mean((expit(archive_test@w)-k)**2))}
          for mode,w in fits.items()}
    np.savez_compressed(args.model,center=center,scale=scale,**fits)
    receipt=dict(schema='tt24-native-pong-object-continuation-1',source_actor_sha256=_hash(CHECKPOINT),
                 weights_sha256=_hash(args.model),train_episodes=SOURCE_SPLIT[0],held_episodes=SOURCE_SPLIT[1],
                 train_windows=len(y),train_continued=int(y.sum()),held_windows=len(k),held_continued=int(k.sum()),
                 labels='elapsed real native point duration, no contact or object identity labels',
                 held=held,rows=rows)
    _write(args.receipt,receipt)
    print({key:value for key,value in receipt.items() if key!='rows'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--receipt',type=Path,default=SOURCE_RECEIPT)
    observe_source(parser.parse_args())
