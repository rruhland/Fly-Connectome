"""Fit a fixed reservoir of causal motor windows acquired during native Pong."""

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.linalg import expm
import torch

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE,CarriedWristPolicy
from .tt24_continual_wrist import ContinualWrist
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import WARM,arcade
from .tt24_learned_tip_transfer import NativeTip
from .tt24_motor_bank import I,STEP,MotorBank,decoder
from .tt24_online_tip_transfer import pass_through
from .tt24_pong_arm import NATIVE_CAP,historical_retention
from .tt24_pong_arm_local import ALPHAS,WINDOW,regress,robust_error
from .tt24_pong_arm_tip import Joystick,plan
from .tt4_transfer_trial import _hash,_read,_write


SOURCE = ROOT/'online-tip-reservoir-source.json.gz'
TARGET = ROOT/'online-tip-reservoir-held.json.gz'
CONTINUITY = ROOT/'online-tip-reservoir-continuity.json.gz'
POINTS = 12
PER_POINT = 20
CAPACITY = POINTS*PER_POINT
MODES = ('reservoir','stored_batch','stale')
TARGET_RNG = 784980363


class CausalWindow:
    """A fixed history shared by calibration and independently measured validation."""

    def begin(self,encoder):
        self.track = np.zeros(5)
        self.filtered = np.zeros((len(ALPHAS),5))
        self.travel = np.zeros(5)
        self.previous = np.asarray(encoder,float).copy()
        self.history = deque([(self.filtered.copy(),self.previous,self.travel.copy())],maxlen=WINDOW+1)
        self.alphas = np.array(ALPHAS)[:,None]

    def observe(self,request,encoder):
        measured = np.asarray(encoder,float)
        self.track += STEP*np.asarray(request)
        self.filtered += self.alphas*(self.track-self.filtered)
        self.travel += abs(measured-self.previous)
        self.previous = measured.copy()
        self.history.append((self.filtered.copy(),self.previous,self.travel.copy()))
        if len(self.history)<=WINDOW:
            return
        prior,earlier,distance = self.history[0]
        x = self.filtered-prior
        y = measured-earlier
        present = ((self.travel-distance)[None,:]>=.00065)&(np.linalg.norm(x,axis=1)[:,None]>.00055)
        self.consume(x,y,present)


class Reservoir(CausalWindow):
    """Uniform sampling within each independently started point."""

    def __init__(self,seed):
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros((POINTS,PER_POINT,len(ALPHAS),5))
        self.y = np.zeros((POINTS,PER_POINT,5))
        self.activity = np.zeros(self.x.shape,dtype=bool)
        self.seen = np.zeros(POINTS,dtype=np.int64)
        self.filled = np.zeros(POINTS,dtype=np.int64)
        self.point = -1

    def begin(self,joints):
        self.point += 1
        slot = self.point % POINTS
        self.seen[slot] = 0
        self.filled[slot] = 0
        self.x[slot].fill(0.)
        self.y[slot].fill(0.)
        self.activity[slot].fill(False)
        super().begin(joints)

    def consume(self,x,y,present):
        if not np.any(present):
            return
        group = self.point % POINTS
        self.seen[group] += 1
        count = int(self.seen[group])
        slot = count-1 if count<=PER_POINT else int(self.rng.integers(count))
        if slot>=PER_POINT:
            return
        self.x[group,slot] = x
        self.y[group,slot] = y
        self.activity[group,slot] = present
        self.filled[group] = min(count,PER_POINT)

    def points(self):
        return range(max(0,self.point+1-POINTS),self.point+1)

    def examples(self,lag):
        examples = []
        for epoch in self.points():
            i = epoch % POINTS
            size = self.filled[i]
            if size:
                examples.append((self.x[i,:size,lag],self.y[i,:size],self.activity[i,:size,lag]))
        return examples

    @property
    def retained_numeric_bytes(self):
        persistent = sum(getattr(self,name).nbytes for name in ('x','y','activity','seen','filled'))
        live_ring = 8*((WINDOW+1)*(len(ALPHAS)*5+10)+len(ALPHAS)*6+15)
        return persistent+live_ring

    def memory(self):
        return {f'reservoir_{name}':getattr(self,name).copy() for name in ('x','y','activity','seen','filled')} | {
            'reservoir_point':np.array(self.point), 'reservoir_rng':np.array(json.dumps(self.rng.bit_generator.state))}

    @classmethod
    def load(cls,path):
        learner = cls(0)
        with np.load(path,allow_pickle=False) as saved:
            for name in ('x','y','activity','seen','filled'):
                setattr(learner,name,saved[f'reservoir_{name}'].copy())
            learner.rng.bit_generator.state=json.loads(str(saved['reservoir_rng']))
            learner.point=int(saved['reservoir_point'])
        return learner


class HeldPrediction(CausalWindow):
    """Score fixed regressions without updating them or retaining public raw episodes."""

    def __init__(self,fit):
        self.selection = [ALPHAS.index(row['alpha']) for row in fit]
        self.weights = np.asarray([row['matrix_row'] for row in fit])
        self.errors = [[] for _ in range(5)]
        self.original = [[] for _ in range(5)]
        self.grams = np.zeros((5,5,5))

    def consume(self,x,y,present):
        for index,lag in enumerate(self.selection):
            if present[lag,index]:
                row = x[lag]
                self.errors[index].append(abs(float(y[index]-row@self.weights[index])))
                self.original[index].append(abs(float(y[index]-row[index])))
                self.grams[index] += np.outer(row,row)

    def evidence(self):
        rows = []
        for errors,stale,gram in zip(self.errors,self.original,self.grams,strict=True):
            sing=np.sqrt(np.maximum(0.,np.linalg.eigvalsh(gram)))[::-1]
            rows.append(dict(p70_rad=float(np.quantile(errors,.70)),stale_p70_rad=float(np.quantile(stale,.70)),
                             samples=len(errors),singular_values=sing.tolist()))
        return rows


class LivePaddle(NativeTip):
    """Consume and discard each live command and encoder event after hashing it."""

    def __init__(self,arena,joystick,pose,wiring,spec,sink):
        super().__init__(arena,joystick,pose,wiring,spec)
        self.sink = sink
        self.digest = sha256()
        self.frames = 0
        begin = np.asarray(self.traces.pop(),dtype='<f8')
        self.digest.update(begin.tobytes())
        if sink is not None:
            sink.begin(begin)

    def step(self,drive):
        super().step(drive)
        if self.commands:
            requested = np.asarray(self.commands.pop(),dtype='<f8')
            measured = np.asarray(self.traces.pop(),dtype='<f8')
            self.digest.update(requested.tobytes())
            self.digest.update(measured.tobytes())
            if self.sink is not None:
                self.sink.observe(requested,measured)
            self.frames += 1


def native_point(arena,joystick,seed,pose,wiring,weights,sink=None):
    paddle = None
    def attach(spec):
        nonlocal paddle
        paddle=LivePaddle(arena,joystick,pose,wiring,spec,sink)
        return paddle
    outcome=arcade(weights,[seed],cap=NATIVE_CAP,opponent=.35,paddle_factory=attach)[0,0]
    return dict(seed=seed,score=int(outcome[0]),returns=int(outcome[1]),ticks=int(outcome[2]),
                physical_frames=paddle.frames,public_events_sha256=paddle.digest.hexdigest())


def physical_setup(index):
    """Reproduce the two evaluator plants without opening historical source clips."""
    random = np.random.default_rng(29845128+index)
    skew = random.normal(size=(5,5))
    skew -= skew.T.copy()
    return expm(skew*((.80+.2*index)/np.linalg.norm(skew,2)))@np.diag(random.uniform(.60,1.14,5))


def robust_fit(reservoir):
    chosen = [None]*5
    for lag,alpha in enumerate(ALPHAS):
        examples=reservoir.examples(lag)
        folds=(examples[::2],examples[1::2])
        for joint in range(5):
            loss=float(np.mean([robust_error(regress(a,joint),b,joint) for a,b in (folds,folds[::-1])]))
            if chosen[joint] is None or loss<chosen[joint]['p70_source_cv']:
                chosen[joint]=dict(lag=lag,alpha=alpha,p70_source_cv=loss)
    for joint,item in enumerate(chosen):
        item['matrix_row']=regress(reservoir.examples(item['lag']),joint).tolist()
    return chosen


def assert_preserved(reservoir,path):
    rebuilt=Reservoir.load(path)
    if any(not np.array_equal(value,rebuilt.memory()[key]) for key,value in reservoir.memory().items()):
        raise ValueError('bounded replay memory changed when saving/loading')
    return rebuilt


class SplitPublicStream:
    def __init__(self,*sinks):
        self.sinks = sinks

    def begin(self,joints):
        for sink in self.sinks:
            sink.begin(joints)
            if isinstance(sink,Reservoir):
                slot = sink.point % POINTS
                if (sink.seen[slot] or sink.filled[slot] or sink.activity[slot].any()
                        or sink.x[slot].any() or sink.y[slot].any()):
                    raise ValueError('new source point inherited stale circular-buffer values')

    def observe(self,requested,encoder):
        for sink in self.sinks:
            sink.observe(requested,encoder)


def same_memory(a,b):
    right=b.memory()
    return all(np.array_equal(value,right[key]) for key,value in a.memory().items())


def continuity(arena):
    """Check actual continued play without modifying the frozen first-12 fits."""
    study = _read(SOURCE)
    joystick = Joystick()
    rows = []
    with TemporaryDirectory(prefix='tip-continuity-') as folder:
        for item in study['drives']:
            index = item['index']
            path = ROOT/item['portable']
            if _hash(path)!=item['portable_sha256']:
                raise ValueError('legacy source checkpoint changed before continuity verification')
            intact = Reservoir.load(path)
            resumed = Reservoir.load(path)
            if digest_json(robust_fit(intact))!=digest_json(item['fit']):
                raise ValueError('chronological ring changed the original first-12 fit')
            with np.load(path,allow_pickle=False) as saved:
                actor = saved['native_pong_readout'].copy()
            wiring=physical_setup(index)
            starts,_=plan(joystick,count=18,seed=80412718+index)
            checkpoints=[]
            for ordinal in (12,13):
                seed,pose=starts[ordinal]
                game=native_point(arena,joystick,seed,pose,wiring,actor,SplitPublicStream(intact,resumed))
                if game!=item['held_games'][ordinal-12] or not same_memory(intact,resumed):
                    raise ValueError('actual source chronology changed across save and resume')
                fit_a,fit_b=robust_fit(intact),robust_fit(resumed)
                if digest_json(fit_a)!=digest_json(fit_b):
                    raise ValueError('the continuation chose different response models')
                order=list(intact.points())
                if order!=list(range(ordinal+1-POINTS,ordinal+1)) or int(intact.filled.sum())>CAPACITY:
                    raise ValueError('circular ordering or capacity broke during real play')
                checkpoints.append(dict(point=ordinal+1,retained_points=[n+1 for n in order],
                                        game=game,stored=int(resumed.filled.sum()),
                                        selected_alpha=[r['alpha'] for r in fit_a],same_fit=True))
                if ordinal==12:
                    mid=Path(folder)/f'continued-{index}.npz'
                    np.savez_compressed(mid,**resumed.memory())
                    resumed=assert_preserved(resumed,mid)
            checks=HeldPrediction(fit_a),HeldPrediction(fit_b)
            held=[native_point(arena,joystick,seed,pose,wiring,actor,SplitPublicStream(*checks))
                  for seed,pose in starts[14:]]
            stats_a,stats_b=[observer.evidence() for observer in checks]
            if held!=item['held_games'][2:] or digest_json(stats_a)!=digest_json(stats_b):
                raise ValueError('held predictions changed between intact and resumed fits')
            matrix=np.asarray([r['matrix_row'] for r in fit_a])
            relative=float(np.linalg.norm(matrix-wiring)/np.linalg.norm(wiring))
            passed=bool(relative<.22 and all(r['p70_rad']<r['stale_p70_rad'] and
                        r['singular_values'][-1]>.005*r['singular_values'][0] for r in stats_a))
            rows.append(dict(index=index,cases=checkpoints,score_holdout_seeds=[g['seed'] for g in held],
                             final_matrix_sha256=sha256(np.ascontiguousarray(matrix,dtype='<f8').tobytes()).hexdigest(),
                             held_decision_identical=True,qualification=passed,continuation_relative_error=relative,
                             original_checkpoint_reproduced=True))
            print('rolling reservoir continuity',index,checkpoints[-1]['retained_points'],passed,flush=True)
    _write(CONTINUITY,dict(source_sha256=_hash(SOURCE),capacity=CAPACITY,drives=rows))


def package(path,reservoir,screen,estimate):
    learner=ContinualWrist(BUNDLE)
    learner.bank=MotorBank([I,estimate])
    learner.bank.selected=1
    learner.bank.inverse=decoder(estimate)
    learner.save_between_releases(path)
    with np.load(path,allow_pickle=False) as data:
        merged=dict(data)
    np.savez_compressed(path,**merged,**reservoir.memory(),native_pong_readout=screen)
    assert_preserved(reservoir,path)
    restored=ContinualWrist(path)
    if restored.bank.selected!=1 or not np.array_equal(restored.bank.maps,[I,estimate]):
        raise ValueError('portable learner lost an actuator context')


def source(arena):
    online = _read(ROOT/'online-tip-source.json.gz')
    screen=np.asarray(_read(WARM)['weights'])
    joystick=Joystick()
    receipts=[]
    with TemporaryDirectory(prefix='tip-reservoir-') as directory:
        for index in (0,1):
            learner=Reservoir(654459998+index)
            wiring=physical_setup(index)
            scenarios,_=plan(joystick,count=18,seed=80412718+index)
            use=[]
            start=perf_counter()
            for i,(seed,pose) in enumerate(scenarios[:POINTS]):
                use.append(native_point(arena,joystick,seed,pose,wiring,screen,learner))
                if i==5:
                    middle=Path(directory)/f'prefix-{index}.npz'
                    np.savez_compressed(middle,**learner.memory())
                    learner=assert_preserved(learner,middle)
                print('reservoir native',index,i+1,int(learner.filled.sum()),flush=True)
            collect_wall=perf_counter()-start
            start=perf_counter()
            model=robust_fit(learner)
            fit_wall=perf_counter()-start
            matrix=np.asarray([row['matrix_row'] for row in model])
            destination=ROOT/f'online-tip-reservoir-learner-{index}.npz'
            package(destination,learner,screen,matrix)
            held=HeldPrediction(model)
            tests=[native_point(arena,joystick,seed,pose,wiring,screen,held)
                   for seed,pose in scenarios[POINTS:]]
            check=held.evidence()
            relative=float(np.linalg.norm(matrix-wiring)/np.linalg.norm(wiring))
            gate=(relative<.22 and all(row['p70_rad']<row['stale_p70_rad'] and
                                      row['singular_values'][-1]>.005*row['singular_values'][0] for row in check))
            receipts.append(dict(index=index,fit=model,qualification=bool(gate),held=check,train_games=use,
                                 held_games=tests,portable=destination.name,portable_sha256=_hash(destination),
                                 evaluator_relative_matrix_error=relative,
                                 old_recursive_error=online['drives'][index]['evaluator_relative_matrix_error'],
                                 old_batch_error=online['drives'][index]['stored_batch_evaluator_error'],
                                 stored_windows=int(learner.filled.sum()),available_windows=int(learner.seen.sum()),
                                 retained_numeric_bytes=learner.retained_numeric_bytes,
                                 source_motion_frames=sum(row['physical_frames'] for row in use),
                                 tiny_matrix_solves=(len(ALPHAS)*5*2+5)*13,
                                 source_wall_s=collect_wall,fit_wall_s=fit_wall,midpoint_reload_exact=True))
            print('reservoir fit',index,relative,[(r['p70_rad'],r['stale_p70_rad']) for r in check],bool(gate),flush=True)
    output=dict(drives=receipts,bundle_sha256=_hash(BUNDLE),native_sha256=_hash(WARM),maximum_windows=CAPACITY,
                holdout_frames_do_not_update=True,one_pass_live_natural_actions=True)
    _write(SOURCE,output)
    return output


def score(rows):
    modes={k:np.asarray([r['conditions'][k]['metrics']['longest_consecutive_rebounds'] for r in rows])
           for k in MODES}
    return {key:dict(mean=float(v.mean()),max=int(v.max()),ten=int((v>=10).sum()),
                     wins_vs_stale=int((v>modes['stale']).sum())) for key,v in modes.items()}


def held(arena,source_receipt):
    result=_read(TARGET) if TARGET.exists() else dict(source_sha256=_hash(SOURCE),drives={})
    if result['source_sha256']!=_hash(SOURCE) or source_receipt['bundle_sha256']!=_hash(BUNDLE):
        raise ValueError('sealed source learner differs from the held study')
    starts=np.random.default_rng(TARGET_RNG).integers(1,2**31-1,(2,8)).tolist()
    for item,roster in zip(source_receipt['drives'],starts,strict=True):
        if not item['qualification']:
            continue
        index=item['index']
        path=ROOT/item['portable']
        if _hash(path)!=item['portable_sha256']:
            raise ValueError('portable learner changed before live testing')
        loaded=ContinualWrist(path)
        batch=ContinualWrist(ROOT/f'learned-tip-learner-{index}.npz')
        wiring=physical_setup(index)
        matrices=dict(reservoir=loaded.bank.maps[1],stored_batch=batch.bank.maps[1],stale=loaded.bank.maps[0])
        if not np.array_equal(matrices['reservoir'],np.array([r['matrix_row'] for r in item['fit']])):
            raise ValueError('loaded reservoir map differs from the sealed estimate')
        block=result['drives'].setdefault(str(index),dict(cases=[]))
        policy=CarriedWristPolicy(path)
        for seed in roster[len(block['cases']):]:
            items={which:pass_through(arena,policy,seed,wiring,matrices[which]) for which in MODES}
            block['cases'].append(dict(seed=seed,conditions=items))
            _write(TARGET,result)
            print('reservoir wrist',index,seed,
                  [items[which]['metrics']['longest_consecutive_rebounds'] for which in MODES],flush=True)
        first=block['cases'][0]
        repeated=pass_through(arena,CarriedWristPolicy(path),first['seed'],wiring,ContinualWrist(path).bank.maps[1])
        if digest_json(first['conditions']['reservoir'])!=digest_json(repeated):
            raise ValueError('the reloaded wrist behavior changed')
        with np.load(path,allow_pickle=False) as archive:
            screen=archive['native_pong_readout'].copy()
        joystick=Joystick()
        scenarios,_=plan(joystick,count=18,seed=80412718+index)
        seed,pose=scenarios[-1]
        restored=native_point(arena,joystick,seed,pose,wiring,screen)
        if restored!=item['held_games'][-1]:
            raise ValueError('native source behavior changed after reloading and wrist play')
        block['reloaded_wrist_seed']=first['seed']
        block['retained_native']=restored
        block['summary']=score(block['cases'])
        _write(TARGET,result)
    pooled=[row for drive in result['drives'].values() for row in drive['cases']]
    if pooled:
        if 'original_stock_retention' not in result:
            result['original_stock_retention']=historical_retention()
        result['summary']=score(pooled)
        result['matched_tapes']=len(MODES)*len(pooled)
    _write(TARGET,result)
    print(result.get('summary',{}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    options=parser.add_mutually_exclusive_group()
    options.add_argument('--source-only',action='store_true')
    options.add_argument('--held',action='store_true')
    options.add_argument('--continuity',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    arena=BounceArena(view='wrist',width=192,height=144,max_steps=300)
    try:
        if args.continuity:
            continuity(arena)
            return
        receipt=_read(SOURCE) if args.held else source(arena)
        if not args.source_only:
            held(arena,receipt)
    finally:
        arena.close()


if __name__=='__main__':
    main()
