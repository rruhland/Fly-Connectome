"""Bounded single-pass motor identification during reward-trained native Pong."""

import argparse
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE,CarriedWristPolicy
from .tt24_continual_wrist import ContinualWrist
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import WARM,arcade
from .tt24_learned_tip_transfer import NativeTip
from .tt24_motor_bank import I,STEP,MotorBank,decoder
from .tt24_pong_arm import NATIVE_CAP,bounce,historical_retention
from .tt24_pong_arm_local import ALPHAS,WINDOW,crosscheck
from .tt24_pong_arm_tip import Joystick,plan
from .tt4_transfer_trial import _hash,_read,_write


SOURCE = ROOT/'online-tip-source.json.gz'
TARGET = ROOT/'online-tip-held.json.gz'
TRAINING_POINTS = 12
MODE_NAMES = ('online','stored_batch','stale','stored_shuffle')
EVALUATION_SEED = 1308215433


class OnlineResponse:
    """Receive one requested command and one subsequent measured encoder sample."""

    def __init__(self):
        n = len(ALPHAS)
        self.weights = np.tile(I,(n,1,1))
        self.covariance = np.broadcast_to(I/.00003,(n,5,5,5)).copy()
        self.error = np.zeros((n,5))
        self.counts = np.zeros((n,5),dtype=np.int64)
        self.alpha = np.asarray(ALPHAS)[:,None]
        self.history = deque(maxlen=WINDOW+1)
        self.integrated = np.zeros(5)
        self.filtered = np.zeros((n,5))
        self.motion = np.zeros(5)
        self.ticks = 0
        self.effective = 0

    def begin(self,joints):
        self.integrated.fill(0.)
        self.filtered.fill(0.)
        self.motion.fill(0.)
        self.history.clear()
        self.last = np.asarray(joints,float).copy()
        self.history.append((self.filtered.copy(),self.last,self.motion.copy()))

    def observe(self,requested,joints):
        measured = np.asarray(joints,float)
        self.integrated += STEP*requested
        self.filtered += self.alpha*(self.integrated-self.filtered)
        self.motion += abs(measured-self.last)
        self.last = measured.copy()
        self.history.append((self.filtered.copy(),self.last,self.motion.copy()))
        self.ticks += 1
        if len(self.history)<=WINDOW:
            return
        filtered,previous,travel = self.history[0]
        x = self.filtered-filtered
        y = measured-previous
        visible = ((self.motion-travel)[None,:]>=.00065) & (np.linalg.norm(x,axis=1)[:,None]>.00055)
        self.effective += int(visible.any())
        residual = y[None,:]-np.einsum('ljd,ld->lj',self.weights,x)
        px = np.einsum('ljde,le->ljd',self.covariance,x)
        leverage = np.einsum('ljd,ld->lj',px,x)
        # Early directions carry higher prediction uncertainty. Late inconsistent
        # observations receive little weight once similar commands are well covered.
        scale = .00065*np.sqrt(1.+np.maximum(0.,leverage)) + .008*np.exp(-self.counts/120.)
        soft = 1. / (1.+(residual/scale)**2)
        update = np.where(visible,soft/(1.+soft*leverage),0.)
        self.weights += (update*residual)[:,:,None]*px
        self.covariance -= update[:,:,None,None]*px[:,:,:,None]*px[:,:,None,:]
        self.covariance = .5*(self.covariance+self.covariance.swapaxes(-1,-2))
        self.error += .004 * visible * (np.minimum(abs(residual),.012)-self.error)
        self.counts += visible

    def selected(self):
        chosen = np.argmin(self.error,axis=0)
        return [dict(alpha=ALPHAS[k],matrix_row=self.weights[k,j].tolist(),
                     prequential_loss=float(self.error[k,j]),available_windows=int(self.counts[k,j]))
                for j,k in enumerate(chosen)]

    def memory(self):
        values = {name:getattr(self,name).copy() for name in ('weights','covariance','error','counts')}
        return values|dict(ticks=np.asarray(self.ticks),effective=np.asarray(self.effective))

    @classmethod
    def load(cls,checkpoint):
        algorithm = cls()
        with np.load(checkpoint,allow_pickle=False) as saved:
            for name in ('weights','covariance','error','counts'):
                setattr(algorithm,name,saved[f'online_{name}'].copy())
            for name in ('ticks','effective'):
                setattr(algorithm,name,int(saved[f'online_{name}']))
        return algorithm

    @property
    def numeric_memory_bytes(self):
        persistent = self.alpha.nbytes+sum(v.nbytes for v in self.memory().values())
        working = 8*(len(ALPHAS)*5 + 15 + (WINDOW+1)*(len(ALPHAS)*5+10))
        return persistent+working


class StreamingTip(NativeTip):
    """Expose source events immediately, holding no clip inside the learner or paddle."""

    def __init__(self,arena,joystick,pose,wiring,spec,expected,sink):
        super().__init__(arena,joystick,pose,wiring,spec)
        self.expected,self.sink = expected,sink
        self.received = 0
        start = self.traces.pop()
        if not np.array_equal(start,expected['encoders'][0]):
            raise ValueError('native encoder start changed')
        if sink is not None:
            sink.begin(start)

    def step(self,drive):
        super().step(drive)
        if self.commands:
            request,position = self.commands.pop(),self.traces.pop()
            i = self.received
            if not (np.array_equal(request,self.expected['commands'][i]) and
                    np.array_equal(position,self.expected['encoders'][i+1])):
                raise ValueError('native plant differed from the archived public event stream')
            if self.sink is not None:
                self.sink.observe(np.asarray(request),position)
            self.received += 1


def point(arena,joystick,pose,wiring,native,expected,sink=None):
    tip = None
    def attach(spec):
        nonlocal tip
        tip = StreamingTip(arena,joystick,pose,wiring,spec,expected,sink)
        return tip
    score,hits,ticks = arcade(native,[expected['seed']],cap=NATIVE_CAP,opponent=.35,paddle_factory=attach)[0,0]
    if ((score,hits,ticks)!=(expected['score'],expected['hits'],expected['ticks']) or
            tip.received!=len(expected['commands'])):
        raise ValueError('native game/physical replay did not preserve its archived result')
    return int(tip.received)


def save_midpoint(learner,path):
    np.savez_compressed(path,**{f'online_{key}':value for key,value in learner.memory().items()})
    restored = OnlineResponse.load(path)
    if any(not np.array_equal(value,restored.memory()[key]) for key,value in learner.memory().items()):
        raise ValueError('midpoint reload changed online motor memory')
    return restored


def portable(path,online,native,matrix):
    learner = ContinualWrist(BUNDLE)
    learner.bank = MotorBank([I,matrix])
    learner.bank.selected = 1
    learner.bank.inverse = decoder(matrix)
    learner.save_between_releases(path)
    with np.load(path,allow_pickle=False) as saved:
        output = dict(saved)
    output['native_pong_readout'] = native
    output.update({f'online_{key}':value for key,value in online.memory().items()})
    np.savez_compressed(path,**output)
    resumable = OnlineResponse.load(path)
    if any(not np.array_equal(value,resumable.memory()[key]) for key,value in online.memory().items()):
        raise ValueError('portable learner changed online fit state')
    loaded = ContinualWrist(path)
    if loaded.bank.selected!=1 or not np.array_equal(loaded.bank.maps,[I,matrix]):
        raise ValueError('portable learner changed one of the saved actuator contexts')
    return loaded


def source(arena):
    native = np.asarray(_read(WARM)['weights'])
    joystick = Joystick()
    receipts = []
    with TemporaryDirectory(prefix='online-tip-') as tmp:
        for index in (0,1):
            asset = ROOT/f'learned-tip-transfer-{index}.json.gz'
            archived = _read(asset)
            if archived['native_checkpoint_sha256']!=_hash(WARM):
                raise ValueError('archived source points used a different native policy')
            wiring = np.asarray(archived['wiring'])
            clips = archived['clips']
            scenarios,_ = plan(joystick,count=len(clips),seed=80412718+index)
            learner = OnlineResponse()
            midpoint = Path(tmp)/f'midpoint-{index}.npz'
            for i,((seed,pose),reference) in enumerate(zip(scenarios,clips,strict=True)):
                if reference['seed']!=seed:
                    raise ValueError('archived poses and native seeds are inconsistent')
                if i>=TRAINING_POINTS:
                    break
                point(arena,joystick,pose,wiring,native,reference,learner)
                if i==5:
                    learner = save_midpoint(learner,midpoint)
                print('online native',index,i+1,learner.ticks,flush=True)
            model = learner.selected()
            estimate = np.asarray([entry['matrix_row'] for entry in model])
            path = ROOT/f'online-tip-learner-{index}.npz'
            portable(path,learner,native,estimate)
            for (_,pose),reference in zip(scenarios[TRAINING_POINTS:],clips[TRAINING_POINTS:],strict=True):
                point(arena,joystick,pose,wiring,native,reference)
            held = crosscheck(model,clips[TRAINING_POINTS:])
            relative = float(np.linalg.norm(estimate-wiring)/np.linalg.norm(wiring))
            usable = relative<.22 and all(h['p70_rad']<h['stale_p70_rad'] and
                         h['singular_values'][-1]>.005*h['singular_values'][0] for h in held)
            receipts.append(dict(stream=index,archive=asset.name,archive_sha256=_hash(asset),
                                 portable=path.name,portable_sha256=_hash(path),training_points=TRAINING_POINTS,
                                 source_frames=learner.ticks,usable_windows=learner.effective,
                                 stored_numeric_memory_bytes=learner.numeric_memory_bytes,
                                 held_rounds=len(clips)-TRAINING_POINTS,gate=bool(usable),rows=model,
                                 held_encoder=held,stored_batch_held=archived['validation']['source'],
                                 evaluator_relative_matrix_error=relative,
                                 stored_batch_evaluator_error=float(np.linalg.norm(
                                     np.asarray([r['matrix_row'] for r in archived['fits']['source']])-wiring)/
                                     np.linalg.norm(wiring)),midpoint_reload_matched=True))
            print('online fit',index,relative,
                  [(item['p70_rad'],item['stale_p70_rad']) for item in held],bool(usable),flush=True)
    result = dict(architecture='live interleaved public events; fixed size causal windows; robust RLS; causal lag selection',
                  bundle_sha256=_hash(BUNDLE),native_checkpoint_sha256=_hash(WARM),drives=receipts)
    _write(SOURCE,result)
    return result


def pass_through(arena,policy,seed,wire,estimate):
    now = bounce(arena,policy,seed,wire,estimate)
    replay = bounce(arena,policy,seed,wire,estimate,tape=now['actions'])
    if digest_json(now)!=digest_json(replay):
        raise ValueError('physical wrist trial did not replay exactly')
    return now


def summaries(rows):
    data = {label:np.asarray([row['conditions'][label]['metrics']['longest_consecutive_rebounds'] for row in rows])
            for label in MODE_NAMES}
    return {label:dict(mean=float(vals.mean()),max=int(vals.max()),ten=int((vals>=10).sum()),
                       wins_over_stale=int((vals>data['stale']).sum()),
                       ties_with_stale=int((vals==data['stale']).sum())) for label,vals in data.items()}


def evaluate(arena,receipt):
    if receipt['bundle_sha256']!=_hash(BUNDLE):
        raise ValueError('carried wrist policy changed during the study')
    result = _read(TARGET) if TARGET.exists() else dict(source_sha256=_hash(SOURCE),drives={})
    if result['source_sha256']!=_hash(SOURCE):
        raise ValueError('held wrist evidence used a different online fit')
    rosters = np.random.default_rng(EVALUATION_SEED).integers(1,2**31-1,(2,8)).tolist()
    for item,seeds in zip(receipt['drives'],rosters,strict=True):
        if not item['gate']:
            continue
        index = item['stream']
        previous = _read(ROOT/item['archive'])
        path = ROOT/item['portable']
        if _hash(path)!=item['portable_sha256'] or _hash(ROOT/item['archive'])!=item['archive_sha256']:
            raise ValueError('portable learner or paired comparator changed before held testing')
        loaded = ContinualWrist(path)
        with np.load(path,allow_pickle=False) as saved:
            native = saved['native_pong_readout'].copy()
        wiring = np.asarray(previous['wiring'])
        maps = dict(online=loaded.bank.maps[1],stale=loaded.bank.maps[0],
                    stored_batch=np.asarray([r['matrix_row'] for r in previous['fits']['source']]),
                    stored_shuffle=np.asarray([r['matrix_row'] for r in previous['fits']['shuffled']]))
        if not np.array_equal(maps['online'],np.asarray([r['matrix_row'] for r in item['rows']])):
            raise ValueError('loaded online head differs from the sealed estimate')
        if set(seeds).intersection(previous['target_seeds']):
            raise ValueError('fresh wrist roster overlaps the earlier batch study')
        block = result['drives'].setdefault(str(index),dict(cases=[]))
        wrist = CarriedWristPolicy(path)
        for seed in seeds[len(block['cases']):]:
            conditions = {name:pass_through(arena,wrist,seed,wiring,maps[name]) for name in MODE_NAMES}
            block['cases'].append(dict(seed=seed,conditions=conditions))
            _write(TARGET,result)
            print('online wrist',index,seed,
                  [row['metrics']['longest_consecutive_rebounds'] for row in conditions.values()],flush=True)
        first = block['cases'][0]
        restored = pass_through(arena,CarriedWristPolicy(path),first['seed'],wiring,
                                ContinualWrist(path).bank.maps[1])
        if digest_json(first['conditions']['online'])!=digest_json(restored):
            raise ValueError('reloaded actuator context changed its live wrist feedback')
        joystick = Joystick()
        starts,_=plan(joystick,count=len(previous['clips']),seed=80412718+index)
        reference=previous['clips'][-1]
        point(arena,joystick,starts[-1][1],wiring,native,reference)
        block['native_retention_seed']=reference['seed']
        block['wrist_reload_seed']=first['seed']
        block['summary'] = summaries(block['cases'])
        _write(TARGET,result)
    pooled = [row for b in result['drives'].values() for row in b['cases']]
    if pooled:
        if 'historical_original_stock' not in result:
            result['historical_original_stock']=historical_retention()
        result['summary']=summaries(pooled)
        result['matched_tapes']=len(pooled)*len(MODE_NAMES)
    _write(TARGET,result)
    print(result.get('summary',{}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options=parser.add_mutually_exclusive_group()
    options.add_argument('--source-only',action='store_true')
    options.add_argument('--held',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    arena=BounceArena(view='wrist',width=192,height=144,max_steps=300)
    try:
        receipt=_read(SOURCE) if args.held else source(arena)
        if not args.source_only:
            evaluate(arena,receipt)
    finally:
        arena.close()


if __name__=='__main__':
    main()
