"""Learn the sign of a discrete motor step from paired native durations."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import binomtest

from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_pong import ROLE_MODEL, cases
from .tt22_duration_transfer import MACRO
from .tt22_motor_transfer import MotorPrimitive
from .tt23_anonymous_action import CHECK, OPTIONS, WIDTH, collect, proposal
from .tt23_anonymous_replication import TRAIN
from .tt23_anonymous_target import target_context
from .tt23_optical_contact import OUT


SOURCE = OUT/'direction-source.json.gz'
NATIVE = OUT/'direction-native.json.gz'
ARM = OUT/'direction-arm.json.gz'
ARM_BANK = OUT/'smooth-jog-arm.json.gz'
ARM_SEED = 846093157
NATIVE_OFFSET = 1284500000
NATIVE_COUNT = 48
PERMUTATION_SEED = 1351429811
CHOICES = (2,3,4)
MIRROR = np.array((-1.,-1.,1.,1.))
MODES = ('learned','neutral','shuffled','reversed')


def durations(rows):
    scenes,ticks,seeds = [],[],[]
    for r in zip(*(rows[str(i)] for i in range(len(OPTIONS))),strict=True):
        reference = r[0]
        if any(item['case']!=reference['case'] or item['context']!=reference['context'] or item['option']!=index
               for index,item in enumerate(r)):
            raise ValueError('source rolls differ before the fixed motor decision')
        seeds.append(reference['case']['serve'])
        if reference['context'] is not None:
            scenes.append(reference['context'])
            ticks.append([item['ticks'] for item in r])
    return np.asarray(scenes),np.asarray(ticks),seeds


def marginal_votes(ticks):
    return np.sign(ticks[:,4:]-ticks[:,2::-1]).mean(axis=1)


def policy(scenes,votes):
    x = np.concatenate((scenes,scenes*MIRROR))
    signed = np.r_[votes,-votes]
    values = np.full((len(x),len(OPTIONS)),-2.)
    values[:,2] = -signed
    values[:,4] = signed
    values[:,3] = 1.e-12
    return dict(x=x.tolist(),value=values.tolist())


def fitting(output):
    began = perf_counter()
    original,replications = _read(CHECK),_read(TRAIN)
    banks = [(1559000000,original['training']),*((row['offset'],row['rows']) for row in replications['sets'])]
    if len(banks)!=5:
        raise ValueError('expected five complete signed action training archives')
    seen,source_sets = set(),[]
    source_x,source_y,null_y = [],[],[]
    for offset,archive in banks:
        scenes,ticks,seeds = durations(archive)
        if len(set(seeds))!=len(seeds) or seen.intersection(seeds):
            raise ValueError('training starts overlap across source sets')
        seen.update(seeds)
        permutation = np.random.default_rng(np.random.SeedSequence([PERMUTATION_SEED,offset]))
        indexes = permutation.random(ticks.shape).argsort(axis=1)
        labels,shuffled = marginal_votes(ticks),marginal_votes(np.take_along_axis(ticks,indexes,axis=1))
        source_x.append(scenes)
        source_y.append(labels)
        null_y.append(shuffled)
        source_sets.append(dict(offset=offset,starts=seeds,detected=len(scenes),votes=labels.tolist(),
                                shuffle_indexes=indexes.tolist(),shuffle_votes=shuffled.tolist(),
                                learned=policy(scenes,labels),shuffled=policy(scenes,shuffled)))
    targets = cases(NATIVE_COUNT,NATIVE_OFFSET)
    if seen.intersection(case.serve for case in targets):
        raise ValueError('native held scenes overlap training reels')
    x,y,z = (np.concatenate(array) for array in (source_x,source_y,null_y))
    data = dict(schema='tt23-direction-source-1',sources={p.name:_hash(p) for p in (CHECK,TRAIN,ROLE_MODEL)},
                native_offset=NATIVE_OFFSET,native_count=NATIVE_COUNT,native_seeds=[c.serve for c in targets],
                arm_seed=ARM_SEED,width=WIDTH.tolist(),step=float(OPTIONS[4]),arm_step_m=float(OPTIONS[4]*.848),
                permutation_seed=PERMUTATION_SEED,used=len(y),sets=source_sets,
                pooled=dict(learned=policy(x,y),shuffled=policy(x,z)),seconds=perf_counter()-began)
    _write(output,data)
    print('direction frozen',len(y),'observed of',len(seen),'source scenes',flush=True)
    return data


def pick(context,model,mode):
    if context is None or mode=='neutral':
        return 3
    weights = model['shuffled' if mode=='shuffled' else 'learned']
    index = int(np.argmax(proposal(weights,context)))
    if index not in CHOICES:
        raise ValueError('the scorer chose an unavailable motor displacement')
    return 6-index if mode=='reversed' else index


def selected(rows,contexts,model,mode):
    return [row['options'][pick(x,model,mode)] for row,x in zip(rows,contexts,strict=True)]


def native_stats(rows,contexts,model):
    result = {}
    for mode in MODES:
        trials = selected(rows,contexts,model,mode)
        result[mode] = dict(ticks=sum(row['ticks'] for row in trials),
                            first_returns=sum(bool(row['contacts'] and row['contacts'][0]['returned']) for row in trials),
                            choices=[pick(x,model,mode) for x in contexts])
    return result


def replay_native_choice(case,row,model,role):
    values = np.asarray(model['learned']['value']).copy()
    values[:,[2,4]] = values[:,[4,2]]
    reversed_model = dict(x=model['learned']['x'],value=values.tolist())
    specs = dict(learned=(3,model['learned']),neutral=(3,None),shuffled=(3,model['shuffled']),
                 reversed=(3,reversed_model))
    direct = collect([case],specs,role)
    for mode in MODES:
        expected = row['options'][pick(row['context'],model,mode)]
        if expected!=direct[mode][0]:
            raise ValueError('direct native play differs from the chosen fixed-option tape')
    return len(direct)


def native(source,output):
    import torch

    started = perf_counter()
    torch.set_num_threads(1)
    model = _read(source)
    if model['sources'][ROLE_MODEL.name]!=_hash(ROLE_MODEL):
        raise ValueError('the generic optical role tracker changed after source fitting')
    role = RoleModel(**_read(ROLE_MODEL))
    schedule = cases(model['native_count'],model['native_offset'])
    if [case.serve for case in schedule]!=model['native_seeds']:
        raise ValueError('native held schedule changed after source fitting')
    outcomes = collect(schedule,{str(i):(i,None) for i in CHOICES},role)
    rows = []
    for i,case in enumerate(schedule):
        seen = [outcomes[str(choice)][i] for choice in CHOICES]
        context = seen[0]['context']
        if any(trial['case']!=seen[0]['case'] or trial['context']!=context or trial['option']!=choice
               for choice,trial in zip(CHOICES,seen,strict=True)):
            raise ValueError('the true native options did not share an observed context')
        options = [None]*len(OPTIONS)
        for choice,trial in zip(CHOICES,seen,strict=True):
            options[choice] = trial
        rows.append(dict(seed=case.serve,context=context,options=options))
    contexts = [row['context'] for row in rows]
    detected = [i for i,x in enumerate(contexts) if x is not None]
    scores = native_stats(rows,contexts,model['pooled'])
    individual = []
    for data in model['sets']:
        stats = native_stats(rows,contexts,data)
        agreement = sum(stats['learned']['choices'][i]==scores['learned']['choices'][i] for i in detected)
        individual.append(dict(offset=data['offset'],scores=stats,agreement=agreement/len(detected)))
    consistency = sum(data['agreement']>=.75 for data in individual)
    gate = consistency>=4 and scores['learned']['ticks']>max(scores[name]['ticks'] for name in ('neutral','shuffled'))
    verified = replay_native_choice(schedule[0],rows[0],model['pooled'],role)
    result = dict(schema='tt23-direction-native-1',source=source.name,source_sha256=_hash(source),rows=rows,
                  points=len(rows),detected=len(detected),scores=scores,independent=individual,
                  consistency=consistency,gate=gate,verified_direct=verified,seconds=perf_counter()-started)
    _write(output,result)
    print('native direction',len(rows),'consistent fits',consistency,'gate',gate,
          {mode:(row['ticks'],row['first_returns']) for mode,row in scores.items()},flush=True)
    return result


def paired_returns(after,before):
    gain = sum(a and not b for a,b in zip(after,before,strict=True))
    loss = sum(b and not a for a,b in zip(after,before,strict=True))
    return dict(gains=gain,losses=loss,p_greater=float(binomtest(gain,gain+loss,.5,alternative='greater').pvalue)
                if gain+loss else 1.)


def arm_stats(rows,contexts,model):
    outcomes = {mode:selected(rows,contexts,model,mode) for mode in MODES}
    legal = {mode:[trial['legal']>0 for trial in trials] for mode,trials in outcomes.items()}
    scores = {mode:dict(legal=sum(legal[mode]),contacts=sum(trial['contacts']>0 for trial in trials),
                        choices=[pick(x,model,mode) for x in contexts]) for mode,trials in outcomes.items()}
    beats = {mode:paired_returns(legal['learned'],legal[mode]) for mode in MODES if mode!='learned'}
    return dict(scores=scores,paired=beats)


def arm(source,native_path,bank,output):
    frozen,native_receipt,physical = (_read(path) for path in (source,native_path,bank))
    if native_receipt['source_sha256']!=_hash(source):
        raise ValueError('held native results use a different frozen direction policy')
    rows = physical['rows']
    if (physical.get('seed_generator')!=frozen['arm_seed'] or not physical.get('complete') or len(rows)!=64
            or len({r['seed'] for r in rows})!=64 or physical.get('macro_hash')!=_hash(MACRO)):
        raise ValueError('arm bank is incomplete or differs from the registered physical protocol')
    ready = MotorPrimitive(np.asarray(_read(MACRO)['parameters']),None).ready_world
    contexts = [target_context(row['sight'],ready) for row in rows]
    pooled = arm_stats(rows,contexts,frozen['pooled'])
    fits = [dict(offset=item['offset'],**arm_stats(rows,contexts,item)) for item in frozen['sets']]
    scores = pooled['scores']
    gate = native_receipt['gate'] and scores['learned']['legal']>max(scores[m]['legal'] for m in ('neutral','shuffled'))
    by_domain = {key:arm_stats([r for r in rows if r['domain']==key],
                              [x for r,x in zip(rows,contexts,strict=True) if r['domain']==key],frozen['pooled'])
                 for key in sorted({row['domain'] for row in rows})}
    result = dict(schema='tt23-direction-arm-1',source_sha256=_hash(source),native_sha256=_hash(native_path),
                  bank=bank.name,bank_sha256=_hash(bank),starts=[r['seed'] for r in rows],
                  identified=sum(x is not None for x in contexts),pooled=pooled,independent=fits,by_domain=by_domain,
                  gate=gate)
    _write(output,result)
    print('arm direction',len(rows),'gate',gate,{mode:(r['contacts'],r['legal']) for mode,r in scores.items()},flush=True)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fit','native','arm'))
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--native',type=Path,default=NATIVE)
    parser.add_argument('--bank',type=Path,default=ARM_BANK)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.phase=='fit':
        fitting(args.output or args.source)
    elif args.phase=='native':
        native(args.source,args.output or args.native)
    else:
        arm(args.source,args.native,args.bank,args.output or ARM)
