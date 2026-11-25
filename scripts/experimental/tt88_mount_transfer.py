"""Separately test the learned motor plant on the mechanically qualified tilt mount."""

from pathlib import Path

import numpy as np

from . import tt84_installation_system as mount
from . import tt88_inverse_body as study
from .tt4_transfer_trial import _hash,_read,_write

ROOT = study.ROOT / 'installed'
TRAIN = ROOT / 'babble.npy'
HOLD = ROOT / 'babble-held.npy'
MODEL = ROOT / 'motor.npz'
PLAN = ROOT / 'protocol.json'
BENCH = ROOT / 'body-benchmark.json'
TRAIN_EP, HELD_EP, TICKS = 320, 10, 240


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    setup=_read(mount.BUILD)
    row=setup['rows'][setup['chosen']]
    ride=row['candidate']
    plan=dict(schema='tt88-tilted-arm-transfer-1',commissioning=_hash(mount.BUILD),centered_motor=_hash(study.MODEL),
              hardware=ride['v'],ready=row['commission']['ready'],shape=ride['mount'],
              train_episodes=TRAIN_EP,held_episodes=HELD_EP,ticks=TICKS,train_seed=881249,eval_seed=887201,
              scope='Same arm on the physically qualified combined tilt and translation. Use body-only exploration '
                    'without reading any TT84 held ball or reward. Same independent tilted targets for fixed '
                    'lead, zero-shot centered motor model, and locally trained motor model.')
    if not PLAN.exists():
        _write(PLAN,plan)
    if _read(PLAN)!=plan:
        raise ValueError('Mounted-body study protocol changed')
    return plan


def run():
    p=freeze()
    with mount.installed(p['hardware'],p['shape']),study.mpc.centered(p['ready']):
        a=study.mpc.arena(width=64,height=48)
        try:
            a.reset(render=False)
            body=study.Body(a)
            for path,eps,seed in ((TRAIN,p['train_episodes'],p['train_seed']), (HOLD,p['held_episodes'],p['eval_seed'])):
                if not path.exists():
                    tmp=path.with_suffix('.tmp.npy')
                    body.collect(tmp,eps,TICKS,seed)
                    tmp.replace(path)
                    print('mounted data',path,eps*TICKS,flush=True)
            local=study.Inverse(body,a,MODEL)
            if not MODEL.exists():
                local.motor.fit(np.load(TRAIN,mmap_mode='r'))
                np.savez_compressed(MODEL,coeff=local.motor.coeff,
                                    source=np.asarray(_hash(TRAIN)),protocol=np.asarray(_hash(PLAN)))
            reused=study.Inverse(body,a,study.MODEL)
            validation=dict(protocol=_hash(PLAN),training=_hash(MODEL),held=_hash(HOLD),trained_source=_hash(TRAIN))
            out=_read(BENCH) if BENCH.exists() else dict(**validation,rows=[],complete=False)
            if any(out.get(k)!=v for k,v in validation.items()):
                raise ValueError('Mounted motor audit inputs changed')
            if out['complete']:
                return
            rng=np.random.default_rng(8822001)
            for i,ep in enumerate(np.load(HOLD,mmap_mode='r')):
                for j,start in enumerate((32,92,166)):
                    uv=rng.uniform(-.037,.037,2)
                    if any(v['episode']==i and v['start']==start for v in out['rows']):
                        continue
                    body.read(a)
                    study.move(body,ep[:start,15:20])
                    h=(6,10,15)[(i+j)%3]
                    w=body.eval(ep[None,start:start+h,15:20])[0]
                    target=study.world_goal(w,uv)
                    state=body.state
                    local_ctrl, baseline=local.solve(state,target,uv,h)
                    zero_ctrl,_=reused.solve(state,target,uv,h)
                    reached=body.eval(np.stack([baseline,zero_ctrl,local_ctrl]))
                    rank={tag:study.errors(q,target,uv) for tag,q in
                          zip(('kinematic','centered_reused','mounted_learned'),reached,strict=True)}
                    out['rows'].append(dict(episode=i,start=start,ticks=h,target=target.tolist(),patch=uv.tolist(),
                                            achieved=rank))
                out['complete']=len(out['rows'])==HELD_EP*3
                _write(BENCH,out)
                print('mounted held',i,[round(float(np.median([r['achieved'][s]['weighted'] for r in out['rows']])),3)
                                        for s in ('kinematic','centered_reused','mounted_learned')],flush=True)
        finally:
            a.close()



def public():
    from . import tt88_public_inverse as policy
    from .tt19_view_full_game import digest_json
    from .tt37_optical_clinic import OpticalClinic
    p=freeze()
    bench=_read(BENCH)
    if not bench['complete']:
        raise ValueError('Complete disjoint mounted-body controls before public receiving')
    native_receipt=_read(mount.OUT)
    if not native_receipt['complete'] or native_receipt['build']!=p['commissioning']:
        raise ValueError('Physical mounted qualification changed')
    population=[r for r in native_receipt['rows'] if r['body']==1]
    low=next(r for r in population if r['case']['role']=='low')
    routine=next(r for r in population if r['case']['role']=='routine')
    roster=(low,routine)
    forbid={r['case']['id'] for r in population}
    destination=ROOT/'public-matched.json.gz'
    signature=dict(model=_hash(MODEL),bench=_hash(BENCH),qualification=_hash(mount.OUT),
                   cases=[r['case']['id'] for r in roster],
                   sources=[_hash(policy.DATA),_hash(policy.DATA.with_suffix('.json'))],
                   note='First fixed hard and routine installation exams are used for evaluation only. '
                        'Neither their commands nor their score enters inverse babble, endpoint lookup, or ranking. '
                        'Unchanged public motor gate, with matched pre-existing command tape only as comparator.')
    out=_read(destination) if destination.exists() else dict(**signature,rows=[],complete=False)
    if any(out.get(k)!=v for k,v in signature.items()):
        raise ValueError('Public mounted qualification receipt changed')
    if out['complete']:
        return
    qual, tape = policy.mechanism.qual, policy.mechanism.tape
    with mount.installed(p['hardware'],p['shape']),study.mpc.centered(p['ready']),mount.fixture.school.enabled():
        a=study.mpc.arena()
        cad=study.mpc.arena(width=64,height=48,detail='cad')
        cam,rig,*_=study.mpc.resources(a)
        clear=qual.GripClearance(cad.model,stock=True)
        try:
            for r in roster[len(out['rows']):]:
                s=r['case']
                c,row,anchor=qual.get(_read(Path(s['source'])),s)
                raw=r['trial']['stock_commands']
                exact=raw['placements']['delivered'][0]
                results=[]
                for key,model,mode in (('kinematic',study.MODEL,'kinematic'),
                                       ('centered_reused',study.MODEL,'learned'),('mounted_learned',MODEL,'learned')):
                    engine=OpticalClinic(body=True)
                    try:
                        choose=policy.PublicRoots(engine.twin,exclude=forbid,model=model)
                        x=policy.run_case(a,clear,cam,rig,engine,choose,c,row,anchor,raw['commands'],mode)
                    finally:
                        engine.twin.close()
                    x['method']=key
                    if not policy.mechanism.first_ball(x,exact):
                        raise ValueError('Mounted optical continuation changed the original incoming')
                    if not x['issued'] and digest_json(x['events'])!=digest_json(raw['events']):
                        raise ValueError('Unchanged installed commands did not repeat')
                    if qual.received(x,True):
                        verify=tape.matched_motion(a,c,row,anchor,x['commands'],clear)
                        if not qual.received(verify,True) or digest_json(x['events'])!=digest_json(verify['events']):
                            raise ValueError('Installed optical earned contact failed exact replay')
                        x['repeat_sha256']=digest_json(verify['events'])
                    results.append(x)
                    print('mounted public',s['id'],key,x['issued'],x['landed'],x['answered'],
                          sum(n['gate'].get('nominal_legal',0) for n in x['notes']),flush=True)
                out['rows'].append(dict(id=s['id'],role=s['role'],assay=results))
                out['complete']=len(out['rows'])==len(roster)
                _write(destination,out)
        finally:
            cam.close()
            cad.close()
            a.close()


if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('body','public'),default='body',nargs='?')
    {'body':run,'public':public}[cli.parse_args().stage]()
