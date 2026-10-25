"""Two-round native-return search in a small coherent existing-policy subspace."""
import argparse,json,multiprocessing as mp,os,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from fly_connectome.vision import VisualStateEncoder
from fast_pong_feature_rollout import run_episode
import run_practical_pong_v2 as base
import run_preserved_rewarded_residual_pong as seeds
from pong_policy_subspace import build_basis,map_weights,population,acceptable,measured_slope,legal_rms
from pong_safe_mutations import balanced_buffer

OPTIMIZER_SEED=2791654083;TAG=61541
CONFIG=dict(rounds=2,candidates=16,screen_per_phase=32,confirm_per_phase=64,promote=4,finalists=1,
    calibration_per_phase=16,calibration_rows=16,dose=.15,maximum_phase_regression=.03,training_camera_cap=360000,
    wall_seconds=1200,fresh_reserve_seconds=75,workers=4)


def pins():return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_native_subspace_pong.py','pong_policy_subspace.py','fast_pong_feature_rollout.py','run_practical_pong_v2.py')}


def stage(directory,source):
    directory=Path(directory);source=Path(source);sm=base.read(source/'manifest.json');sr=base.read(source/'summary.json')
    assert sr['selected']['label']=='proposal' and base.read(source/'completion-verification.json')['passed']
    directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    for n in ('episodes','features'):(directory/n).mkdir()
    (directory/'current.npy').write_bytes((source/'selected.npy').read_bytes());(directory/'radial-cap.json').write_bytes((source/'radial-cap.json').read_bytes())
    current=np.load(directory/'current.npy');direct=np.load(base.EXPERIMENTS/'pong-leftward-population-20261022-r1/endpoint.npy');original=np.load(base.PARENT/'endpoint.npy')
    raw=base.old_engine.read_calibration(base.old_engine.CACHE);encoder=VisualStateEncoder(64,64,.02)
    basis,mapping,info=build_basis(current,direct,original,encoder.layout,raw)
    assert len(info['names'])==8 and map_weights(current,basis,np.zeros(len(basis))).tobytes()==current.tobytes()
    base.save_array(directory/'basis.npy',basis);base.save_array(directory/'raw-mapping.npy',mapping)
    capture_matrix=np.vstack([current,basis.reshape(-1,4389)]);base.save_array(directory/'capture-matrix.npy',capture_matrix)
    bank=sm['bank'];hits={};references={}
    for path in (source/'episodes').glob('proposal_*.json'):
        r=base.read(path);assert r['complete'] and r['observations_exact'] and r['first_hit_cutoff'] and r['weights_sha256']==base.sha(directory/'current.npy')
        standard=base.job(directory,'check',r['seed'],'current.npy',r['phase'],'train');assert base.config_key(r['config'])==base.config_key(standard['config'])
        hits[str(r['seed'])]=int(r['report']['hits']>0);references[str(r['seed'])]=str(path)
    assert len(hits)==sum(map(len,bank.values()))==782
    rng=np.random.default_rng(np.random.SeedSequence([OPTIMIZER_SEED,TAG,1]));order={p:[v[int(i)] for i in rng.permutation(len(v))] for p,v in bank.items()}
    fresh=seeds.fresh_seeds();fresh.update(bank=bank,ordered_training=order,optimizer_seed=OPTIMIZER_SEED,optimizer_tag=TAG)
    base.atomic_json(directory/'schedules.json',fresh)
    m=dict(files=pins(),source=base.source_signature(),source_block=str(source),config=CONFIG,basis=info,current_hits=hits,cached_receipts=references,
        current_hash=base.sha(directory/'current.npy'),basis_hash=base.sha(directory/'basis.npy'),mapping_hash=base.sha(directory/'raw-mapping.npy'),
        calibration_source_hash=base.sha(base.old_engine.CACHE),fitness='native first returns;harmonic phase mean;no action-trajectory constraints',
        phase_floor_anchor='original block-start782context rates;no morethan3ppregression',
        runtime_bridge='source-only78b2db9/parent5aeb272 exactquantile batching;parentbitexactparity;calibrationrechecks32cachednativeoutcomes',
        qualification2_out_of_training=True,assistance='own learned donor differences and generic feature-block gains;fixed scenario reuse;no BP/privileged inputs/new architecture')
    base.atomic_json(directory/'manifest.json',m);print(json.dumps(dict(staged=True,new_observations=0,subspace=info,bank=base.phases(bank,hits),config=CONFIG)),flush=True)


def task(directory,j,deadline):
    directory=Path(directory)
    if not j.get('capture_calibration'):return base.task(directory,j,deadline)
    r=run_episode(np.load(directory/j['weights']),j['config'],first_hit_cutoff=True,max_transitions=500,deadline_monotonic=deadline,
        feature_matrix=np.load(directory/'capture-matrix.npy'),feature_rows_per_episode=16)
    data=r.pop('feature_capture',None);assert data is not None
    error=float(np.max(abs(data['X'][:,:3]-data['q'])));assert error<1e-10 and np.array_equal(np.argmax(data['X'][:,:3],axis=1),data['actions'])
    name='features/'+j['id']+'.npz';np.savez(directory/name,**data)
    r.update(id=j['id'],seed=j['seed'],phase=j['phase'],purpose='train',weights_file=j['weights'],weights_sha256=base.sha(directory/j['weights']),feature_file=name,feature_q_error=error)
    base.atomic_json(directory/'episodes'/(j['id']+'.json'),r);return r


def dispatch(directory,jobs,pool,deadline,cost,training=True):
    pending=list(jobs);active={};reports={}
    while pending or active:
        while pending and len(active)<4 and time.monotonic()<deadline-2:
            if training and cost['train']+sum(v['camera_grant'] for v in active.values())+pending[0]['camera_grant']>CONFIG['training_camera_cap']:break
            j=pending.pop(0)
            with (directory/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(event='dispatch',job=j,wall_ns=time.time_ns()))+'\n');f.flush();os.fsync(f.fileno())
            active[pool.submit(task,str(directory),j,deadline-5)]=j
        if not active:break
        done,_=wait(active,timeout=1.,return_when=FIRST_COMPLETED)
        for future in done:
            j=active.pop(future);r=future.result();reports[j['id']]=r;cost[r['purpose']]+=r['transitions'];cost['initials']+=r['initial_cameras']
    return reports,len(reports)==len(jobs) and all(r['complete'] and not r['failed'] for r in reports.values())


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');s=base.read(directory/'schedules.json');assert m['files']==pins() and m['source']==base.source_signature() and m['config']==CONFIG
    assert all(m[k]==base.sha(directory/name) for k,name in [('current_hash','current.npy'),('basis_hash','basis.npy'),('mapping_hash','raw-mapping.npy')]);assert not (directory/'work-clock.json').exists()
    start=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=start));deadline=start+1200;train_deadline=deadline-75
    current=np.load(directory/'current.npy');basis=np.load(directory/'basis.npy');mapping=np.load(directory/'raw-mapping.npy');inc=np.zeros(len(basis));inc_hits=dict(m['current_hits']);anchor=base.phases(s['bank'],inc_hits)
    cache={(m['current_hash'],int(v)):h for v,h in inc_hits.items()};cost=dict(train=0,phase_probe=0,full_probe=0,initials=0)
    base.save_array(directory/'selected.npy',current);rounds=[];rng=np.random.default_rng(np.random.SeedSequence([OPTIMIZER_SEED,TAG,2]));improved=False
    def checkpoint():
        base.atomic_json(directory/'checkpoint.json',dict(coefficients=inc.tolist(),raw_coefficients=(mapping@inc).tolist(),hits=inc_hits,score=base.phases(s['bank'],inc_hits),completed_rounds=len(rounds),rng_state=rng.bit_generator.state,actual_cost=cost))
    checkpoint()
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        calibration=[]
        for p in ('left','right'):
            for i,seed in enumerate(s['ordered_training'][p][:16]):
                j=base.job(directory,f'cal_{p}_{i:02d}',seed,'current.npy',p,'train');j['capture_calibration']=True;calibration.append(j)
        rr,ok=dispatch(directory,calibration,pool,train_deadline,cost);assert ok,'partial score-free calibration'
        cases=[];error=0.
        for j in calibration:
            r=rr[j['id']];assert int(r['report']['hits']>0)==m['current_hits'][str(j['seed'])],'cached native outcome must remain exact'
            with np.load(directory/r['feature_file']) as f:cases.append(dict(phase=j['phase'],X=f['X']))
            error=max(error,r['feature_q_error'])
        raw_outputs,w=balanced_buffer(cases);base_logits=raw_outputs[:,:3];outputs=raw_outputs[:,3:].reshape(len(raw_outputs),len(basis),3);outputs-=outputs.mean(2,keepdims=True)
        np.savez(directory/'legal-outputs.npz',base_logits=base_logits,outputs=outputs,weights=w)
        base.atomic_json(directory/'preparation.json',dict(episodes=32,legal_rows=len(w),effective_dimensions=len(basis),maximum_q_error=error,all_cached_native_outcomes_exact=True,new_observations=base.accounting(directory)))
        def evaluate(label,weights_file,bank):
            h=base.sha(directory/weights_file);jobs=[]
            for p,values in bank.items():
                for i,seed in enumerate(values):
                    if (h,seed) not in cache:jobs.append(base.job(directory,f'{label}_{p}_{i:03d}',seed,weights_file,p,'train'))
            reports,complete=dispatch(directory,jobs,pool,train_deadline,cost)
            for r in reports.values():
                if r['complete'] and r['observations_exact']:cache[(h,r['seed'])]=int(r['report']['hits']>0)
            if not complete:return None
            values={str(seed):cache[(h,seed)] for v in bank.values() for seed in v};return dict(score=base.phases(bank,values),hits=values,new_episodes=len(jobs))
        for generation in range(2):
            candidates=population(rng,inc,mapping,outputs,w,base_logits);rows=[];complete=True
            for i,c in enumerate(candidates):
                c={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in c.items()};c.update(id=f'r{generation+1}c{i:02d}',weights=f'r{generation+1}c{i:02d}.npy')
                base.save_array(directory/c['weights'],map_weights(current,basis,np.array(c['coefficients'])));rows.append(c)
            base.atomic_json(directory/f'round{generation+1}-candidates.json',rows)
            bank64={p:s['ordered_training'][p][:32] for p in ('left','right')};bank128={p:s['ordered_training'][p][:64] for p in ('left','right')}
            for c in rows:
                result=evaluate(c['id']+'_s64',c['weights'],bank64)
                if result is None:complete=False;break
                c['screen']=result['score']
            if not complete:break
            promoted=sorted(rows,key=lambda c:(-c['screen']['key'][0],-c['screen']['key'][1],c['id']))[:4]
            for c in promoted:
                result=evaluate(c['id']+'_s128',c['weights'],bank128)
                if result is None:complete=False;break
                c['confirm']=result['score']
            if not complete:break
            finalist=sorted(promoted,key=lambda c:(-c['confirm']['key'][0],-c['confirm']['key'][1],c['id']))[0]
            result=evaluate(finalist['id']+'_full',finalist['weights'],s['bank'])
            if result is None:break
            incumbent_score=base.phases(s['bank'],inc_hits);accepted=acceptable(result['score'],incumbent_score,anchor)
            if accepted:
                inc=np.array(finalist['coefficients']);inc_hits=result['hits'];improved=True;(directory/'selected.npy').write_bytes((directory/finalist['weights']).read_bytes())
            record=dict(round=generation+1,candidates=rows,finalist=finalist['id'],full_score=result['score'],accepted=accepted,incumbent_score=base.phases(s['bank'],inc_hits),actual_cost=dict(cost))
            rounds.append(record);base.atomic_json(directory/f'round{generation+1}-result.json',record);checkpoint();print(json.dumps(dict(round=generation+1,full_score=result['score'],accepted=accepted,cost=cost)),flush=True)
        probe={}
        if improved:
            for p,values in s['probe'].items():
                js=[base.job(directory,f'probe_{p}_{head}_{i:02d}',seed,head+'.npy',p,'phase_probe') for i,seed in enumerate(values) for head in ('current','selected')]
                reports,complete=dispatch(directory,js,pool,deadline,cost,training=False)
                if not complete:break
                probe[p]={head:[int(reports[f'probe_{p}_{head}_{i:02d}']['report']['hits']>0) for i in range(32)] for head in ('current','selected')}
        result=dict(status='review_block_complete' if len(rounds)==2 else 'bounded_partial',completed_rounds=len(rounds),improved=improved,starting_score=anchor,selected_score=base.phases(s['bank'],inc_hits),
            coefficients=inc.tolist(),raw_coefficients=(mapping@inc).tolist(),basis_names=m['basis']['names'],probe=probe,accounting=base.accounting(directory),wall_seconds=time.monotonic()-start,
            interpretation='actual greedy native reward in a fixed small policy subspace;no copying loss/trajectoryveto;fresh not used for selection')
        base.atomic_json(directory/'summary.json',result);print(json.dumps(result),flush=True)


def backtrack(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');s=base.read(directory/'schedules.json');r1=base.read(directory/'round1-result.json')
    assert not r1['accepted'] and base.read(directory/'checkpoint.json')['completed_rounds']==1 and m['source']==base.source_signature()
    assert all(pins()[k]==v for k,v in m['files'].items() if k not in ('run_native_subspace_pong.py','pong_policy_subspace.py'))
    assert m['basis_hash']==base.sha(directory/'basis.npy') and m['mapping_hash']==base.sha(directory/'raw-mapping.npy') and m['current_hash']==base.sha(directory/'current.npy')
    assert not (directory/'adaptive-backtrack-result.json').exists()
    start=base.read(directory/'work-clock.json')['started_monotonic'];deadline=start+1200;train_deadline=deadline-75
    current=np.load(directory/'current.npy');basis=np.load(directory/'basis.npy');mapping=np.load(directory/'raw-mapping.npy');anchor=base.phases(s['bank'],m['current_hits'])
    assert np.array_equal(current,np.load(directory/'selected.npy'))
    with np.load(directory/'legal-outputs.npz') as f:outputs=f['outputs'];w=f['weights'];base_logits=f['base_logits']
    bank128={p:s['ordered_training'][p][:64] for p in ('left','right')};inc128=base.phases(bank128,m['current_hits'])
    confirmed=[c for c in r1['candidates'] if 'confirm' in c]
    best=sorted(confirmed,key=lambda c:(-c['confirm']['key'][0],-c['confirm']['key'][1],c['id']))[0]
    left=[c for c in confirmed if c['confirm']['rates']['left']>inc128['rates']['left']]
    probes=[dict(id='half_best128',coefficients=(.5*np.array(best['coefficients'])).tolist(),source=best['id'])]
    if left:
        bestleft=sorted(left,key=lambda c:(-c['confirm']['rates']['left'],-c['confirm']['key'][0],c['id']))[0]
        if bestleft['id']!=best['id']:probes.append(dict(id='half_left',coefficients=(.5*np.array(bestleft['coefficients'])).tolist(),source=bestleft['id']))
    combined,detail=measured_slope(r1['candidates'],outputs,w);raw=mapping@combined;scale=min(1.,1/max(1.,float(np.max(abs(raw)))));combined*=scale
    probes.append(dict(id='measured_combination',coefficients=combined.tolist(),construction=detail,box_scale=scale))
    accounting=base.accounting(directory);cost=dict(accounting['actual_observed']);cache={(m['current_hash'],int(seed)):h for seed,h in m['current_hits'].items()}
    for path in (directory/'episodes').glob('*.json'):
        rr=base.read(path)
        if rr.get('complete') and rr.get('observations_exact'):cache[(rr['weights_sha256'],rr['seed'])]=int(rr['report']['hits']>0)
    for p in probes:
        c=np.array(p['coefficients']);p.update(weights=p['id']+'.npy',raw_coefficients=(mapping@c).tolist(),legal_output_rms=legal_rms(outputs,w,c),
            legal_action_disagreement=float(w@(np.argmax(base_logits+np.einsum('nka,k->na',outputs,c),1)!=np.argmax(base_logits,1))))
        base.save_array(directory/p['weights'],map_weights(current,basis,c))
    base.atomic_json(directory/'adaptive-backtrack-amendment.json',dict(reason='jointTRAIN-only adaptation after harmful firstfullcandidate;replace remainingrandomround with≤3informedshortprobes',
        original_files=m['files'],new_files=pins(),original_clock_preserved=True,elapsed_before_backtrack=time.monotonic()-start,original_deadline_seconds=1200,
        original_training_cap=360000,prior_actual_accounting=accounting,partial_round2_cost=accounting['actual_observed']['train']-r1['actual_cost']['train'],probes=probes))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        def evaluate(p,bank,suffix):
            h=base.sha(directory/p['weights']);jobs=[]
            for phase,values in bank.items():
                for i,seed in enumerate(values):
                    if (h,seed) not in cache:jobs.append(base.job(directory,f'bt_{p["id"]}_{suffix}_{phase}_{i:03d}',seed,p['weights'],phase,'train'))
            reports,complete=dispatch(directory,jobs,pool,train_deadline,cost)
            for rr in reports.values():
                if rr['complete'] and rr['observations_exact']:cache[(h,rr['seed'])]=int(rr['report']['hits']>0)
            if not complete:return None
            hits={str(seed):cache[(h,seed)] for values in bank.values() for seed in values};return dict(score=base.phases(bank,hits),hits=hits)
        complete=True
        for p in probes:
            result=evaluate(p,bank128,'128')
            if result is None:complete=False;break
            p['score128']=result['score'];base.atomic_json(directory/'adaptive-backtrack-progress.json',probes)
        accepted=False;full=None;chosen=None;probe={}
        if complete:
            better=[p for p in probes if p['score128']['key'][0]>inc128['key'][0]+1e-15]
            if better:
                chosen=sorted(better,key=lambda p:(-p['score128']['key'][0],-p['score128']['key'][1],p['id']))[0];full=evaluate(chosen,s['bank'],'full')
                if full is not None and acceptable(full['score'],anchor,anchor):
                    accepted=True;(directory/'selected.npy').write_bytes((directory/chosen['weights']).read_bytes())
                    base.atomic_json(directory/'adaptive-selected.json',dict(coefficients=chosen['coefficients'],raw_coefficients=chosen['raw_coefficients'],hits=full['hits'],score=full['score']))
        if accepted:
            for phase,values in s['probe'].items():
                js=[base.job(directory,f'bt_fresh_{phase}_{head}_{i:02d}',seed,head+'.npy',phase,'phase_probe') for i,seed in enumerate(values) for head in ('current','selected')]
                rr,ok=dispatch(directory,js,pool,deadline,cost,training=False)
                if not ok:complete=False;break
                probe[phase]={head:[int(rr[f'bt_fresh_{phase}_{head}_{i:02d}']['report']['hits']>0) for i in range(32)] for head in ('current','selected')}
        result=dict(status='adaptive_backtrack_complete' if complete and (chosen is None or full is not None) else 'bounded_partial',probes=probes,incumbent128=inc128,full=full['score'] if full else None,
            accepted=accepted,selected_unchanged=not accepted,fresh=probe,accounting=base.accounting(directory),wall_seconds=time.monotonic()-start,
            original_clock_preserved=True,interpretation='boundedTRAIN-onlybacktracking;no furthermacro tuning after null;partialrandomround cost retained')
        base.atomic_json(directory/'adaptive-backtrack-result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','run','backtrack'));p.add_argument('root',type=Path);p.add_argument('--source',type=Path);a=p.parse_args()
    if a.mode=='stage':stage(a.root,a.source)
    elif a.mode=='backtrack':backtrack(a.root)
    else:run(a.root)
