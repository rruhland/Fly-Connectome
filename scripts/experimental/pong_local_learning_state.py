"""Compact completed-step state for repeated own-reward repair learning."""
from pathlib import Path
import numpy as np
import run_practical_pong_v2 as base


def absolute_case(source,item):
    return dict(item,feature_file=str((source/item['feature_file']).resolve()))


def load_state(source):
    source=Path(source)
    if (source/'learning-state.json').exists():return base.read(source/'learning-state.json')
    m=base.read(source/'manifest.json');r=base.read(source/'summary.json');s=base.read(source/'schedules.json')
    assert r['status']=='one_reweighted_fit_complete' and r['selected']['label']=='proposal_4x'
    hits={}
    for path in (source/'episodes').glob('proposal_*.json'):
        rr=base.read(path);hits[str(rr['seed'])]=int(rr['report']['hits']>0)
    current={}
    for kind in ('preservation','guard'):
        for item in m['cases'][kind]:current[item['seed']]=dict(absolute_case(source,item),kind=kind)
    donor_source=Path(m['source_block']);donors=[]
    for path in sorted((donor_source/'episodes').glob('donor_*.json')):
        rr=base.read(path);assert rr['report']['hits']>0
        donors.append(dict(seed=rr['seed'],phase=rr['phase'],feature_file=str(donor_source/rr['feature_file']),reward_receipt=str(path)))
    return dict(bank=m['bank'],hits=hits,prior_hits=m['carry_hits'],theta=str(source/'selected_theta.npy'),weights=str(source/'selected.npy'),
        prior_theta=str(source/'carry_theta.npy'),prior_weights=str(source/'carry.npy'),donors=donors,current_cases=list(current.values()),
        probe=r['probe'],probe_seeds=s['probe'])


def prepare_next(directory,source,target_counts=None,include_evaluation=None):
    directory=Path(directory);source=Path(source);state=load_state(source)
    directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    for n in ('episodes','features','reused_features','memory_donors','mutants'):(directory/n).mkdir()
    for target,key in [('carry_theta.npy','theta'),('carry.npy','weights'),('fallback_theta.npy','prior_theta'),('fallback.npy','prior_weights')]:
        (directory/target).write_bytes(Path(state[key]).read_bytes())
    for name in ('old-transform.npy','extra-transform.npy','search-transform.npy','radial-cap.json','feature-buffer.npz'):(directory/name).write_bytes((source/name).read_bytes())
    bank={p:list(v) for p,v in state['bank'].items()};hits=dict(state['hits']);fallback=dict(state['prior_hits']);latest={p:[] for p in bank};new_guards=[]
    for p,arms in state['probe'].items():
        for seed,a,b in zip(state['probe_seeds'][p],arms['carry'],arms['selected']):
            assert seed not in bank[p];bank[p].append(seed);hits[str(seed)]=b;fallback[str(seed)]=a
            if not b:latest[p].append(seed)
            if b and not a:new_guards.append(dict(seed=seed,phase=p))
    evaluation_queries=[];evaluation_record=None
    if include_evaluation:
        evaluation=Path(include_evaluation);em=base.read(evaluation/'manifest.json');er=base.read(evaluation/'summary.json');es=base.read(evaluation/'schedules.json')
        assert er['status']=='complete' and er['first_hit_complete'] and em['no_learning']
        assert em['current_hash']==base.sha(directory/'carry.npy') and em['original_hash']==base.sha(evaluation/'original.npy')
        (directory/'original_donor.npy').write_bytes((evaluation/'original.npy').read_bytes())
        assert np.array_equal(np.load(base.PARENT/'theta_at6.npy')@np.load(base.PARENT/'search-transform.npy'),np.load(directory/'original_donor.npy'))
        qualification_priority={p:[] for p in bank}
        for cohort,phases in es['cohorts'].items():
            for p,seeds in phases.items():
                qualification_misses=[];successes=[];gained=[]
                for i,seed in enumerate(seeds):
                    assert seed not in bank[p] and base.leftward(seed)==(p=='left');values={}
                    for head in ('current','original'):
                        rr=base.read(evaluation/'episodes'/f'{cohort}_{p}_{head}_{i:02d}.json');standard=base.job(directory,'check',seed,'carry.npy',p,'train')
                        assert rr['seed']==seed and rr['complete'] and rr['observations_exact'] and rr['first_hit_cutoff'] and rr['weights_sha256']==em[head+'_hash']
                        assert base.config_key(rr['config'])==base.config_key(standard['config'])
                        values[head]=int(rr['report']['hits']>0)
                    bank[p].append(seed);hits[str(seed)]=values['current']
                    if not values['current']:qualification_misses.append(seed)
                    else:
                        successes.append(seed)
                        if not values['original']:gained.append(seed)
                    if values['original'] and not values['current']:evaluation_queries.append(dict(seed=seed,phase=p,weights='original_donor.npy',expected_success=True,donor_source='ORIGINAL'))
                qualification_priority[p].extend(qualification_misses)
                if p=='right':
                    selected=[gained[ix] for ix in np.linspace(0,len(gained)-1,min(4,len(gained)),dtype=int)]
                    other=[v for v in successes if v not in selected]
                    if len(selected)<4:selected.extend(other[ix] for ix in np.linspace(0,len(other)-1,min(4-len(selected),len(other)),dtype=int))
                    for seed in selected:new_guards.append(dict(seed=seed,phase='right',origin='qualification_current_success_prefer_gain',cohort=cohort))
        for p in bank:latest[p]=qualification_priority[p]+latest[p]
        evaluation_record=dict(root=str(evaluation),manifest_hash=base.sha(evaluation/'manifest.json'),summary_hash=base.sha(evaluation/'summary.json'),
            current_contexts_added=256,prior_evaluation_immutable=True,usage='TRAIN from this new learning step; previous reliability report is unchanged',missing_prior_head_outcomes='left absent; carry-only selection')
    theta=np.load(directory/'carry_theta.npy');reused=[]
    for item in state['current_cases']:
        if not hits[str(item['seed'])]:
            assert item['kind']!='guard';continue
        with np.load(item['feature_file']) as f:case={k:f[k] for k in f.files}
        q=case['X']@theta.T;unchanged=np.array_equal(np.argmax(q,axis=1),case['actions'])
        if not unchanged:assert item['kind']!='guard';continue
        case['q']=q;name=f'reused_features/{len(reused):02d}.npz';np.savez(directory/name,**case)
        reused.append(dict(seed=item['seed'],phase=item['phase'],kind=item['kind'],feature_file=name))
    memory=[]
    for item in state['donors']:
        rr=base.read(item['reward_receipt']);assert rr['complete'] and rr['observations_exact'] and rr['report']['hits']>0
        name=f'memory_donors/{len(memory):02d}.npz';(directory/name).write_bytes(Path(item['feature_file']).read_bytes())
        memory.append(dict(item,feature_file=name))
    old_donors=[v for v in memory if not hits[str(v['seed'])]]
    misses={int(v) for v,h in hits.items() if not h};known=set()
    for root in base.EXPERIMENTS.glob('pong-*-20261025-r*'):
        for path in (root/'episodes').glob('*.json'):
            rr=base.read(path);seed=rr.get('seed')
            if seed not in misses or not rr.get('complete') or not rr.get('observations_exact') or not rr.get('first_hit_cutoff') or rr['report']['hits']<1:continue
            phase='left' if seed in bank['left'] else 'right';standard=base.job(directory,'check',seed,'carry.npy',phase,'train')
            if base.config_key(rr['config'])==base.config_key(standard['config']):known.add(seed)
    targets={};selection={};pools={};target_counts=target_counts or dict(left=16,right=16)
    for p in bank:
        recent=list(latest[p]);never=sorted(v for v in bank[p] if v in misses-known and v not in recent)
        other=sorted(v for v in bank[p] if v in misses and v not in recent and v not in never)
        pools[p]=recent+never+other;targets[p]=pools[p][:target_counts[p]]
        selection[p]=dict(latest=sum(v in recent for v in targets[p]),persistent_never_hit=sum(v in never for v in targets[p]),known_other_success=sum(v in other for v in targets[p]))
    guards=[v['seed'] for v in reused if v['kind']=='guard']+[v['seed'] for v in new_guards]
    assert len(guards)==len(set(guards)) and all(hits[str(v)] for v in guards)
    assert len({v%base.MODULUS for seeds in bank.values() for v in seeds})==sum(map(len,bank.values()))
    return dict(bank=bank,hits=hits,fallback=fallback,targets=targets,target_pools=pools,guards=guards,new_guards=new_guards,reused=reused,
        donor_memory=memory,reused_donors=old_donors,target_selection=selection,previous_probe_reclassified_as_TRAIN=True,
        included_evaluation=evaluation_record,evaluation_queries=evaluation_queries)


def export_state(directory,m,s,hits,probe,donor_reports,capture_reports,chosen_donors=None):
    directory=Path(directory);memory={v['seed']:absolute_case(directory,v) for v in m.get('donor_memory',[])}
    for r in donor_reports.values():
        memory[r['seed']]=dict(seed=r['seed'],phase=r['phase'],feature_file=str(directory/r['feature_file']),reward_receipt=str(directory/'episodes'/(r['id']+'.json')))
    for entry in chosen_donors or []:memory[entry['seed']]=absolute_case(directory,entry)
    current={v['seed']:absolute_case(directory,v) for v in m['reused']}
    for r in capture_reports.values():
        if r.get('capture_kind')=='guard':current[r['seed']]=dict(seed=r['seed'],phase=r['phase'],kind='guard',feature_file=str(directory/r['feature_file']))
    state=dict(bank=m['bank'],hits=hits,prior_hits=m['carry_hits'],theta=str(directory/'selected_theta.npy'),weights=str(directory/'selected.npy'),
        prior_theta=str(directory/'carry_theta.npy'),prior_weights=str(directory/'carry.npy'),donors=list(memory.values()),current_cases=list(current.values()),probe=probe,probe_seeds=s['probe'])
    base.atomic_json(directory/'learning-state.json',state)
