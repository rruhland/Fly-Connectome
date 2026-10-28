"""Manifest-driven new TRAIN collection from fixed own-policy repertoire."""
import argparse,json,multiprocessing as mp,re,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from pong_fixed_rbf import load_readout
from run_rally_raw_optimizer import native_task
from run_rally_fresh_screen import metrics
import run_practical_pong_v2 as base

FILES=('run_rally_repertoire_collection.py','run_rally_raw_optimizer.py','rally_raw_capture.py','run_rally_fresh_screen.py','return_metrics.py','pong_fixed_rbf.py','pong_fixed_pairwise.py')


def select_starts(root,tag,excluded,per_phase=32):
    used=set(excluded);used.add(root%base.MODULUS);rng=np.random.default_rng(np.random.SeedSequence([root,tag]));buckets=dict(left=[],right=[]);rejected=[]
    while min(map(len,buckets.values()))<per_phase:
        seed=int(rng.integers(1,2**32,dtype=np.uint64));phase='left' if base.leftward(seed) else 'right';canonical=seed%base.MODULUS
        reason='prior_or_reserved_canonical_seed' if canonical in used else 'phase_quota' if len(buckets[phase])==per_phase else None;used.add(canonical)
        if reason:rejected.append(dict(seed=seed,reason=reason))
        else:buckets[phase].append(seed)
    seeds=[s for pair in zip(buckets['left'],buckets['right']) for s in pair]
    assert len(set(s%base.MODULUS for s in seeds))==2*per_phase and not {s%base.MODULUS for s in seeds}&set(excluded)
    return dict(TRAIN=seeds,phases=buckets,rejected=rejected,root=root,tag=tag,selection='uniform ordinary starts with predeclared phase quotas; interleaved left/right; no outcome filtering')


def exclusions(directory):
    used=set();proof={}
    def collect(v):
        if type(v) is int and 0<v<2**32:used.add(v%base.MODULUS)
        elif isinstance(v,list):
            for a in v:collect(a)
        elif isinstance(v,dict):
            for a in v.values():collect(a)
    paths=set(base.EXPERIMENTS.rglob('*schedule*.json'))|set(base.EXPERIMENTS.rglob('*manifest*.json'))|set(base.EXPERIMENTS.rglob('seed-exclusions.json'))|set(base.EXPERIMENTS.rglob('admission.json'))
    for p in sorted(paths):collect(base.read(p));proof[str(p)]=base.sha(p)
    # Include actual historical games even if an ad-hoc run had no schedule.
    # Seed is in every receipt's first8192bytes; hash that exact examined header.
    receipts={};pattern=re.compile(rb'"seed"\s*:\s*(\d+)')
    import hashlib
    for p in sorted(base.EXPERIMENTS.rglob('episodes/*.json')):
        with p.open('rb') as stream:header=stream.read(8192)
        match=pattern.search(header);assert match is not None, ('receipt_seed_not_in_header',str(p));seed=int(match.group(1));assert 0<seed<2**32;used.add(seed%base.MODULUS);receipts[str(p)]=dict(seed=seed,header8192_sha256=hashlib.sha256(header).hexdigest(),bytes=p.stat().st_size)
    collect(list(base.old_engine.FINAL));base.atomic_json(directory/'seed-exclusion-proof.json',dict(metadata_sha256=proof,actual_episode_headers=receipts,canonical_ids=sorted(used),ids=len(used),new_observations=0));return used


def run(directory,spec_path):
    directory=Path(directory);spec_path=Path(spec_path);assert not directory.exists();directory.mkdir();began=time.monotonic();spec=base.read(spec_path);assert spec['purpose']=='NEW_TRAIN_own_repertoire_no_fitting' and spec['seconds']==600 and spec['starts_per_phase']==32 and len(spec['heads'])==2;deadline=began+spec['seconds'];base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=600));(directory/'collection-spec.json').write_bytes(spec_path.read_bytes());files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};source=base.source_signature()
    excluded=exclusions(directory);schedule=select_starts(spec['seed_root'],spec['seed_tag'],excluded,32);base.atomic_json(directory/'schedules.json',schedule);seeds=schedule['TRAIN'];reference=None;heads={}
    for head,entry in spec['heads'].items():
        assert head.isidentifier();root=Path(entry['root']);assert base.read(root/'completion-verification.json')['passed'] and base.sha(root/'rbf-model.npz')==entry['policy_sha256'] and base.sha(root/'proposal.npy')==entry['base_weights_sha256'];policy=load_readout(root/'rbf-model.npz');assert np.array_equal(policy.base.weights,np.load(root/'proposal.npy')) and policy.basis.output_dim==512;cap=base.read(root/'radial-cap.json')
        if reference is None:reference=(policy,cap)
        else:assert cap==reference[1] and all(np.array_equal(getattr(policy.basis,n),getattr(reference[0].basis,n)) for n in ('hashes','signs','scale','mean','rms'))
        folder=directory/head;(folder/'native_candidate/episodes').mkdir(parents=True);(folder/'native_candidate/features').mkdir()
        for name in ('proposal.npy','rbf-model.npz','radial-cap.json'):(folder/name).write_bytes((root/name).read_bytes())
        heads[head]=dict(entry,radial_cap_sha256=base.sha(folder/'radial-cap.json'))
    manifest=dict(purpose=spec['purpose'],source=source,files=files,spec_path=str(spec_path),spec_sha256=base.sha(spec_path),heads=heads,seeds=seeds,starts=64,planned_native_games=128,seconds=600,workers=4,settings='native20s/.65opponent/K1/defaultlegalcamera/unchangednative_task',outcome_selection=False,fitting=False,existing_fresh_admitted=False,new_TRAIN_contexts=True,admission_requires_completed_receipt_review=True)
    base.atomic_json(directory/'manifest.json',manifest);results={h:{} for h in heads};errors=[];submitted=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter((head,i,seed) for i,seed in enumerate(seeds) for head in heads);active={}
        def fill():
            while not errors and len(active)<4 and time.monotonic()<deadline-2:
                try:head,i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(native_task,str(directory/head),seed,i,deadline-1)]=(head,i,seed);submitted.append((head,i,seed))
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                head,i,seed=active.pop(future)
                try:results[head][i]=future.result()
                except Exception as e:errors.append(dict(head=head,index=i,seed=seed,error=repr(e)))
            base.atomic_json(directory/'progress.json',dict(purpose='NEW_TRAIN',saved_games=sum(map(len,results.values())),complete_games=sum(r['complete'] for v in results.values() for r in v.values()),policy_cameras=sum(r['transitions'] for v in results.values() for r in v.values()),submitted_games=len(submitted),errors=errors));fill()
    # Re-read saved receipts, including any saved immediately before worker error.
    results={head:{i:base.read(p) for i in range(64) if (p:=directory/head/'native_candidate/episodes'/f'train_{i:02d}.json').exists()} for head in heads};groups={head:metrics([v[i] for i in sorted(v) if v[i]['complete']]) for head,v in results.items()};complete=not errors and all(len(v)==64 and all(r['complete'] for r in v.values()) for v in results.values());changes=[]
    first,second=heads
    for i in sorted(set(results[first])&set(results[second])):
        a,b=results[first][i],results[second][i];assert a['seed']==b['seed']==seeds[i]
        if a['complete'] and b['complete'] and a['report']['hits']!=b['report']['hits']:changes.append(dict(seed=seeds[i],first_head=first,first_hits=a['report']['hits'],second_head=second,second_hits=b['report']['hits']))
    hashes_frozen=all(base.sha(directory/h/'rbf-model.npz')==entry['policy_sha256'] and base.sha(directory/h/'proposal.npy')==entry['base_weights_sha256'] and base.sha(directory/h/'radial-cap.json')==entry['radial_cap_sha256'] for h,entry in heads.items());unknown_errors=[e for e in errors if e['index'] not in results[e['head']]]
    summary=dict(status='complete' if complete else 'native_error' if errors else 'bounded_partial',purpose=spec['purpose'],groups_complete_episodes=groups,completed_games={h:sum(r['complete'] for r in v.values()) for h,v in results.items()},saved_games={h:len(v) for h,v in results.items()},per_seed_return_differences=changes,new_transition_cameras=sum(r['transitions'] for v in results.values() for r in v.values()),new_initial_cameras=sum(r['initial_cameras'] for v in results.values() for r in v.values()),unknown_unsaved_failed_native_cost=bool(unknown_errors),errors=errors,wall_seconds=time.monotonic()-began,source_frozen=source==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},weights_frozen=hashes_frozen,spec_unchanged=base.sha(spec_path)==manifest['spec_sha256'],no_fitting=True,no_existing_fresh_admission=True,no_new_bank_admission_yet=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--manifest',required=True);a=p.parse_args();run(a.directory,a.manifest)
