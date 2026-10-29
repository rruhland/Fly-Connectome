"""Explicit recovery of missing frozen games; original receipts/clock stay immutable."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time
from run_rally_ogma_empty_guard import DEST as ORIGINAL,task,metrics,read,write,sha,BANK
from run_rally_raw_fresh_screen import paired_metrics
import run_practical_pong_v2 as base

DEST=ORIGINAL/'recovery-20261029-r1'


def run():
    assert not DEST.exists();DEST.mkdir();spec=read(ORIGINAL/'spec.json')
    assert all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    available={};provenance=[]
    for group in ('screen8','screen32'):
        for p in sorted((ORIGINAL/group).glob('game_*.json')):
            r=read(p);assert r['complete'] and r['policy_id']==spec['policy_id'];available[r['seed']]=r;provenance.append(dict(path=str(p),sha256=sha(p)))
    jobs=[dict(index=i,seed=s) for i,s in enumerate(spec['screen32']) if s not in available]
    assert len(available)==17 and len(jobs)==15
    orphan=ORIGINAL/'screen32/game_16-raw.npz'
    manifest=dict(purpose='explicit120s_recovery_only15missing_frozen_TRAIN_games',original_spec_sha256=sha(ORIGINAL/'spec.json'),policy_id=spec['policy_id'],
        available=provenance,jobs=jobs,source_sha256=sha(__file__),original_clock=read(ORIGINAL/'work-clock.json'),
        original_saved_complete_transition_cameras=10274,original_saved_complete_initials=17,
        original_orphan_raw=dict(path=str(orphan),sha256=sha(orphan),transition_rows=166,initials=1,unverified_outcome=True),
        original_saved_transition_camera_lower_bound=10440,original_saved_initial_camera_lower_bound=18,
        interrupted_unsaved_work='UNKNOWN; never equate saved lower bound with all executed work',
        recovery_seconds=120,fresh_seconds_if_strict_gate=180,no_training=True,no_new_warmup=True,no_refit=True,no_policy_source_changes=True)
    write(DEST/'manifest.json',manifest);began=time.monotonic();deadline=began+120
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=120,separate_authorized_allocation=True))
    folder=DEST/'missing_TRAIN';folder.mkdir();bank={e['seed']:e for e in read(BANK)['bank']};new=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        for offset in range(0,len(jobs),4):
            if time.monotonic()>=deadline-2:break
            futures=[pool.submit(task,str(folder),j['seed'],j['index'],deadline,spec['policy_id'],True,bank[j['seed']]['receipt']) for j in jobs[offset:offset+4]]
            new.extend(r for r in [f.result() for f in futures] if r is not None)
            write(DEST/'progress.json',dict(completed_jobs=len(new),complete_games=sum(r['complete'] for r in new),new_transition_cameras=sum(r['transitions'] for r in new),new_initial_cameras=sum(r['initial_cameras'] for r in new),elapsed=time.monotonic()-began))
        available.update({r['seed']:r for r in new});candidate=[available[s] for s in spec['screen32'] if s in available]
        a=metrics([read(bank[s]['receipt']) for s in spec['screen32']]);b=metrics(candidate);complete=len(candidate)==32 and all(r['complete'] for r in candidate)
        gate=complete and b['aggregate']['own_returns']>a['aggregate']['own_returns'] and b['more']['later_returns']>a['more']['later_returns'] and b['aggregate']['first_incoming_returns']>=a['aggregate']['first_incoming_returns'] and b['more']['tail']['2']>=a['more']['tail']['2'] and b['more']['later_fraction']>=a['more']['later_fraction']
        summary=dict(status='TRAIN32_complete' if complete else 'TRAIN32_recovery_partial',baseline=a,candidate=b,complete=complete,fresh_gate=gate,
            reused_completed_games=17,recovery_complete_games=sum(r['complete'] for r in new),recovery_transition_cameras=sum(r['transitions'] for r in new),recovery_initial_cameras=sum(r['initial_cameras'] for r in new),
            recovery_wall_seconds=time.monotonic()-began,recovery_post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),
            sequential_audit_rows=sum(r.get('sequential_audit_rows',0) for r in candidate),baseline_empty_prefix_action_Q_hits_exact_rows=sum(r.get('baseline_prefix_action_Q_hits_exact_rows',0) for r in candidate),
            original_saved_camera_lower_bound=10440,original_saved_initial_lower_bound=18,original_unknown_interrupted_work=True,policy_id=spec['policy_id'],no_training=True,no_automatic_promotion=True)
        write(DEST/'TRAIN32-summary.json',summary);print(dict(stage='TRAIN32',**summary),flush=True)
        if gate:
            fresh=DEST/'fresh';fresh.mkdir();(fresh/'schedules.json').write_bytes((ORIGINAL/'fresh/schedules.json').read_bytes());seeds=read(fresh/'schedules.json')['probe'];groups={'carry':[],'candidate':[]};folders={h:fresh/h for h in groups}
            for f in folders.values():f.mkdir()
            fresh_began=time.monotonic();fresh_deadline=fresh_began+180;write(fresh/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=fresh_began,deadline_monotonic=fresh_deadline,seconds=180,separate_authorized_qualification=True,unchanged_reserved_schedule_sha256=sha(ORIGINAL/'fresh/schedules.json')))
            pending=[(head,i,seed) for i,seed in enumerate(seeds) for head in ('carry','candidate')]
            for offset in range(0,len(pending),4):
                if time.monotonic()>=fresh_deadline-2:break
                futures=[(head,pool.submit(task,str(folders[head]),seed,i,fresh_deadline,spec['policy_id'],head=='candidate')) for head,i,seed in pending[offset:offset+4]]
                for head,future in futures:
                    r=future.result()
                    if r is not None:groups[head].append(r)
                write(fresh/'progress.json',dict(games={h:len(g) for h,g in groups.items()},new_transition_cameras=sum(r['transitions'] for g in groups.values() for r in g),elapsed=time.monotonic()-fresh_began))
            for learned,baseline in zip(groups['candidate'],groups['carry']):
                assert learned['seed']==baseline['seed'];n=learned['initial_empty_prefix']
                for r,expected in zip(learned['records'][:n],baseline['records'][:n]):assert all(r[k]==expected[k] for k in ('action','q','hits'))
            complete=all(len(g)==32 and all(r['complete'] for r in g) for g in groups.values())
            fs=dict(status='complete' if complete else 'bounded_partial',groups={h:metrics(g) for h,g in groups.items()},paired=paired_metrics(groups['carry'],groups['candidate']) if complete else None,
                new_transition_cameras=sum(r['transitions'] for g in groups.values() for r in g),new_initial_cameras=sum(r['initial_cameras'] for g in groups.values() for r in g),wall_seconds=time.monotonic()-fresh_began,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-fresh_deadline))
            write(fresh/'summary.json',fs);summary['fresh']=fs;summary['status']='fresh_'+fs['status'];print(dict(stage='fresh',**fs),flush=True)
    summary.update(source_frozen=all(sha(p)==h for p,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['pinned_assets'].items()),recovery_orchestrator_frozen=sha(__file__)==manifest['source_sha256'],reused_receipts_untouched=all(sha(p['path'])==p['sha256'] for p in provenance),orphan_untouched=sha(orphan)==manifest['original_orphan_raw']['sha256'])
    write(DEST/'summary.json',summary)


if __name__=='__main__':run()
