"""Fixed c1-versus-c0 learning-trajectory control using the frozen generic task."""
import argparse
from pathlib import Path
import time
from stateful_pong_comparison import prepare as base_prepare,native_task,comparison
from run_rally_option_memory import ROOT,BANK,read,write,sha,parallel_jobs
from run_rally_raw_fresh_screen import paired_metrics
import run_practical_pong_v2 as base

DEST=ROOT/'pong-smdp-cycle-comparison-20261029-r1'
C0=ROOT/'pong-smdp-adaptive-forest-20261029-r1/final-forest.npz'
C1=ROOT/'pong-smdp-closed-loop-refit-20261029-r1/final-forest.npz'
CACHED=ROOT/'pong-smdp-native-20261029-r1'
C0_SHA='ca99b349d1ea26a2036214e37d04e58a2d39a59cf26b44d5aeed7e18484df295'
ADAPTER='pong_smdp_policy_adapter'


def prepare():
    fit=read(C1.parent/'fit-summary.json');assert fit['status']=='fqi_endpoint' and fit['final_iteration']>=1 and fit['final_sha256']==sha(C1) and sha(C0)==C0_SHA
    old=read(CACHED/'spec.json');assert old['source']==base.source_signature() and all(sha(p)==h for p,h in old['files'].items())
    base_prepare(DEST,C1,ADAPTER,63249,[Path(__file__),Path(__file__).with_name('pong_smdp_adaptive_forest.py'),C1.parent/'fit-summary.json',C1.parent/'spec.json'])
    spec=read(DEST/'spec.json');cached={}
    for p in sorted((CACHED/'screen8').glob('candidate_*.json')):
        r=read(p);assert r['complete'] and r['model_sha256']==C0_SHA and r['full_stateful_audit']['passed'];cached[str(r['seed'])]=str(p)
        for path in (p,Path(r['feature_file']),Path(r['structured_feature_file'])):spec['pinned_assets'][str(path)]=sha(path)
    assert len(cached)==8 and set(map(int,cached))==set(spec['screen8'])
    spec.update(c0_model=str(C0),c0_sha256=C0_SHA,cached_c0_receipts=cached,comparison_seconds=180,fresh_seconds=180,
        qualification='first8c1must strictlyimprovec0total/laterwithfirstparity; thenmatched32c0/c1; freshonlyifc1beatscached40069total/laterwithfirstparity,secondcount/laterfractionnotworse',
        explicit_learning_cycle=True,no_new_training_or_model_selection=True)
    spec['pinned_assets'][str(C0)]=C0_SHA;write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),c1_sha256=sha(C1)),flush=True)


def collect(folder,jobs,deadline):
    folder.mkdir();rs=parallel_jobs(native_task,[(ADAPTER,model,str(folder),seed,index,deadline-1,True) for model,seed,index in jobs],deadline,progress=folder/'progress.json');return rs


def tagged_task(head,model,folder,seed,index,deadline):
    result=native_task(ADAPTER,str(model),folder,seed,index,deadline,True)
    result['comparison_head']=head
    return result


def stats(c0,c1,bank):
    return dict(c0_to_c1=comparison(c0,c1),baseline_to_c1=comparison([read(bank[r['seed']]['receipt']) for r in c1],c1))


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'comparison-clock.json').exists()
    assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    began=time.monotonic();deadline=began+180;write(DEST/'comparison-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=180));bank={e['seed']:e for e in read(BANK)['bank']};c0={int(seed):read(path) for seed,path in spec['cached_c0_receipts'].items()};c1={};summary=dict(status='comparing',new_transition_cameras=0,new_initial_cameras=0,cached_c0_games=8)
    for size in (8,32):
        target=spec['screen8' if size==8 else 'screen32'];new=[]
        if size==8:
            jobs=[(str(C1),seed,i) for i,seed in enumerate(target)];rs=collect(DEST/'c1-screen8',jobs,deadline);c1.update({r['seed']:r for r in rs if not r.get('not_started')});new+=rs
        else:
            # Missing c0/c1 games are interleaved under the SAME180s comparison clock.
            folder=DEST/'matched32';folder.mkdir();pending=[(head,model,seed,i) for i,seed in enumerate(target) for head,model,known in [('c0',C0,c0),('c1',C1,c1)] if seed not in known]
            heads={head:folder/head for head in ('c0','c1')}
            for f in heads.values():f.mkdir()
            jobs=[(head,str(model),str(heads[head]),seed,i,deadline-1) for head,model,seed,i in pending]
            rs=parallel_jobs(tagged_task,jobs,deadline,progress=folder/'progress.json');new+=rs
            for r in rs:
                if not r.get('not_started'):(c0 if r['comparison_head']=='c0' else c1)[r['seed']]=r
        summary['new_transition_cameras']+=sum(r['transitions'] for r in new);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in new);complete=all(seed in c0 and seed in c1 and c0[seed]['complete'] and c1[seed]['complete'] for seed in target);common=[s for s in target if s in c0 and s in c1];a=[c0[s] for s in common];b=[c1[s] for s in common];result=stats(a,b,bank);result['complete']=complete;write(DEST/f'comparison{size}.json',result);summary[f'comparison{size}']=result;print(dict(stage=f'comparison{size}',**result),flush=True)
        if not complete:summary['status']=f'comparison{size}_partial';break
        progress=result['c0_to_c1'];improved=progress['candidate']['own_returns']>progress['baseline']['own_returns'] and progress['candidate_more']['later_returns']>progress['baseline_more']['later_returns'] and progress['candidate']['first_incoming_returns']==progress['baseline']['first_incoming_returns']
        if size==8 and not improved:summary['status']='no_learning_gain8';break
        if size==32:
            summary['matched32_learning_gain']=improved;summary['comparison_wall_seconds']=time.monotonic()-began;summary['comparison_post_deadline_audit_cleanup_seconds']=max(0.,time.monotonic()-deadline)
            q=result['baseline_to_c1'];gate=q['candidate']['own_returns']>q['baseline']['own_returns'] and q['candidate_more']['later_returns']>q['baseline_more']['later_returns'] and q['candidate']['first_incoming_returns']==q['baseline']['first_incoming_returns'] and q['candidate_more']['tail']['2']>=q['baseline_more']['tail']['2'] and q['candidate_more']['later_fraction']>=q['baseline_more']['later_fraction'];summary['fresh_gate']=gate;summary['status']='learning_gain_without_baseline_qualification' if improved else 'TRAIN32_complete_without_learning_gain'
            if gate:
                fresh=DEST/'fresh';fb=time.monotonic();fd=fb+180;write(fresh/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=fb,deadline_monotonic=fd,seconds=180,separate_authorized_qualification=True));seeds=read(fresh/'schedules.json')['probe'];jobs=[(ADAPTER,str(C1),str(fresh),seed,i,fd-1,candidate) for i,seed in enumerate(seeds) for candidate in (False,True)];rs=parallel_jobs(native_task,jobs,fd,progress=fresh/'progress.json');summary['new_transition_cameras']+=sum(r['transitions'] for r in rs);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in rs);groups={c:{r['seed']:r for r in rs if not r.get('not_started') and (r['model_sha256'] is not None)==c} for c in (False,True)};complete=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values());fs=dict(complete=complete,wall_seconds=time.monotonic()-fb,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-fd),groups={str(c):dict(games=len(g),returns=sum(r['report']['hits'] for r in g.values())) for c,g in groups.items()})
                if complete:
                    a=[groups[False][s] for s in seeds];b=[groups[True][s] for s in seeds];fs.update(comparison(a,b),paired=paired_metrics(a,b))
                write(fresh/'summary.json',fs);summary.update(fresh=fs,status='fresh_complete' if complete else 'fresh_partial')
    summary.setdefault('comparison_wall_seconds',time.monotonic()-began);summary.setdefault('comparison_post_deadline_audit_cleanup_seconds',max(0.,time.monotonic()-deadline))
    summary.update(wall_seconds=time.monotonic()-began,source_frozen=all(sha(p)==h for p,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['pinned_assets'].items()),no_training=True,no_automatic_promotion=True);write(DEST/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    if a.operation=='prepare':prepare()
    else:run(a.approved_spec_sha256)
