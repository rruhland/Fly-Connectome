"""C2 learning comparison against all cached C1 starts and the preserved baseline."""
import argparse
from pathlib import Path
import time
from stateful_pong_comparison import prepare as base_prepare,native_task,comparison
from run_rally_option_memory import ROOT,BANK,read,write,sha,parallel_jobs
from run_rally_raw_fresh_screen import paired_metrics
import run_practical_pong_v2 as base

DEST=ROOT/'pong-smdp-cycle2-comparison-20261029-r1'
CANDIDATE=ROOT/'pong-smdp-feedback2-refit-20261029-r1/final-forest.npz'
REFERENCE=ROOT/'pong-smdp-closed-loop-refit-20261029-r1/final-forest.npz'
REFERENCE_SHA='e58a73f8ba9fe695aa7e1815fc268aaf43bef0759f8a65760fa58bcf8009f2cd'
PREVIOUS=ROOT/'pong-smdp-cycle-comparison-20261029-r1'
ADAPTER='pong_smdp_policy_adapter'


def prepare():
    fit=read(CANDIDATE.parent/'fit-summary.json');assert fit['status']=='fqi_endpoint' and fit['final_iteration']>=1 and fit['final_sha256']==sha(CANDIDATE) and sha(REFERENCE)==REFERENCE_SHA
    old=read(PREVIOUS/'spec.json');assert old['source']==base.source_signature() and all(sha(p)==h for p,h in old['files'].items())
    base_prepare(DEST,CANDIDATE,ADAPTER,63265,[Path(__file__),Path(__file__).with_name('pong_smdp_adaptive_forest.py'),CANDIDATE.parent/'fit-summary.json',CANDIDATE.parent/'spec.json'])
    spec=read(DEST/'spec.json');cached={};paths=sorted((PREVIOUS/'c1-screen8').glob('candidate_*.json'))+sorted((PREVIOUS/'matched32/c1').glob('candidate_*.json'))
    for p in paths:
        r=read(p);assert r['complete'] and r['model_sha256']==REFERENCE_SHA and r['full_stateful_audit']['passed'];cached[str(r['seed'])]=str(p)
        for f in (p,Path(r['feature_file']),Path(r['structured_feature_file'])):spec['pinned_assets'][str(f)]=sha(f)
    assert len(cached)==32 and set(map(int,cached))==set(spec['screen32'])
    spec.update(reference_model=str(REFERENCE),reference_sha256=REFERENCE_SHA,cached_reference_receipts=cached,comparison_seconds=180,fresh_seconds=180,
        qualification='8c2strictly>c1total16/later9withfirst7parity;then32c2vsallcachedc1(total50,later21,second13,first29)and40069(total82);freshonlyc2beats40069total/laterwithfirstparity,secondcount/laterfractionnotworse',
        explicit_second_feedback_cycle=True,no_new_training_or_model_selection=True)
    spec['pinned_assets'][str(REFERENCE)]=REFERENCE_SHA;write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),candidate_sha256=sha(CANDIDATE)),flush=True)


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'comparison-clock.json').exists();assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    began=time.monotonic();deadline=began+180;write(DEST/'comparison-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=180));reference={int(seed):read(path) for seed,path in spec['cached_reference_receipts'].items()};bank={e['seed']:e for e in read(BANK)['bank']};candidate={};summary=dict(status='comparing',new_transition_cameras=0,new_initial_cameras=0,reused_reference_games=32)
    for size in (8,32):
        folder=DEST/f'screen{size}';folder.mkdir();target=spec['screen8' if size==8 else 'screen32'];jobs=[(ADAPTER,str(CANDIDATE),str(folder),seed,i,deadline-1,True) for i,seed in enumerate(target) if seed not in candidate];rs=parallel_jobs(native_task,jobs,deadline,progress=folder/'progress.json');summary['new_transition_cameras']+=sum(r['transitions'] for r in rs);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in rs);candidate.update({r['seed']:r for r in rs if not r.get('not_started')});complete=all(s in candidate and candidate[s]['complete'] for s in target);seeds=[s for s in target if s in candidate];a=[reference[s] for s in seeds];b=[candidate[s] for s in seeds];result=dict(c1_to_c2=comparison(a,b),baseline_to_c2=comparison([read(bank[s]['receipt']) for s in seeds],b),complete=complete);write(DEST/f'comparison{size}.json',result);summary[f'comparison{size}']=result;print(dict(stage=f'comparison{size}',**result),flush=True)
        if not complete:summary['status']=f'comparison{size}_partial';break
        p=result['c1_to_c2'];improved=p['candidate']['own_returns']>p['baseline']['own_returns'] and p['candidate_more']['later_returns']>p['baseline_more']['later_returns'] and p['candidate']['first_incoming_returns']==p['baseline']['first_incoming_returns']
        if size==8 and not improved:summary['status']='no_learning_gain8';break
        if size==32:
            summary.update(matched32_learning_gain=improved,comparison_wall_seconds=time.monotonic()-began,comparison_post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline));q=result['baseline_to_c2'];gate=q['candidate']['own_returns']>q['baseline']['own_returns'] and q['candidate_more']['later_returns']>q['baseline_more']['later_returns'] and q['candidate']['first_incoming_returns']==q['baseline']['first_incoming_returns'] and q['candidate_more']['tail']['2']>=q['baseline_more']['tail']['2'] and q['candidate_more']['later_fraction']>=q['baseline_more']['later_fraction'];summary.update(fresh_gate=gate,status='learning_gain_without_baseline_qualification' if improved else 'TRAIN32_without_learning_gain')
            if gate:
                fresh=DEST/'fresh';fb=time.monotonic();fd=fb+180;write(fresh/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=fb,deadline_monotonic=fd,seconds=180,separate_authorized_qualification=True));seeds=read(fresh/'schedules.json')['probe'];jobs=[(ADAPTER,str(CANDIDATE),str(fresh),seed,i,fd-1,c) for i,seed in enumerate(seeds) for c in (False,True)];rs=parallel_jobs(native_task,jobs,fd,progress=fresh/'progress.json');summary['new_transition_cameras']+=sum(r['transitions'] for r in rs);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in rs);groups={c:{r['seed']:r for r in rs if not r.get('not_started') and (r['model_sha256'] is not None)==c} for c in (False,True)};complete=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values());fs=dict(complete=complete,wall_seconds=time.monotonic()-fb,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-fd),groups={str(c):dict(games=len(g),returns=sum(r['report']['hits'] for r in g.values())) for c,g in groups.items()})
                if complete:
                    a=[groups[False][s] for s in seeds];b=[groups[True][s] for s in seeds];fs.update(comparison(a,b),paired=paired_metrics(a,b))
                write(fresh/'summary.json',fs);summary.update(fresh=fs,status='fresh_complete' if complete else 'fresh_partial')
    summary.setdefault('comparison_wall_seconds',time.monotonic()-began);summary.setdefault('comparison_post_deadline_audit_cleanup_seconds',max(0.,time.monotonic()-deadline));summary.update(wall_seconds=time.monotonic()-began,source_frozen=all(sha(p)==h for p,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['pinned_assets'].items()),no_training=True,no_automatic_promotion=True);write(DEST/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    if a.operation=='prepare':prepare()
    else:run(a.approved_spec_sha256)
