"""Fixed16 TRAIN LS-velocity pilot; all candidate games actually executed."""
import argparse,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from pong_velocity_ls import TRANSPORT,synthetic_diagnostic
from run_rally_raw_optimizer import native_task,more
from return_metrics import aggregate
import run_practical_pong_v2 as base


def run(directory,bank_path):
    directory=Path(directory);bank_path=Path(bank_path);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+600;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=600));bank=base.read(bank_path);expected='3ac7f87fbac23abab14e633b38cfe1e50986a8f03bb38b049bed8f4cc840bd5e';assert base.sha(bank['current_head'])==bank['current_head_sha256']==expected and bank['source']==base.source_signature();rows={e['seed']:(e,base.read(Path(e['receipt']))) for e in bank['bank']};chosen=[]
    for left in (True,False):
        misses=sorted(s for s,(e,r) in rows.items() if base.leftward(s)==left and r['report']['hits']>0 and r['report']['losses']>0);assert len(misses)>=4
        chosen.extend(dict(seed=s,group='later_miss',phase='left' if left else 'right') for s in misses[:4])
    for left in (True,False):
        remaining=[s for s,(e,r) in rows.items() if base.leftward(s)==left and s not in {c['seed'] for c in chosen} and r['report']['hits']>0];stable=sorted(remaining,key=lambda s:(not rows[s][1]['report']['truncations'],-rows[s][1]['report']['hits'],s));assert len(stable)>=4;chosen.extend(dict(seed=s,group='stable_high_return',phase='left' if left else 'right') for s in stable[:4])
    assert len(chosen)==len({c['seed'] for c in chosen})==16
    for c in chosen:c.update(receipt=rows[c['seed']][0]['receipt'],baseline_hits=rows[c['seed']][1]['report']['hits'])
    base.atomic_json(directory/'schedules.json',dict(train=[c['seed'] for c in chosen],cases=chosen,selection='4later misses/initial direction by seed;4other successes/direction sorted timeout first then descending returns then seed; fixed before native',TRAIN_not_holdout=True));base.atomic_json(directory/'synthetic-diagnostic.json',synthetic_diagnostic());(directory/'proposal.npy').write_bytes(Path(bank['current_head']).read_bytes());(directory/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes());rule=dict(transport=TRANSPORT,observations=4,estimator='unweighted least-squares position slope over last at most4observed endpoints of each active track',timestamps='nominal vision sample index*.02seconds',units='normalized image units per second',missing_observations='retain last observed fit; use actual nominal sample gaps',reset='clear on scene reset or inactive/new track',extra_game_logic=False);base.atomic_json(directory/'velocity-ls.json',rule);files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_velocity_ls.py','pong_velocity_ls.py','run_rally_raw_optimizer.py','rally_action_equivalence.py')};manifest=dict(source=base.source_signature(),files=files,bank=str(bank_path),bank_sha256=base.sha(bank_path),base_sha256=expected,rule=rule,rule_sha256=base.sha(directory/'velocity-ls.json'),base_head_frozen=True,frozen_vision_scaler_actuator=True,feature_dimension=4389,no_candidate_episode_reuse=True,baseline='prior actual default-transport outcomes only; no claim of candidate action equivalence',fit=False,seconds=600);base.atomic_json(directory/'manifest.json',manifest)
    for name in ('episodes','features'):(directory/'native_candidate'/name).mkdir(parents=True)
    outcomes=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(chosen));active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:i,c=next(pending)
                except StopIteration:break
                active[pool.submit(native_task,str(directory),c['seed'],i,deadline-1)]=c
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);outcomes.append(f.result())
            base.atomic_json(directory/'progress.json',dict(games=len(outcomes),transition_cameras=sum(r['transitions'] for r in outcomes),initial_cameras=sum(r['initial_cameras'] for r in outcomes)));fill()
    baseline=[rows[r['seed']][1] for r in outcomes];a=aggregate([r['annotation'] for r in baseline]);b=aggregate([r['annotation'] for r in outcomes]);am=more(baseline);bm=more(outcomes);complete=len(outcomes)==16 and all(r['complete'] for r in outcomes);useful=complete and b['own_returns']>a['own_returns'] and b['first_incoming_returns']>=a['first_incoming_returns'] and bm['later_returns']>am['later_returns'];groups={}
    for group in ('later_miss','stable_high_return'):
        seeds={c['seed'] for c in chosen if c['group']==group};old=[r for r in baseline if r['seed'] in seeds];new=[r for r in outcomes if r['seed'] in seeds];groups[group]=dict(baseline=aggregate([r['annotation'] for r in old]),candidate=aggregate([r['annotation'] for r in new]),baseline_more=more(old),candidate_more=more(new))
    summary=dict(status='complete' if complete else 'bounded_partial',baseline=a,candidate=b,baseline_more=am,candidate_more=bm,groups=groups,per_seed_hit_changes=[dict(seed=r['seed'],before_hits=rows[r['seed']][1]['report']['hits'],after_hits=r['report']['hits']) for r in outcomes if r['report']['hits']!=rows[r['seed']][1]['report']['hits']],clearly_useful_without_opening_loss=useful,gate='strictly more total and later returns; no first-return count loss; descriptive TRAIN pilot, no automatic learning or promotion',transition_cameras=sum(r['transitions'] for r in outcomes),initial_cameras=sum(r['initial_cameras'] for r in outcomes),newly_executed_games=len(outcomes),reused_candidate_games=0,wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},base_unchanged=base.sha(bank['current_head'])==expected,no_learning=True,no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bank',required=True);a=p.parse_args();run(a.directory,a.bank)
