"""One frozen-head bank comparison, optional fixed fresh screen; no fitting."""
import argparse,json,time
from pathlib import Path
from run_rally_guarded_curriculum import evaluate
from run_rally_raw_optimizer import more
from return_metrics import aggregate
import run_practical_pong_v2 as base


def run(directory,bank_path,candidate,cached_native_root=None,fresh_root=None,practical_later_screen=False,native_seconds=500):
    directory=Path(directory);bank_path=Path(bank_path);bank=base.read(bank_path);candidate=Path(candidate);assert not directory.exists();directory.mkdir();assert bank['source']==base.source_signature()
    (directory/'proposal.npy').write_bytes(candidate.read_bytes());(directory/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes());entries=bank['bank'];baseline=[base.read(Path(e['receipt'])) for e in entries];byseed={r['seed']:r for r in baseline}
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_fixed_bank_check.py','run_rally_guarded_curriculum.py','run_rally_raw_optimizer.py','rally_action_equivalence.py')}
    manifest=dict(source=base.source_signature(),files=files,bank_path=str(bank_path),bank_sha256=base.sha(bank_path),baseline_head=bank['current_head'],baseline_sha256=bank['current_head_sha256'],candidate_head=str(candidate),candidate_sha256=base.sha(candidate),cached_native_root=str(cached_native_root) if cached_native_root else None,no_fitting=True,native_seconds=native_seconds,individual_regressions_descriptive=True,conditional_fresh='strictly more total hits, no first-return loss, no lower later-return fraction; descriptive screen, no statistical gate')
    if practical_later_screen:manifest['conditional_fresh']='more total and later returns; opening/per-seed tradeoffs reported, not automatic vetoes'
    base.atomic_json(directory/'manifest.json',manifest);start=time.monotonic();outcomes,cost=evaluate(directory,bank,entries,baseline,directory/'proposal.npy',cached_native_root,native_seconds);complete=len(outcomes)==len(entries) and all(r['complete'] for r in outcomes);a=aggregate([r['annotation'] for r in baseline]);b=aggregate([r['annotation'] for r in outcomes]);am=more(baseline);bm=more(outcomes)
    differences=[dict(seed=r['seed'],before_hits=byseed[r['seed']]['report']['hits'],after_hits=r['report']['hits']) for r in outcomes if r['report']['hits']!=byseed[r['seed']]['report']['hits']]
    eligible=complete and b['own_returns']>a['own_returns'] and (bm['later_returns']>am['later_returns'] if practical_later_screen else b['first_incoming_returns']>=a['first_incoming_returns'] and bm['later_fraction']>=am['later_fraction'])
    frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files}
    summary=dict(status='complete' if complete else 'bounded_partial',baseline=a,candidate=b,baseline_more=am,candidate_more=bm,per_seed_hit_changes=differences,native=cost,conditional_fresh_eligible=eligible,wall_seconds=time.monotonic()-start,source_frozen=frozen,proposal_sha256=base.sha(directory/'proposal.npy'),no_fitting=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);assert all(r['weights_sha256']==manifest['candidate_sha256'] for r in outcomes);base.atomic_json(directory/'completion-verification.json',dict(passed=complete and frozen and base.sha(directory/'proposal.npy')==manifest['candidate_sha256'],native_games=len(outcomes),new_verification_observations=0));print(json.dumps(summary),flush=True)
    if eligible and fresh_root:
        from run_rally_baseline_pong import schedule
        from run_rally_raw_fresh_screen import run as fresh_run
        import run_rally_baseline_pong as schedule_module
        fresh=Path(fresh_root);assert not fresh.exists();fresh.mkdir();schedule_module.ROOT=2915406833;schedule_module.TAG=62897;drawn=schedule();all_seeds=drawn.pop('train');drawn.update(probe=all_seeds[:32],unused_reserved=all_seeds[32:],selection='genuinely fresh unfiltered ordinary full games, fixed declared candidate versus bank carry; no fitting/no admission')
        base.atomic_json(fresh/'schedules.json',drawn);fresh_run(fresh,Path(bank['current_head']).parent,directory)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bank',required=True);p.add_argument('--candidate',required=True);p.add_argument('--cached-native-root');p.add_argument('--fresh-root');p.add_argument('--practical-later-screen',action='store_true');a=p.parse_args();run(a.directory,a.bank,a.candidate,a.cached_native_root,a.fresh_root,a.practical_later_screen)
