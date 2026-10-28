"""Refresh existing TRAIN receipts, then merge complete rewarded discoveries."""
import argparse
from pathlib import Path
from return_metrics import aggregate
from run_rally_raw_optimizer import more
from run_rally_earned_merge import run as merge_run,donors_in
import run_practical_pong_v2 as base


def run(directory,discovery,reference_root,bank_path,refreshed_bank,feature_metric):
    directory=Path(directory);discoveries=[Path(p) for p in discovery];reference_root=Path(reference_root);bank_path=Path(bank_path);refreshed_bank=Path(refreshed_bank)
    assert not directory.exists() and not refreshed_bank.exists();directory.mkdir();refreshed_bank.parent.mkdir(exist_ok=True)
    bank=base.read(bank_path);head=reference_root/'proposal.npy';head_hash=base.sha(head)
    assert bank['source']==base.source_signature()==base.read(reference_root/'manifest.json')['source'];assert base.read(reference_root/'completion-verification.json')['passed']
    for discovery in discoveries:
        manifest=base.read(discovery/'manifest.json');summary=base.read(discovery/'summary.json');assert summary['status']=='complete' and summary['source_frozen'];assert manifest['source']==bank['source'] and manifest['head_sha256']==head_hash
    current={r['seed']:(p,r) for p in (reference_root/'native_candidate/episodes').glob('*.json') for r in [base.read(p)]}
    assert set(current)=={e['seed'] for e in bank['bank']};entries=[];receipts=[];native_cases={}
    for old in bank['bank']:
        seed=old['seed'];p,r=current[seed];assert r['complete'] and r['weights_sha256']==head_hash
        feature=Path(r['feature_file']);feature=feature if feature.is_absolute() else reference_root/feature
        entries.append(dict(old,receipt=str(p),raw_feature=str(feature),provenance='refreshed completed current-head receipt for an existing TRAIN seed'))
        receipts.append(r);ends=[i+1 for i,v in enumerate(r['records']) if v['hits']]
        if ends:native_cases[seed]=dict(seed=seed,receipt=str(p),raw_feature=str(feature),end=ends[-1],source_head=str(head),source_head_sha256=head_hash)
    refreshed=dict(bank,current_head=str(head),current_head_sha256=head_hash,bank=entries,current_native_aggregate=aggregate([r['annotation'] for r in receipts]),current_more=more(receipts),new_observations=0,previous_bank=str(bank_path),previous_bank_sha256=base.sha(bank_path),completion_receipt=str(reference_root/'completion-verification.json'),guard_deduplication='guards reconstructed directly from current native earned prefixes; this refresh appends no separate witnesses',guard_deduplication_proof=[],refresh=dict(reference_root=str(reference_root),existing_seeds_only=True,fresh_admitted=False,new_observations=0,previous_completion_receipt=bank.get('completion_receipt'),previous_guard_deduplication_proof=bank.get('guard_deduplication_proof'),current_summary_sha256=base.sha(reference_root/'summary.json')))
    base.atomic_json(refreshed_bank,refreshed);targets=[];donors=[]
    for discovery in discoveries:
        for case in base.read(discovery/'summary.json')['cases']:
            for donor in donors_in(case):
                seed=donor['seed'];assert seed in current and seed not in targets;targets.append(seed)
                native_cases.pop(seed,None)
                donors.append(dict(donor,source_head_sha256=base.sha(donor['source_head']),source_native_root=str(reference_root),discovery_summary=str(discovery/'summary.json')))
    assert targets
    cases=donors+list(native_cases.values());index=dict(cases=cases,targets=targets,reference_head=str(head),reference_sha256=head_hash,feature_metric=str(feature_metric),rows=sum(c['end'] for c in cases),new_observations=0,reward_tradeoff='replace each target old earned prefix with its complete strictly better reward-verified donor; preserve every other current earned prefix',fresh_admitted=False)
    index_path=directory/'merge-index.json';base.atomic_json(index_path,index)
    merge_run(directory,index_path,refreshed_bank,reference_root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--discovery',required=True,action='append');p.add_argument('--reference-root',required=True);p.add_argument('--bank',required=True);p.add_argument('--refreshed-bank',required=True);p.add_argument('--feature-metric',required=True);a=p.parse_args();run(a.directory,a.discovery,a.reference_root,a.bank,a.refreshed_bank,a.feature_metric)
