"""One saved-data solver comparison, without new game observations."""
import argparse,json,time
from pathlib import Path
import numpy as np
from pong_earned_trace_merge import merge
from run_rally_earned_merge import load_case
import run_practical_pong_v2 as base


def run(directory,index_path,baseline_report):
    directory=Path(directory);index_path=Path(index_path);baseline_report=Path(baseline_report);assert not directory.exists();directory.mkdir()
    index=base.read(index_path);assert base.sha(index['reference_head'])==index['reference_sha256'];cases=[load_case(e) for e in index['cases']];normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_merge_solver_check.py','pong_earned_trace_merge.py','run_rally_earned_merge.py')}
    manifest=dict(source=base.source_signature(),files=files,index=str(index_path),index_sha256=base.sha(index_path),reference_sha256=index['reference_sha256'],baseline_report=str(baseline_report),baseline_report_sha256=base.sha(baseline_report),seconds=60,changes='coarse intermediate solves, mapped dual warm starts, equivalent row normalization; same objective and constraints; strict original-logit and greedy-choice acceptance',no_new_cameras=True)
    base.atomic_json(directory/'manifest.json',manifest);start=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=start,deadline_monotonic=start+60));candidate,report=merge(np.load(index['reference_head']),cases,np.load(index['feature_metric']),normal['kappa'],start+60,efficient_working_set=True);base.atomic_json(directory/'merge-report.json',report)
    if candidate is not None:base.save_array(directory/'proposal.npy',candidate)
    frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files};summary=dict(status='verified_saved_data_merge' if candidate is not None else 'bounded_no_verified_merge',baseline=base.read(baseline_report),optimized=report,new_observations=0,source_frozen=frozen,proposal_sha256=base.sha(directory/'proposal.npy') if candidate is not None else None);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],source_frozen=frozen,proposal_sha256=summary['proposal_sha256'],wall_seconds=report['wall_seconds'],rounds=len(report.get('history',[])),last_round=report.get('history',[None])[-1],new_observations=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--index',required=True);p.add_argument('--baseline-report',required=True);a=p.parse_args();run(a.directory,a.index,a.baseline_report)
