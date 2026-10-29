"""Prepare/run SMDP via the reusable frozen stateful comparison harness."""
import argparse
from pathlib import Path
from stateful_pong_comparison import ROOT,prepare,run,read,sha

DEST=ROOT/'pong-smdp-native-20261029-r1'
MODEL=ROOT/'pong-smdp-adaptive-forest-20261029-r1/final-forest.npz'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    if a.operation=='prepare':
        fit_summary=MODEL.parent/'fit-summary.json';fit=read(fit_summary)
        assert fit['status']=='fqi_endpoint' and fit['final_iteration']>=1 and fit['final_sha256']==sha(MODEL)
        sources=[Path(__file__),Path(__file__).with_name('pong_smdp_adaptive_forest.py'),fit_summary,MODEL.parent/'spec.json']
        print(prepare(DEST,MODEL,'pong_smdp_policy_adapter',63233,sources),flush=True)
    else:run(DEST,a.approved_spec_sha256)
