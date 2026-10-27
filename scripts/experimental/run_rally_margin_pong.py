"""Explicit fit-function adapter: same original data and native comparison runner."""
import argparse
from pathlib import Path
import run_practical_pong_v2 as base
import run_rally_imitation_pong as runner
from pong_rally_margin import fit


def pins():
    return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_margin_pong.py','pong_rally_margin.py','run_rally_imitation_pong.py','pong_rally_imitation.py','run_rally_baseline_pong.py','pong_rewarded_residual.py')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--discovery',default='/workspace/m1b-experiments/pong-rally-discovery-20261027-r1');a=p.parse_args()
    # Only the closed-form fitter is substituted. Dataset construction, grouping,
    # immutable carry, native64 starts and frozen evaluation are reused unchanged.
    runner.fit=fit;runner.pins=pins
    runner.run(a.directory,a.discovery)
