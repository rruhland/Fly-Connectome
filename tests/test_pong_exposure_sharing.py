import importlib.util
from pathlib import Path
import sqlite3
import pytest

root=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('exposure_sharing',root/'scripts/experimental/run_pong_exposure_sharing.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def report(wins,hits,*,complete=True):
    return dict(wins=wins,own_hit_followed_wins=hits,complete=complete,episodes=24)


def test_prospective_endpoint_selection_uses_both_lifts_and_ties_to_raw():
    baseline=[report(7,1),report(8,2),report(6,0),report(9,3)]
    assert study.choose_arm([report(12,7),report(13,6)],baseline) is None
    assert study.choose_arm([report(13,7),report(13,7)],baseline)=='raw'
    assert study.choose_arm([report(13,7),report(14,8)],baseline)=='sharing'
    assert study.choose_arm([report(15,9,complete=False),report(13,7)],baseline)=='sharing'


def test_allocation_changes_only_sharing_at_fixed_longer_budget():
    learning,baselines,evaluation=study.declared_jobs()
    assert sum(j['transitions'] for j in learning)==60000
    assert all(j['seed']==172900127 and j['alpha']==.001 and j['lam']==.995 and j['opponent_speed']==.25 for j in learning)
    assert all(j['checkpoints']==[6000,15000,30000] for j in learning)
    assert learning[0]['scaling_from'] is None and learning[1]['scaling_from']=='spatial-transform.json'
    assert len(baselines)==4 and len(evaluation)==2
    assert all(j['config']['episodes']==24 and j['config']['seed']==904271833 for j in baselines+evaluation)
    assert all(j['weights_from'].endswith('at30000.npy') for j in evaluation)


def test_prefix_comparison_checks_actual_records_and_length(tmp_path):
    paths=[tmp_path/name for name in ('a.db','b.db')]
    for path in paths:
        with sqlite3.connect(path) as c:
            c.execute('CREATE TABLE replay_samples(seq INTEGER,record BLOB)')
            c.execute('INSERT INTO replay_samples VALUES(0,?)',(b'{"action":1}',))
    study.verify_prefix(*paths,1)
    with pytest.raises(ValueError,match='prefix differs'):study.verify_prefix(*paths,2)
    with sqlite3.connect(paths[1]) as c:c.execute('UPDATE replay_samples SET record=?',(b'{"action":0}',))
    with pytest.raises(ValueError,match='prefix differs'):study.verify_prefix(*paths,1)
