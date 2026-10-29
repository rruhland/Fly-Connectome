import pytest
from run_relational_replication_validation import require_learned_endpoints,ordered_assignments,replica_summary


def test_both_learned_endpoints_are_required_before_any_validation():
    e=dict(acquisition_complete=True,learned_endpoint=True,model_path='frozen.npz')
    assert len(require_learned_endpoints(dict(both_endpoints_frozen=True,endpoints={'0':e,'1':e})))==2
    with pytest.raises(ValueError):require_learned_endpoints(dict(both_endpoints_frozen=False,endpoints={'0':e,'1':e}))
    baseline=dict(e,learned_endpoint=False,model_path=None)
    with pytest.raises(ValueError):require_learned_endpoints(dict(both_endpoints_frozen=True,endpoints={'0':e,'1':baseline}))


def test_duration_jobs_interleave_replicas_and_reserve_exact_worst_case():
    schedules={r:dict(fresh=dict(assignments=[dict(index=i,seed=int(r)*100+i,
        stratum='long60' if i<32 else 'standard20',max_transition_cameras=3000 if i<32 else 1000)
        for i in range(64)])) for r in ('0','1')}
    jobs=ordered_assignments(schedules)
    assert len(jobs)==128 and sum(j['max_transition_cameras'] for j in jobs)==256000
    for index in range(0,128,2):
        assert [j['replica'] for j in jobs[index:index+2]]==['0','1']
        assert jobs[index]['index']==jobs[index+1]['index']
    assert all(j['stratum']=='long60' for j in jobs[:64])
    assert all(j['stratum']=='standard20' for j in jobs[64:])


def row(stratum,first=True,later=10,miss_later=0):
    seconds=60. if stratum=='long60' else 20.
    if not first:later=miss_later=0;seconds=1.
    hits=int(first)+later;opportunities=1+later+miss_later
    annotation=dict(own_opportunities=opportunities,own_returns=hits,first_incoming=dict(success=first),
        maximum_consecutive_returns=hits,seconds=seconds,unfinished_incoming_approach=False,
        opponent_opportunities=0,opponent_returns=0,post_own_opponent_opportunities=0,post_own_opponent_returns=0,
        natural_serve_win=False)
    bins=[]
    counts=[later,0,0] if stratum=='standard20' else [later//3,later//3,later-2*(later//3)]
    for index,count in enumerate(counts):
        exposure=max(0,min(20,seconds-20*index));first_opp=int(index==0)
        bins.append(dict(exposure_seconds=exposure,own_opportunities=count+first_opp+(miss_later if index==0 else 0),
            own_returns=count+int(index==0 and first),first_opportunities=first_opp,
            first_returns=int(index==0 and first),later_opportunities=count+(miss_later if index==0 else 0),later_returns=count))
    return dict(complete=True,fitness_eligible=True,validation_stratum=stratum,annotation=annotation,
        own_hits=hits,report=dict(truncations=int(first)),time_bins=bins,streaming_audit=dict(checked_rows=100))


def test_first_target_is_combined_but_later_targets_and_exposure_are_separate():
    rows=[row('long60',first=i>=2) for i in range(32)]+[row('standard20',later=2) for _ in range(32)]
    result=replica_summary(rows)
    assert result['combined_first_returns']==62 and result['combined_first_opportunities']==64
    assert result['strata']['long60']['metrics']['first_incoming_return_fraction']<.95
    assert result['meets_declared_point_targets']
    weak=[row('long60',later=5) for _ in range(32)]+[row('standard20',later=2) for _ in range(32)]
    assert not replica_summary(weak)['long_later_exposure_sufficient']
    standard_failure=[row('long60') for _ in range(32)]+[row('standard20',later=1,miss_later=1) for _ in range(32)]
    assert not replica_summary(standard_failure)['meets_declared_point_targets']


def test_missing_game_never_yields_complete_or_point_target_pass():
    rows=[row('long60') for _ in range(32)]+[row('standard20',later=2) for _ in range(31)]
    result=replica_summary(rows)
    assert not result['complete'] and not result['meets_declared_point_targets']
