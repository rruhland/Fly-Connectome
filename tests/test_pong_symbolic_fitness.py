import time

from pong_symbolic_fitness import fitness_key,evaluate
from pong_symbolic_program import Program
from run_pong_symbolic_search import CameraBudget


def test_camera_reservations_include_inflight_then_release_unspent_grants():
    budget=CameraBudget(4000)
    assert budget.reserve(1000,3000) and budget.reserve(1000,3000) and budget.reserve(1000,3000)
    assert not budget.reserve(1000,3000)
    budget.finish(1000,dict(transitions=120,initial_cameras=1))
    assert budget.used==120 and budget.reserved==2000 and not budget.reserve(1000,3000)
    budget.finish(1000,dict(transitions=130,initial_cameras=1))
    assert budget.reserve(1000,3000)
    assert budget.used+budget.reserved<=3000
    assert budget.initials==2


def test_fitness_cache_identity_requires_program_seed_and_runtime():
    p=Program.zero().active_hash()
    assert fitness_key(p,1,'pinned')==fitness_key(p,1,'pinned')
    assert len({fitness_key(p,1,'pinned'),fitness_key(p,2,'pinned'),
        fitness_key(p,1,'changed-source'),fitness_key('other-program',1,'pinned')})==4


def test_expired_job_never_constructs_session_or_takes_initial_camera(tmp_path,monkeypatch):
    import pong_symbolic_fitness as worker
    def forbidden(*args,**kwargs):
        raise AssertionError('expired grant must not begin a native game')
    monkeypatch.setattr(worker,'RewardOnlyEvaluator',forbidden)
    r=evaluate(Program.zero().to_dict(),1,tmp_path,time.monotonic()-1,1000,'pinned')
    assert r['not_started'] and not r['fitness_eligible'] and r['transitions']==r['initial_cameras']==0
