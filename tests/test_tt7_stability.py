from scripts.experimental.tt7_stability import StrokeMemory
from scripts.experimental.tt7_stability_consolidate import consolidate
from scripts.experimental.tt7_stability_residual import StrikeMemory
from scripts.experimental.tt7_stability_trial import restore
from scripts.experimental.tt4_transfer_trial import _write


def test_stroke_memory_learns_and_can_resume():
    memory = StrokeMemory(.073)
    empty = memory.speed()
    for _ in range(8):
        memory.observe(.062, 1.)
        memory.observe(.087, -1.)
    assert memory.speed() < empty
    resumed = StrokeMemory.from_state(memory.state())
    for instance in (memory, resumed):
        instance.observe(.074, 0.)
    assert resumed.state() == memory.state()


def test_retrospective_stroke_memory_uses_public_evidence(tmp_path):
    state = tmp_path / 'checkpoint.json.gz'
    practice = tmp_path / 'practice.json.gz'
    _write(state, dict(speed=.073))
    _write(practice, dict(visits=[dict(points=[
        dict(opportunity=True, public_reward=1., impact=dict(used_speed=.062), evaluator=dict(legal_returns=0)),
        dict(opportunity=False, public_reward=-1., impact=dict(used_speed=.090)),
    ])]))
    learned = restore(state, [practice])
    blank = StrokeMemory(.073)
    assert learned.points == 1
    assert learned.speed() < blank.speed()
    _write(state, dict(speed=.089, stroke_memory=learned.state()))
    erased = restore(state, [], erase=True)
    assert erased.state() == blank.state()


def test_body_map_consolidation_retains_qualified_portable_values(tmp_path):
    initial, later = tmp_path / 'initial.json.gz', tmp_path / 'later.json.gz'
    schema = 'tt6-installation-memory-v1'
    _write(initial, dict(schema=schema, models=[dict(motor_map=1)], speed=.073, bounce=[.62]))
    _write(later, dict(schema=schema, models=[dict(motor_map=1), dict(motor_map=2)], speed=.096, bounce=[.95]))
    state = consolidate(initial, later)
    assert state['models'] == [dict(motor_map=1), dict(motor_map=2)]
    assert state['speed'] == .073
    assert state['bounce'] == [.62]
    assert state['consolidation']['added_models'] == 1


def test_optical_strike_memory_retains_bounded_corrections_by_motor_width():
    memory = StrikeMemory()
    for _ in range(50):
        memory.observe(4, [-.04, .08], -1.)
    assert memory.correction(4).tolist() == [-.028, .026]
    assert memory.correction(5).tolist() == [0., 0.]
    assert StrikeMemory.from_state(memory.state()).state() == memory.state()
