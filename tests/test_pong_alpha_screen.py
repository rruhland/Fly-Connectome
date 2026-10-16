import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('alpha_screen',Path(__file__).parents[1]/'scripts/experimental/run_pong_alpha_screen.py')
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)


def report(wins,truncations=0,complete=True,episodes=12):
    return dict(wins=wins,truncations=truncations,complete=complete,episodes=episodes)


def test_prespecified_selection_requires_complete_rallies_and_rejects_truncation():
    assert screen.choose_checkpoint([report(2),report(2)]) is None
    assert screen.choose_checkpoint([report(4,truncations=1),report(2)]) is None
    assert screen.choose_checkpoint([report(4,complete=False,episodes=6),report(3)])==(3,6000)
    assert screen.choose_checkpoint([report(3),report(3)])==(3,2000)
    assert screen.choose_checkpoint([report(3),report(4)])==(4,6000)


def test_learning_changes_only_alpha_and_preserves_seeds():
    jobs=screen.declared_jobs();learning=[job for job in jobs if job['kind']=='learn']
    assert {job['seed'] for job in learning}==set(screen.TRAIN)
    assert all(job['alpha']==.001 and job['lam']==.995 for job in learning)
    assert all('scaling_from' not in job for job in jobs)
    assert {job['config']['seed'] for job in jobs if job['kind']=='evaluate'}=={screen.DEV}
