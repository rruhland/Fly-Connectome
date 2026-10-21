"""Deferred environment-only correctness tests; no fitted/privileged controller."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from fly_connectome.control.pendulum import PendulumCamera,PendulumPhysics,rk4_step


@pytest.mark.parametrize('drive',[-1.,0.,1.])
def test_rk4_matches_independent_high_accuracy_integration(drive):
    cfg=PendulumPhysics();initial=np.array([.08,-.12]);duration=.5
    reference=solve_ivp(lambda t,y:[y[1],8*math.sin(y[0])+12*drive-.1*y[1]],
        [0,duration],initial,method='DOP853',rtol=1e-12,atol=1e-14,max_step=cfg.dt/8).y[:,-1]
    errors=[]
    for scale in (1,2):
        fine=replace(cfg,dt=cfg.dt/scale);angle,velocity=initial
        for _ in range(round(duration/fine.dt)):angle,velocity=rk4_step(angle,velocity,drive,fine)
        errors.append(float(np.max(np.abs(np.array([angle,velocity])-reference))))
    assert errors[0]<1e-7
    if errors[0]>1e-11:assert errors[1]<errors[0]/8


def test_camera_reward_is_actual_survival_and_five_second_bound_is_truncation():
    env=PendulumCamera(739391,max_episode_ticks=7);initial=env.render();steps=[]
    assert initial.dtype==torch.bool and initial.shape==(64,64)
    while not env.needs_reset:steps.append(env.step(1))
    assert [step.physics_ticks for step in steps]==[2,2,3]
    assert sum(step.reward for step in steps)==sum(step.elapsed_seconds for step in steps)==7/120
    assert steps[-1].truncated and not steps[-1].terminated and all(step.hits==0 for step in steps)
    assert torch.equal(steps[-1].frame,env.render())
    with pytest.raises(ValueError,match='reset required'):env.step(1)
    returned=env.reset()
    assert torch.equal(returned,env.render()) and env.episode_ticks==env.decisions==0
    assert env.total_ticks==7 and env.step(1).physics_ticks==2


def test_failure_terminal_takes_precedence_and_returns_actual_final_frame():
    env=PendulumCamera(739391,max_episode_ticks=1)
    env.angle=env.config.failure_angle-1e-4;env.velocity=1.
    result=env.step(1)
    assert result.terminated and not result.truncated and result.physics_ticks==1
    assert result.reward==result.elapsed_seconds==1/120
    assert env.angle>=env.config.failure_angle and torch.equal(result.frame,env.render())


def test_initial_distribution_and_rendered_geometry_are_not_a_stable_hanging_pose():
    observations=[]
    for seed in range(25):
        env=PendulumCamera(seed)
        assert .03<=abs(env.angle)<=.12 and abs(env.velocity)<=.2
        observations.append((env.angle,env.velocity))
        for angle in (-math.pi/3,-.12,-.03,0.,.03,.12,math.pi/3):
            env.angle=angle;frame=env.render()
            bob_x=round(32+24*math.sin(angle));bob_y=round(40-24*math.cos(angle))
            assert frame[40,32] and frame[bob_y,bob_x]
            assert frame.sum()>20 and not frame[0].any() and not frame[-1].any()
    assert any(a<0 for a,v in observations) and any(a>0 for a,v in observations)
    assert len(set(observations))==25


@pytest.mark.parametrize('updates',[dict(dt=0),dict(gravity=float('nan')),dict(torque=-1),
    dict(damping=-1),dict(initial_angle_min=0),dict(initial_angle_max=2)])
def test_invalid_physics_rejected(updates):
    with pytest.raises(ValueError):PendulumCamera(config=replace(PendulumPhysics(),**updates))
