"""Reload the complete learner at every new game and replay every actual probe and visible command."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import mujoco
import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_continual_wrist import ContinualWrist
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm
from .tt24_ready import prepare_step
from .tt24_single_arm_course import CAP


def verify(index):
    with gzip.open(ROOT/f"single-arm-course-{index}.json.gz","rt") as f:
        source=json.load(f)
    policy=ContinualWrist()
    target=ROOT/f"continual-wrist-{index}.npz"
    arms={b:InstalledArm(b) for b in ("stock","alternate")}
    for a in arms.values():
        a.arena.max_steps=CAP
    server=BounceArena(view="none")
    verified=[]
    try:
        for row in source["rows"]:
            plan=row["plan"]
            expected=row["modes"]["adaptive"]
            wire=np.array(source["wires"][row["wire_index"]])
            arm=arms[plan["body"]]
            env=arm.arena
            server.reset(seed=plan["launcher_seed"],split="held",render=False)
            location=server._ball_position.copy()
            obs=arm.reset(plan,location,True)
            count=0
            def send(request):
                nonlocal count
                physical=np.clip(wire@request,-1.,1.)
                if not np.array_equal(physical,expected["inspection"][count]):
                    raise ValueError("inspection action changed after checkpoint restore")
                prepare_step(env,physical,location)
                count+=1
                return env._observation(False)["joints"]
            if policy.open_release(obs):
                policy.inspect(obs["joints"],send,plan["calibration_rng"])
            if count!=len(expected["inspection"]):
                raise ValueError("changed number of setup actions")
            obs=env._observation(True)
            if not np.array_equal(obs["joints"],expected["after_setup_q"]):
                raise ValueError("starting posture mismatch")
            policy.begin_play(obs)
            lead=plan["prep_ticks"]
            changes=dict(plan["changes"])
            for tick,recorded in enumerate(expected["actions"]):
                if tick-lead in changes:
                    arm.switch_gravity(tick-lead,changes[tick-lead])
                request=policy.act(obs)
                physical=np.clip(wire@request,-1.,1.)
                if not np.array_equal(physical,recorded):
                    raise ValueError(f"game action mismatch, stream {index} seed {plan['learner_seed']} tick {tick}")
                if tick<lead:
                    prepare_step(env,physical,location)
                    if tick+1==lead:
                        env._ball_velocity[:]=server._ball_velocity
                        env._ball_spin[:]=server._ball_spin
                        mujoco.mj_forward(env._model,env._data)
                    obs=env._observation(True)
                else:
                    obs,_,_=env.step(physical)
                policy.observe(request,obs)
            if digest_json(env.judge_metrics())!=digest_json(expected["metrics"]):
                raise ValueError("stored policy changed the physical result")
            policy.save_between_releases(target)
            policy=ContinualWrist(target)
            verified.append(dict(setup_ticks=count,game_ticks=len(expected["actions"]),seed=plan["learner_seed"]))
        if len(verified)!=12 or not np.array_equal(policy.bank.maps,source["maps"]):
            raise ValueError("incomplete restored course")
        with open(ROOT/f"continual-wrist-audit-{index}.json","w") as f:
            json.dump(dict(games=verified,bytes=target.stat().st_size),f,indent=2)
    finally:
        server.close()
        for a in arms.values():
            a.arena.close()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(verify,range(3)))
