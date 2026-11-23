"""One live receiving arm inspects its own controls, then releases the held ball without a reset."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import mujoco
import numpy as np

from .tt4_transfer_trial import _write
from .tt19_view_full_game import digest_json
from .tt24_autonomous_drive import MotorMemory
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_effects import transmission
from .tt24_integrated_stress import InstalledArm, schedule
from .tt24_motor_bank import I, STEP, MotorBank, decoder
from .tt24_ready import prepare_step
from .tt24_tool_equilibrium import ToolEquilibrium


STAGES=(0,0,1,1,1,2,2,0,0,1,1,0)
CAP=480


def course_plan(index):
    rng=np.random.default_rng(186275133+index)
    settings=schedule(55841451+index,12)
    by_body={b:iter([s for s in settings if s["body"]==b]) for b in ("stock","alternate")}
    wires=[I,transmission(rng),transmission(rng)@(I+rng.uniform(-.15,.15,(5,5)))]
    result=[]
    for k in STAGES:
        body="alternate" if k else "stock"
        plan=next(by_body[body]).copy()
        plan["changes"]=[(int(0.74*t),g) for t,g in plan["changes"]]
        plan["calibration_rng"]=int(rng.integers(1,2**31-1))
        result.append(plan)
    return result,wires


def trial(arm,launch,plan,wiring,policy,bank,monitor,previous,mode,probe_tape=None,replay=None):
    env=arm.arena
    loc,vel,spin=launch
    obs=arm.reset(plan,loc,replay is None)
    initial=obs["joints"].astype(float)
    inspection=[]
    report=None
    if replay is None:
        policy.begin_release(obs)
        eye=policy.calibration_index
    else:
        eye=None

    def apply_setup(physical):
        physical=np.clip(physical,-1.,1.)
        inspection.append(physical.tolist())
        prepare_step(env,physical,loc)
        return env._observation(False)["joints"]

    if mode=="adaptive" and replay is None:
        check=monitor.pending or (previous is not None and previous!=eye)
        if check:
            requests=[]
            def send(command):
                requests.append(command.copy())
                return apply_setup(wiring@command)
            report=bank.connect(initial,send,plan["calibration_rng"],measured_settling=True)
            report["setup_before_move"]=bool(previous is not None and previous!=eye)
            # Early recognition can finish away from the starting setpoint. Close the requested
            # displacement on this same arm before handing control to the visual policy.
            net=np.sum(requests,axis=0)
            if np.linalg.norm(net)>1e-7:
                for _ in range(4):
                    send(-net/4)
                last=env._observation(False)["joints"].astype(float)
                stable=0
                for _ in range(30):
                    current=send(np.zeros(5))
                    stable=stable+1 if np.max(abs(current-last))<.00045 else 0
                    last=current
                    if stable>=3:
                        break
            monitor.pending=False
        transform=I if bank.selected is None else bank.inverse
        monitor.current=I if bank.selected is None else bank.maps[bank.selected]
    else:
        for u in probe_tape:
            apply_setup(u)
        transform=decoder(wiring) if mode=="known" else I
    obs=env._observation(replay is None)
    if replay is None:
        policy.begin_release(obs)
        monitor.start(obs["joints"])
    audit_state=obs["joints"].astype(float).tolist()
    lead=plan["prep_ticks"]
    changes=dict(plan["changes"])
    events=[]
    game_tape=[]
    for t in range(lead+CAP):
        if t-lead in changes:
            events.append(arm.switch_gravity(t-lead,changes[t-lead]))
        if replay is None:
            desire=policy.act(obs)
            requested=np.clip(transform@desire,-1.,1.)
            physical=np.clip(wiring@requested,-1.,1.)
        else:
            physical=replay[t]
        game_tape.append(np.asarray(physical).tolist())
        if t<lead:
            prepare_step(env,np.asarray(physical),loc)
            if t==lead-1:
                env._ball_velocity[:]=vel
                env._ball_spin[:]=spin
                mujoco.mj_forward(env._model,env._data)
            obs=env._observation(replay is None)
            done=False
        else:
            obs,_,done=env.step(physical,render=replay is None)
        if mode=="adaptive" and replay is None:
            monitor.observe(requested,obs["joints"])
        if done or t==lead+CAP-1:
            break
    return dict(inspection=inspection,report=report,index=eye,after_setup_q=audit_state,
                initial_q=initial.tolist(),actions=game_tape,metrics=env.judge_metrics(),events=events)


def run(index):
    path=ROOT/f"single-arm-course-{index}.json.gz"
    plans,wires=course_plan(index)
    policy=ToolEquilibrium(BUNDLE,.0557)
    policy.mode="workspace_local"
    bank=MotorBank()
    monitor=MotorMemory()
    previous=None
    arms={b:InstalledArm(b) for b in ("stock","alternate")}
    for a in arms.values():
        a.arena.max_steps=CAP
    server=BounceArena(view="none")
    rows=[]
    if path.exists():
        with gzip.open(path,"rt") as f:
            saved=json.load(f)
        rows=saved["rows"]
        bank=MotorBank(np.array(saved["maps"]))
        bank.selected=saved["selected"]
        if bank.selected is not None:
            bank.inverse=decoder(bank.maps[bank.selected])
        monitor.pending=saved["pending"]
        previous=saved["previous"]
    try:
        for d in range(len(rows),len(plans)):
            plan=plans[d]
            server.reset(seed=plan["launcher_seed"],split="held",render=False)
            launch=[v.copy() for v in (server._ball_position,server._ball_velocity,server._ball_spin)]
            arm=arms[plan["body"]]
            modes={}
            for kind in ("adaptive","frozen","known"):
                setup=None if kind=="adaptive" else modes["adaptive"]["inspection"]
                row=trial(arm,launch,plan,wires[STAGES[d]],policy,bank,monitor,previous,kind,setup)
                verify=trial(arm,launch,plan,wires[STAGES[d]],policy,bank,monitor,previous,kind,
                             row["inspection"],row["actions"])
                for key in ("initial_q","after_setup_q","inspection","actions","metrics","events"):
                    if digest_json(row[key])!=digest_json(verify[key]):
                        raise ValueError(f"live course replay mismatch in {key}")
                if kind!="adaptive" and row["after_setup_q"]!=modes["adaptive"]["after_setup_q"]:
                    raise ValueError("comparison started at a different actual joint configuration")
                modes[kind]=row
            previous=modes["adaptive"]["index"]
            rows.append(dict(plan=plan,wire_index=STAGES[d],modes=modes))
            _write(path,dict(rows=rows,maps=[m.tolist() for m in bank.maps],selected=bank.selected,
                             pending=monitor.pending,previous=previous,wires=[m.tolist() for m in wires]))
    finally:
        for a in arms.values():
            a.arena.close()
        server.close()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(run,range(3)))
