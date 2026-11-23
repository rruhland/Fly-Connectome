"""Use a simulation teacher to propose postures, then test camera-only imitation.

The teacher sees ball state. It makes no joint demonstration eligible until the
real arm simulation lands a legal return starting from its ordinary home pose.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .tt4_arena import BALL_RADIUS, TABLE_HEIGHT
from .tt4_transfer_trial import _write
from .tt19_so101 import BLADE, GRIP_BODY, SO101_HOME, _GRASP_POS
from .tt19_view_full_game import digest_json
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import PublicBody, STEP_SIZE
from .tt25_recorded_exchange import step_ball, table_model
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver

OBSERVE = 3


def poses(arena):
    model, data = arena.model, arena.data
    arena.reset(seed=19, render=False)
    axes, joints = arena._arm_ids[1][2][:5], arena._arm_ids[1][0][:5]
    blade, grip = model.geom(BLADE).id, model.body(GRIP_BODY).id
    local = _GRASP_POS + model.geom_pos[blade]
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[blade])
    face = rot.reshape(3,3)[:,2]
    draws = qmc.Sobol(5, scramble=True, seed=175101).random_base2(15)
    goals = qmc.scale(draws, arena._joint_low[axes], arena._joint_high[axes])
    centers, normals = [], []
    for goal in goals:
        data.qpos[joints] = goal
        mujoco.mj_kinematics(model,data)
        R = data.xmat[grip].reshape(3,3)
        centers.append(data.xpos[grip]+R@local)
        normals.append(R@face)
    centers = np.array(centers)
    return cKDTree(centers), goals, centers, np.array(normals)


def incoming(shot, coeff):
    model = table_model()
    data = mujoco.MjData(model)
    data.qpos[:3] = [shot["pos_x"],shot["pos_y"],TABLE_HEIGHT+shot["pos_z"]]
    data.qvel[:3] = [shot["vel_"+a] for a in "xyz"]
    data.qvel[3:] = [shot["w_vel_"+a] for a in "xyz"]
    body, table = model.body("ball").id, model.geom("table").id
    w = np.zeros(3)
    bounced, last = False, False
    rows = []
    for tick in range(800):
        step_ball(model,data,coeff,w,body)
        touched = any(table in c.geom and c.efc_address>=0 for c in data.contact)
        if touched and not last:
            if bounced or data.qpos[1]>0:
                break
            bounced = True
        if data.qpos[2]<TABLE_HEIGHT-.12:
            break
        if bounced and tick % 2==0:
            rows.append((data.time,data.qpos[:3].copy(),data.qvel[:3].copy()))
        last=touched
    return rows


def suggest(bank, track):
    tree, goals, centers, normals = bank
    best = np.full(len(goals), np.inf)
    for t, p, v in track:
        ids = np.array(tree.query_ball_point(p,.093),dtype=int)
        if not len(ids):
            continue
        n = normals[ids]
        distance = centers[ids]-p
        normal_distance = np.abs(np.sum(n*distance,axis=1))
        tangent = np.sqrt(np.maximum(0,np.sum(distance**2,axis=1)-normal_distance**2))
        # Stationary-face reflection provides a cheap ranking of teacher proposals.
        # The physical simulator decides which candidates become demonstrations.
        out = v - 1.85 * (n@v)[:,None]*n
        air = (out[:,2]+np.sqrt(np.maximum(0,out[:,2]**2+2*9.81*(p[2]-TABLE_HEIGHT-BALL_RADIUS))))/9.81
        landing = p[:2]+.84*out[:,:2]*air[:,None]
        crossing = -p[1]/np.maximum(.01,out[:,1])
        net_z = p[2]+out[:,2]*crossing-.5*9.81*crossing**2
        arrival = np.max(np.abs(goals[ids]-np.asarray(SO101_HOME[:5])),axis=1)/5.2 + OBSERVE/30
        loss = (8*np.abs(normal_distance-.022)+3*tangent+abs(landing[:,1]-.56)+.7*abs(landing[:,0])
                + 4*np.maximum(0,.94-net_z) + 4*np.maximum(0,arrival-t))
        best[ids] = np.minimum(best[ids],loss)
    picked=[]
    for idx in np.argsort(best):
        if not np.isfinite(best[idx]):
            break
        if not picked or min(np.linalg.norm(goals[idx]-goals[j]) for j in picked)>.22:
            picked.append(idx)
            if len(picked)==24:
                break
    return [(goals[i],float(best[i])) for i in picked]


def episode(arena,shot,goal=None,memory=None,collect=False,observer=None):
    arena.launch(shot)
    body, vision = PublicBody(arena),ViewHistory()
    key = None
    actions = []
    for tick in range(arena.max_steps):
        if tick <= OBSERVE and (collect or memory is not None):
            obs = vision.observe(arena.render_cameras(),arena.joints,body.target)
            if tick==OBSERVE:
                key=obs[:14].copy()
                if memory is not None:
                    keys, goals, scales = memory
                    goal=goals[np.argmin(np.linalg.norm((keys-key)/scales,axis=1))]
        cmd=np.zeros(arena.action_size)
        if tick>=OBSERVE and goal is not None:
            cmd[:5]=np.clip((goal-body.target[:5])/STEP_SIZE[:5],-1,1)
        if observer is not None:
            observer(tick,arena)
        _,_,done,cap,_=arena.step(cmd,render=False)
        body.issued(cmd)
        vision.issued(cmd)
        actions.append(cmd.tolist())
        if done or cap or arena.legal_returns:
            break
    return dict(legal=arena.legal_returns, contact=arena.contacts, incoming=arena.rival_returns,
                events=digest_json(arena.events), frames=len(actions)), key


def run(source):
    raw=source.read_bytes()
    if sha256(raw).hexdigest()!=json.loads((OUTPUT/"recorded-secondary-reference.json").read_text())["sha256"]:
        raise ValueError("Wrong DeepMind shot data")
    shots={r["id"]:r for r in json.loads(raw)}
    reach=json.loads((OUTPUT/"recorded-reach.json").read_text())["cases"]
    # This cohort was screened for geometric opportunity only. Blind assessment
    # below uses all unfiltered launches and does not receive the geometric flag.
    train=[r["id"] for r in reach[32:128] if r["near_blade"]][:24]
    held=[r["id"] for r in reach[160:192]]
    coeff=json.loads((OUTPUT/"recorded-flight.json").read_text())["held_scores"]["match_fit"]["coefficients"]
    arena=ReferenceReceiver()
    teachers, keys, poses_learned, tests=[],[],[],[]
    try:
        bank=poses(arena)
        for sid in train:
            shot=shots[sid]
            options=suggest(bank,incoming(shot,coeff))
            trials=[]
            for goal,loss in options:
                r,_=episode(arena,shot,goal)
                trials.append(dict(goal=goal.tolist(), rank_loss=loss, **r))
                if r["legal"]:
                    repeated,key=episode(arena,shot,goal,collect=True)
                    if repeated!=r:
                        raise ValueError("Teaching example did not reproduce")
                    keys.append(key)
                    poses_learned.append(goal)
                    break
            teachers.append(dict(id=sid,trials=trials))
            print("teach",sid,len(trials),bool(trials and trials[-1]["legal"]), flush=True)
        if keys:
            scales=np.array([24.,20.,5.,5.,6.,6.,3.] * 2)
            memory=(np.stack(keys),np.stack(poses_learned),scales)
            np.savez_compressed(OUTPUT/"recorded-pose-imitation.npz",image_keys=memory[0],joint_goals=memory[1],scale=scales)
            for sid in held:
                pair={}
                for name,mem in [("camera",memory),("static",None)]:
                    row,_=episode(arena,shots[sid],memory=mem)
                    again,_=episode(arena,shots[sid],memory=mem)
                    if row!=again:
                        raise ValueError("Qualification changed on replay")
                    pair[name]=row
                tests.append(dict(id=sid,results=pair))
        report=dict(train=train,held=held,demonstrated=len(keys),teachers=teachers,tests=tests)
        _write(OUTPUT/"recorded-pose-imitation.json.gz",report)
        print("held",len(keys),{k:sum(r["results"][k]["legal"] for r in tests) for k in ("camera","static")},flush=True)
    finally:
        arena.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind",type=Path)
    run(p.parse_args().deepmind)
