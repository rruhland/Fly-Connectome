"""A received-camera low-ball lift for the SO-101, tested on real issued commands."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import qmc

from . import tt33_joint as joint, tt37_motor_clinic as clinic
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, measured_launches
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt37_belief import ArrivalHistory
from .tt37_joint_deep import DeepField
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt37_visual_inertia import LearnedOptics
from .tt38_motor_value import MotorValue
from .tt38_motorcoach import answerable, setup

ROOT = Path("artifacts")
SCHOOL = ROOT / "tt37/mix-rollin.json.gz"
OPTICS = ROOT / "tt37/rebound.npz"
TRAIN_DEPTH = -1.56
LOW = 1.002
LAND = np.array([0.0, .76, .782])
PROFILES = {
    "scoop": ([-1.57, -1.63, -1.69], [(3.55, .95), (4.08, 1.45)]),
    "early": ([-1.40, -1.47, -1.53], [(3.55, .95), (4.08, 1.45)]),
}
SIZES = np.array([.095,.12,.115,.15,.20,.95,1.15,1.25,1.65,1.5,.026])
PERTURB = np.array([
    [-.002,0,-.004,.01,.02,-.045,-17.],
    [.002,0,.004,-.01,-.02,.045,17.],
])


def fit_under(ik, ball, normal, q, reference, left, stride):
    """Prefer the prescribed impulse plane and accept reachable offsets on its face."""
    radius = .023
    budget = .85 * stride[:5] * (left * 60 + .8)
    edge = .045
    lo = np.maximum(ik.low, q[:5] - budget)
    hi = np.minimum(ik.high, q[:5] + budget)
    if np.any(hi <= lo):
        return None
    cold = np.clip(.72 * q[:5] + .28 * reference[:5], lo + 1e-7, hi - 1e-7)
    def error(motor):
        p, n = ik.pose(motor)
        d = ball - p - radius * n
        face = d @ n
        along = d - face * n
        reach = np.linalg.norm(along)
        return np.r_[2.8 * face, along * (.24 + 1.4 * max(0., reach-edge) / max(.01,reach)),
                     .14 * (n - normal), .017 * (motor - cold)]
    result = least_squares(error, cold, bounds=(lo, hi), max_nfev=36, ftol=1e-4, gtol=1e-5)
    p, n, linear, tilt = ik.pose(result.x, jacobian=True)
    v = ball - p - radius*n
    if abs(v @ n) > .043 or np.linalg.norm(v - (v @ n)*n) > .090:
        return None
    return result.x, n, linear, tilt


def receive(engine, depth):
    """Post-bounce trajectory from the current independent optical estimate."""
    f = engine.latest
    now = engine.tick * engine.twin.control_dt
    if not engine.bounced or f is None or now-f.time > .09:
        return None
    if f.advance(now)[0][1] <= depth or f.advance(now+.46)[0][1] >= depth:
        return None
    time = brentq(lambda p: f.advance(p)[0][1] - depth, now, now+.46)
    p, v = f.advance(time)
    if not (.82 < p[2] < 1.14 and abs(p[0]) < .37 and v[1] < -1.6):
        return None
    return time-now, p, v


def swing(poses, rates, eta, offsets, dt, n=78):
    t = (np.arange(n) + 1) * dt
    tau = np.clip(t[None] - eta[:,None] + (.045+offsets[:,10])[:,None], -.135, .075)
    velocity = rates + offsets[:,5:10]
    q = (poses+offsets[:,:5])[:,None] + tau[:,:,None] * velocity[:,None]
    return np.concatenate((q, np.full((*q.shape[:2],1), clinic.GRIP)), axis=-1)


def consequence(y, f, judge):
    legal = np.isfinite(y[:,0])
    cost = np.full(len(y), 100.)
    # Search motions that actually lift and clear the net, not just motions that touch.
    v = np.nan_to_num(f[:,2:5])
    p = np.nan_to_num(f[:,5:8])
    crossing = -p[:,1] / np.maximum(.8,.90*v[:,1])
    clearance = p[:,2] + v[:,2]*crossing - 4.905*crossing**2
    airborne = np.maximum(.1, (v[:,2]+np.sqrt(np.maximum(0.,v[:,2]**2+19.62*(p[:,2]-.78))))/9.81)
    predicted = p[:,:2] + .91*v[:,:2]*airborne[:,None]
    failed = (
        3.8*np.maximum(0.,.995-clearance) + 1.7*abs(predicted[:,1]-.76)
        + 4.*np.maximum(0., abs(predicted[:,0])-.22)
        + .95*np.maximum(0.,2.7-v[:,1]) + 5.*f[:,8]
    )
    touch = f[:,1]>0
    cost[touch] = 15. + failed[touch]
    cost[~touch] = 24. + 65.*f[~touch,0]
    if legal.any():
        cost[legal] = answerable(y[legal]) + 2.5*np.maximum(0., v[legal,2]-5.5)
        if judge is not None:
            cost[legal] += .36*(judge.cost(y[legal])-5.)
    return cost, legal


def earned_undercuts(doc, engine):
    """Save individual actually issued low-return tapes, never averaged poses."""
    t = engine.twin
    axes = engine.actuators
    stride = t._joint_increments[axes] * engine.predict.micro
    lo, hi = t._joint_low[axes], t._joint_high[axes]
    bank = []
    for card, row in zip(doc["cards"], doc["rows"], strict=True):
        refs = [np.asarray(doc["ready"])]
        for u in row["actions"]:
            refs.append(np.clip(refs[-1] + stride*np.asarray(u), lo, hi))
        refs = np.asarray(refs[1:])
        for stroke in row["placements"]["delivered"]:
            if not stroke["near_landed"]:
                continue
            take = [r for r in row["records"]
                    if stroke["far_contact"]["time"] < r["tick"]/60 < stroke["near_contact"]["time"]]
            if not take or take[-1]["context"][1] > 1.024:
                continue
            choice = take[-1]
            if choice["score"] is None:
                continue
            n = choice["tick"]
            sequence = refs[n:n+22]
            if len(sequence)<15:
                continue
            bank.append(dict(id=card["id"],x=np.asarray(choice["context"]), eta=choice["score"][1],
                             seq=sequence, tag=(card["id"], n)))
    return bank


def earned_candidates(bank, obs, q, eta, seed, *, omit=None, n=78):
    if not bank:
        return [], []
    x = np.r_[obs[1:6],q[:5]]
    scale = np.array([.14,.040,1.2,.80,.90,1.3,1.5,1.2,.80,1.1])
    order = sorted([r for r in bank if r["id"] != omit],
                   key=lambda r: np.linalg.norm((x-r["x"])/scale))[:6]
    raw = qmc.Sobol(6,scramble=True,seed=401091).random_base2(5)*2.-1.
    raw[0] = 0.
    raw[1:5] = 0.
    raw[1:5,0]=[.40,-.40,.95,-.95]
    span = (np.arange(n)+1)/60
    acts, labels = [], []
    for r in order:
        expected = np.clip(r["eta"] + eta-.178, .055, .255)
        dx, dz = (x - r["x"])[:2]
        pan = np.clip(-3.75*dx,-1.05,1.05)
        height = np.clip(3.1*dz,-.22,.22)
        # Moving toward the ball on the transverse axis transports the swing
        # without changing its azimuth. Higher speed inserts a slight phase stretch.
        for z in raw:
            time = (span-expected-.040*z[0])*np.clip(1+.19*z[1],.78,1.26)+r["eta"]
            warp = np.clip(time*60-1,0,len(r["seq"])-1)
            seq = np.array([np.interp(warp,np.arange(len(r["seq"])),r["seq"][:,j]) for j in range(6)]).T
            seq[:,:5] += [pan+.14*z[2], .047*z[3],.38*height+.075*z[4],
                          .71*height+.137*z[5],-.88*pan-.09*z[2]]
            # Some variants start the earned movement earlier and introduce extra
            # upward but nearly tilt-preserving drive at its actual incoming phase.
            pulse = np.clip(span-expected+.047,-.039,.050)
            seq[:,1] += .85*z[3]*pulse
            seq[:,2] += .36*z[3]*pulse
            seq[:,3] -= .91*z[3]*pulse
            acts.append(seq)
            labels.append(r["tag"])
    return acts, labels


class LowLift:
    def __init__(self, engine, practice=None, families=("scoop","early"), earned=None):
        self.ik = BladeIK(engine.twin)
        self.families = families
        self.practice = [] if practice is None else practice
        self.earned = [] if earned is None else earned
        self.omit_release = None
        self.judge = MotorValue(_read(ROOT / "tt37/motor-value.json.gz"))

    def seed(self, engine):
        q = engine.previous
        ref = engine.reference
        data = engine.twin
        stride = data._joint_increments[engine.actuators]*engine.predict.micro
        rows = []
        for family in self.families:
            ds, styles = PROFILES[family]
            for depth in ds:
                arrival = receive(engine, depth)
                if arrival is None or not .040 < arrival[0] < .30:
                    continue
                eta, p, v = arrival
                for pace, brush in styles:
                    dur = (LAND[1]-p[1])/(.88*pace)
                    departing = (LAND-p)/dur
                    departing[1] = pace
                    departing[2] += 4.905*dur
                    n = departing-v
                    n[2] *= .79 # some of the lift is an actively rising face
                    n /= np.linalg.norm(n)
                    fit = fit_under(self.ik, p, n, q, ref, eta, stride)
                    if fit is None:
                        continue
                    pose, face, linear, tilt = fit
                    # The contact rise is resolved on the attained under-ball pose.
                    # The small forward component avoids a backspin chop on a falling ball.
                    system = np.r_[linear, .060*tilt, .021*np.eye(5)]
                    speed = np.linalg.lstsq(system, np.r_[[-.2*v[0],.79,brush],np.zeros(8)],rcond=None)[0]
                    speed = np.clip(speed,-6.,6.)
                    rows.append((family,depth,pace,brush,eta,pose,speed))
        return rows

    def experience_cost(self, name):
        # A weak prior. Responses come only from separate physical practice, not virtual labels.
        r = [bool(x["legal"]) for x in self.practice if x["family"] == name]
        return -.80*(sum(r)+1)/(len(r)+2)

    def plan(self, engine, surf, obs, selected, stock_score, fit, eta):
        note = dict(active=False, low_estimate=float(obs[2]), trials=0)
        status = dict(tape=None,score=None,eta=None,trace=[],note=note)
        if not (.83 < obs[2] < LOW and .054 <= eta < .210 and abs(obs[1]) < .36):
            return status
        if not engine.sync():
            return status
        note["active"] = True
        seeds = self.seed(engine)
        if not seeds:
            note["reason"] = "no reachable under-ball seed"
            return status
        poses = np.asarray([s[-2] for s in seeds])
        rates = np.asarray([s[-1] for s in seeds])
        eta_seed = np.asarray([s[-3] for s in seeds])
        dt = engine.twin.control_dt
        rng = np.random.default_rng([40097,engine.tick])
        raw = 2*qmc.Sobol(11,scramble=True,seed=402091).random_base2(4)-1.
        raw[:6] = 0.
        raw[:6,10] = [-.026,-.008,.007,.021,.007,.014]/SIZES[10]
        raw[4:6,5:10] = .8
        old = clinic.schedules(*fit,eta,np.asarray([selected]),dt,surf.ready,78)[0]
        evaluations = []
        seen = []
        current = np.repeat(np.arange(len(seeds)),len(raw))
        params = np.tile(raw,(len(seeds),1)) * SIZES
        for stage in range(3):
            new = swing(poses[current],rates[current],eta_seed[current],params,dt)
            if stage == 0:
                new = np.r_[old[None],new]
                current = np.r_[[-1],current]
                params = np.r_[np.zeros((1,11)),params]
            f = np.zeros((len(new),9))
            _, y = engine.trajectories(new,clinic.AIM, max(eta+.12,max(eta_seed)+.10),feedback=f)
            loss, legal = consequence(y,f,self.judge)
            for s,i,z,u,b,feedback,estimate in zip(new,current,params,loss,legal,f,y,strict=True):
                evaluations.append((float(u),s,int(i),z.copy(),bool(b),feedback.copy(),estimate.copy()))
            seen.append(dict(trials=len(new),legal=int(legal.sum()),contacts=int(np.sum(f[:,1])),family_search=True))
            useful = [r for r in sorted(evaluations,key=lambda x:x[0]) if r[2]>=0]
            if len([r for r in useful if r[4]]) >= 10:
                break
            top = useful[:min(12,len(useful))]
            choose = rng.integers(len(top),size=58)
            current = np.array([top[i][2] for i in choose])
            params = np.array([top[i][3] for i in choose])
            params += rng.normal(size=params.shape)*SIZES*(.74 if stage==0 else .43)
            params = np.clip(params,-2.5*SIZES,2.5*SIZES)
        if self.earned:
            elite, sources = earned_candidates(self.earned, obs, engine.previous, eta, engine.tick, omit=self.omit_release)
            if elite:
                add = len(seeds)
                seeds.extend([("earned", None, None, None, eta, None, None)] * len(elite))
                f = np.zeros((len(elite),9))
                _, y = engine.trajectories(np.array(elite),clinic.AIM,max(eta+.12,max(eta_seed)+.10),feedback=f)
                loss, legal = consequence(y,f,self.judge)
                for e, g, cost, ok, seen_f, seen_y in zip(range(len(elite)),elite,loss,legal,f,y,strict=True):
                    evaluations.append((float(cost),g,add+e,np.zeros(11),bool(ok),seen_f.copy(),seen_y.copy()))
                seen.append(dict(trials=len(elite),legal=int(legal.sum()),contacts=int(np.sum(f[:,1])),earned_search=True))
                note["earned_tags"] = sorted(set(sources))
                note["earned_legal"] = int(legal.sum())
        good = [r for r in sorted(evaluations,key=lambda x:x[0]) if r[2]>=0 and r[4]]
        base = next(r for r in evaluations if r[2] == -1)
        note.update(seeds=[list(s[:5]) for s in seeds if s[0] != "earned"], trials=sum(x["trials"] for x in seen),
                    private_legal=len(good), stock_contact=bool(base[5][1]), stock_legal=base[4],
                    stock_departure=base[5][2:5].tolist(), stock_nominal=base[0])
        status["trace"] = seen
        if not good:
            note["reason"] = "no legal nominal lift"
            if evaluations:
                r = min(evaluations,key=lambda x:x[0])
                note["best_nonlegal_departure"] = r[5][2:5].tolist()
            return status
        best = []
        for r in good:
            if any(np.max(abs(r[1][:14] - x[1][:14])) < .050 for x in best):
                continue
            best.append(r)
            if len(best)>=10:
                break
        best.insert(0,base)
        goals = np.array([r[1] for r in best])
        t = engine.twin
        p, v = t.ball.copy(), t.velocity.copy()
        angular = t.data.qvel[t._ball_dof+3:t._ball_dof+6]; w = angular.copy()
        scores = [np.array([r[0] for r in best])]
        valid = [np.array([r[4] for r in best])]
        try:
            for z in PERTURB:
                t.ball[:] = p+z[:3]
                t.velocity[:] = v+z[3:6]
                angular[:] = w + [z[6],0,0]
                f = np.zeros((len(goals),9))
                _, yy = engine.trajectories(goals,clinic.AIM,max(eta+.12,max(eta_seed)+.10),feedback=f)
                loss, legal = consequence(yy,f,self.judge)
                scores.append(loss)
                valid.append(legal)
        finally:
            t.ball[:], t.velocity[:], angular[:] = p,v,w
        success = np.sum(valid,axis=0)
        risks = np.mean(scores,axis=0) + .30*np.max(scores,axis=0)
        risks[1:] += [self.experience_cost(seeds[r[2]][0]) for r in best[1:]]
        winner = 1 + int(np.argmin(risks[1:]))
        r = best[winner]
        family, depth, pace, brush, gap = seeds[r[2]][:5]
        note.update(family=family, depth=depth, direction=y_dir(r[5]), outgoing=r[5][2:5].tolist(),
                    beta=[int(success[0]),int(success[winner])],
                    advantage=float(risks[0]-risks[winner]), impact=float(r[6][1]),
                    family_reward_rule=self.experience_cost(family))
        if success[winner] < 2 or risks[0]-risks[winner]<.35 or (success[0]==3 and success[winner]<3):
            note["reason"] = "not better than the frozen incumbent in optical alternatives"
            return status
        note["issued"] = True
        terminal = min(len(r[1]), max(1,round((float(r[6][1])+.09)/dt)))
        return dict(tape=r[1][:terminal],score=r[6].tolist(),eta=float(r[6][1]),trace=seen,note=note)


def y_dir(f):
    return float(np.arctan2(f[4], max(.01,f[3])))


def next_opportunity(row, tick):
    stamps = np.asarray([x["far_contact"]["time"] for x in row["placements"]["delivered"]])
    return int(np.searchsorted(stamps, tick/60, side="right") - 1)


def train_cases(d):
    # Earlier score leading into the given low window is not screened. Choose
    # first four original offers with low optical height that did not return.
    for c,r in zip(d["cards"],d["rows"],strict=True):
        ps = r["placements"]["delivered"]
        if not ps or ps[0]["near_landed"]:
            continue
        touch = ps[0]["near_contact"]
        stop = touch["time"]+.035 if touch else ps[0]["near_bounce"]["time"]+.48
        rr = [x for x in r["records"] if ps[0]["far_contact"]["time"] < x["tick"]/60 < stop]
        if rr and rr[0]["context"][1] < LOW:
            yield c,r,rr[0]


def physical_practice(args):
    d = _read(SCHOOL)
    cases = list(train_cases(d))[:args.limit]
    doc = dict(source=str(SCHOOL),source_sha256=_hash(SCHOOL),source_code_sha256=_hash(Path(__file__)),
               earned=args.earned,profile="reach",protocol="random",
               families=args.families,roster=[dict(card=c,tick=r["tick"]) for c,_,r in cases],
               modes="forced stock prefix, one new actually played receiving attempt",rows=[],training=[])
    model = _read(PUPIL)
    joint.RECEIVE_Y = d["depth"]
    with centered(d["ready"]):
        a = arena()
        engine = OpticalClinic(body=True)
        skill = LowLift(engine, doc["training"],families=args.families,
                        earned=earned_undercuts(d,engine) if args.earned else None)
        stereo,rig,stride,lo,hi = joint.resources(a)
        surf = setup(d["depth"])
        try:
            for c,r,record in cases:
                started = process_time()
                a.launch(r["shot"], card=c); a.partner.reset(seed=c["seed"])
                engine.previous=None; engine.bounced=False
                eye=joint.Vision(rig,a.control_dt)
                eye.eye=LearnedOptics(eye.eye,d["visual"])
                history=ArrivalHistory()
                actor=joint.JointPolicy(model|{"ready":d["ready"]},stride,lo,hi)
                tick=record["tick"]
                old=[]
                for k,u in enumerate(r["actions"][:tick+1]):
                    q=a.joints.copy()
                    obs=eye.update(k,stereo.capture() if k%2==0 else None,q)
                    obs=history.update(k*a.control_dt,obs)
                    engine.observe(k,eye.eye,q,actor.reference.copy())
                    if k==tick:
                        break
                    a.exchange(u,clinic.FAR)
                    actor.reference=np.clip(actor.reference+np.asarray(u)*stride,lo,hi)
                    old.append(u)
                if digest_json(a.events)!=digest_json([e for e in r["events"] if e["step"]<=tick]):
                    raise ValueError("Practice physical prefix could not be reproduced")
                if obs is None or not np.allclose(np.r_[obs[1:6],q[:5]],record["context"],atol=1e-9):
                    raise ValueError("Practice optical prefix could not be reproduced")
                selected=np.asarray(record["action"])
                skill.omit_release = c["id"]
                out=skill.plan(engine,surf,obs,selected,record["score"],clinic.geometry(surf,obs),obs[0])
                if out["tape"] is None:
                    trial=dict(card=c,tick=tick,base=False,legal=None,note=out["note"],cpu_s=process_time()-started)
                else:
                    new=[]
                    after=len(a.events)
                    tape=out["tape"]
                    for k in range(100):
                        goal=tape[min(k,len(tape)-1)] if k<len(tape) else surf.ready
                        u=actor.drive(goal,a.joints.copy(),.8)
                        new.append(u.tolist())
                        _,_,done,cap,_=a.exchange(u,clinic.FAR)
                        if done or cap or audit(a.events,opening_side=1)["learner"]:
                            break
                    scores=audit(a.events,opening_side=1)
                    ok=bool(scores["learner"])
                    label=dict(family=out["note"]["family"], legal=ok, release=c["id"], decision=tick)
                    doc["training"].append(label)
                    trial=dict(card=c,tick=tick,base=False,legal=ok,note=out["note"],events=a.events[after:],
                               result=scores,action_prefix_sha256=digest_json(old),actions=new,cpu_s=process_time()-started)
                doc["rows"].append(trial)
                _write(args.output,doc)
                print(c["id"],trial["legal"],trial["note"],flush=True)
        finally:
            stereo.close();engine.twin.close();a.close()


def held(args):
    roster=_read(args.plan)[args.group][:args.limit]
    practice=_read(args.training)["training"]
    if not any(x["legal"] for x in practice):
        raise ValueError("No physically earned specialist in separate practice")
    train_releases = {c["id"] for c in _read(SCHOOL)["cards"]} if args.earned else {r["release"] for r in practice}
    if {c["id"] for c in roster} & train_releases:
        raise ValueError("Practice and assessment release IDs overlap")
    joint.RECEIVE_Y=TRAIN_DEPTH
    surf=setup(TRAIN_DEPTH)
    shots=measured_launches(args.feeds)
    model=_read(PUPIL)
    checkpoint=_read(ROOT/"tt37/independent-cold-value.json.gz")
    judge=ROOT/"tt37/motor-value.json.gz"
    with np.load(ROOT/"tt37/self-combined.npz") as bank:
        pupil=DeepField(bank,"local")
    params=clinic.generate(35)
    doc=dict(cards=roster,profile="reach",protocol="random",cap=args.cap,training=str(args.training),
             training_sha256=_hash(args.training),rows=[],initial_memory=len(checkpoint["bank"]),
             source_sha256=_hash(Path(__file__)),optic_sha256=_hash(OPTICS))
    with centered(surf.ready):
        a=arena()
        engine=OpticalClinic(body=True); engine.motor_value=MotorValue(_read(judge))
        stereo,rig,stride,lo,hi=joint.resources(a)
        skill=LowLift(engine,practice,families=args.families,
                       earned=earned_undercuts(_read(SCHOOL),engine) if args.earned else None)
        try:
            for c in roster:
                rolls=[]
                for add in (False,True):
                    r=clinic.point(a,stereo,rig,model,surf,stride,lo,hi,c,shots[c["id"]],
                        list(checkpoint["bank"]),params,"optical-fixed",args.cap,.18,3,True,None,None,True,
                        False,None,engine,OPTICS,pupil=pupil,economy=True,impact=skill if add else None)
                    rolls.append(r)
                f,g=rolls
                first=next((k for k,(x,y) in enumerate(zip(f["actions"],g["actions"])) if x!=y),None)
                doc["rows"].append(dict(card=c,baseline=f,special=g,first_change=first))
                _write(args.output,doc)
                print(c["id"],f["score"]["learner"],g["score"]["learner"],first,
                    [x["impact_margin"]["family"] for x in g["records"] if x.get("impact_margin",{}).get("issued")],flush=True)
        finally:
            stereo.close();engine.twin.close();a.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("practice","held"))
    p.add_argument("output",type=Path)
    p.add_argument("--limit",type=int,default=4)
    p.add_argument("--earned",action="store_true")
    p.add_argument("--cap",type=int,default=200)
    p.add_argument("--plan",type=Path,default=ROOT/"tt39/endurance-plan.json")
    p.add_argument("--group",default="assessment")
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--training",type=Path,default=ROOT/"tt40/low-lift-practice.json.gz")
    p.add_argument("--families",nargs="+",choices=tuple(PROFILES),default=tuple(PROFILES))
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    (physical_practice if a.mode=="practice" else held)(a)
