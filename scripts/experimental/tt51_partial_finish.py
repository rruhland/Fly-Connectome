"""Learn direct finishes from a small reversible preparation on archived native offers."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt50_conditional_prep as prior
from .tt37_opponent_audit import CourtClearance, guarded
from .tt39_opportunity_map import projected_height
from .tt40_low_feasibility import Probe, trial
from .tt49_terminal_blind import Observer

mpc = prior.mpc
root = Path("artifacts/tt51_partial_finish")
archive = Path("artifacts/tt48_transfer/live-plain.json.gz")
plan_path = root / "plan.json"
school_path = root / "school.json.gz"
held_path = root / "matched.json.gz"
training = ((1, 1), (2, 1), (5, 1), (7, 1), (4, 2), (20, 2))
held = ((3, 1), (6, 1), (18, 1), (24, 3), (30, 1), (31, 2))
body_envelope = np.array([0., 0., .48, .105, .22, 0.])
future_scale = np.array([.15, .075, .55, .85, .85])
body_scale = np.array([.40, .30, .47, .46, .55])


def freeze(path):
    old = prior._read(archive)
    rows = []
    for role, group in (("school", training), ("held", held)):
        for i, k in group:
            t = old["rows"][i]
            c = old["cards"][i]
            recv = t["placements"]["delivered"][k]
            req = t["placements"]["requested"][k]
            anchor = int(np.floor(req["time_s"] / mpc.DT + 1e-7))
            depart = next(e for e in t["events"] if e.get("collider") == "learner_racket"
                          and abs(e["time"] - req["time_s"]) < .030 and e["kind"] == "separation")
            rows.append(dict(role=role, id=c["id"], index=i, leg=k, anchor=anchor,
                             keep_through=depart["step"] + 1, height=projected_height(recv["receive_line"])))
    if len({c["id"] for c in rows}) != len(rows):
        raise ValueError("Motor train and held releases overlap")
    prior._write(path, dict(schema="tt51-partial-body-plan-1", source_sha256=prior._hash(archive), rows=rows,
                            role="Exact received replies across 12 disjoint releases. Held outcomes never select a motor waveform."))


class Readiness:
    """Keep the last played movement and estimate the current partial-body state."""

    def __init__(self, predictor=None):
        self.predictor = predictor
        self.reset()

    def reset(self):
        self.history = deque(maxlen=12)
        self.depart_q = None
        self.start = None
        self.last_positive = None
        self.last_inbound = None
        self.seen_inbound = False
        self.attempt = None
        self.risk = None
        self.last_ready = 0.
        if self.predictor is not None:
            self.predictor.reset()

    def observe(self, tick, x, q, ref, frames=None):
        self.history.append((tick, np.array(q), np.array(ref)))
        if self.predictor is not None:
            m = self.predictor.update(frames, x, tick)
            if m is not None:
                self.risk = float(np.clip(self.predictor.future[-1], 0., 1.))
        if x is not None and x[4] < -.9:
            self.seen_inbound = True
            if self.last_positive is not None and tick - self.last_positive < 37 and self.last_inbound is None:
                self.last_inbound = tick
        if x is not None and x[4] > .9:
            if self.last_inbound is not None or (self.last_positive is not None and tick-self.last_positive > 15):
                self.start = None
                self.last_inbound = None
                self.risk = None
                self.attempt = None
            self.last_positive = tick
            if self.seen_inbound and -.99 < x[1] < .38 and self.start is None:
                self.start = tick
                self.depart_q = np.asarray(q).copy()
        return self.last_inbound

    def feature_body(self):
        _, q, ref = self.history[-1]
        t0, q0, _ = self.history[max(0, len(self.history)-5)]
        dt = max(1, self.history[-1][0]-t0) * mpc.DT
        return q[:5], (q[:5]-q0[:5]) / dt, ref[:5]-q[:5]

    def correction(self, tick, x, current):
        self.last_ready = 0.
        if self.start is None or tick-self.start > 90 or x is None:
            return np.zeros(6)
        build = np.clip((tick-self.start+1)/22., 0., 1.)
        fade = 1.
        if self.last_inbound is not None:
            fade = min(np.clip((x[1]+.34)/.69, 0., 1.), np.clip((22-(tick-self.last_inbound))/16., 0., 1.))
        elif x[4] < .9:
            fade = .75
        # The previous playback may already have opened this degree. Do not
        # add the whole nominal change a second time on the following stroke.
        clear = .72 + .28*np.clip((-.80-self.depart_q[2])/.80, 0., 1.)
        scale = (1. if self.risk is None else .65 + .35*np.clip(self.risk/.34, 0., 1.))
        self.last_ready = float(np.clip(build*fade*clear*scale, 0., 1.))
        d = self.last_ready * body_envelope
        # Negative readiness nudge cannot silently drive an already lagging axis.
        q, speed, lag = self.feature_body()
        taper = np.clip(1.1 - .32 * abs(lag)/.25 - .12*abs(speed)/4.16, .5, 1.)
        return d*np.r_[taper, 1.]


def axis_references(arr, center):
    seq = np.asarray([z[19:25] for z in arr])
    p = np.asarray([z[13:19] for z in arr])
    return np.asarray([z[0]-center for z in arr]), p + seq*(4.16/.8/60.)


def receive_labels(packet):
    # Paid-for teacher motions receive their flight and body labels only from
    # the RGB and encoder streams that resulted as they were physically played.
    o = packet["offer"]
    hit = o["near_contact"]
    if not o.get("near_landed") or hit is None:
        return None
    rows = np.array(packet["sensed"])
    tick = int(round(hit["time"] / mpc.DT))
    cues = rows[(rows[:, 0] > tick-33) & (rows[:, 0] < tick-7) &
                (rows[:, 2] > -.97) & (rows[:, 2] < -.28) & (rows[:, 5] < -.9)]
    bounce = o["near_bounce"]["time"] / mpc.DT
    cues = cues[cues[:, 0] >= bounce + 2.]
    if len(cues) < 2:
        return None
    ref_rows = rows[(rows[:, 0] >= tick-30) & (rows[:, 0] <= tick+15)]
    if len(ref_rows) < 7:
        return None
    phases, refs = axis_references(ref_rows, tick)
    archive_body = []
    for k,v in enumerate(cues):
        t, *x = v[:7]
        old = rows[np.max(np.where(rows[:, 0] <= t-3))]
        q = v[7:12];dq = (q-old[7:12])/((t-old[0])*mpc.DT)
        lag = v[13:18]-q
        eta = (-1.10-x[1]) / x[4]
        state = [x[0]+eta*x[3],x[2]+eta*x[5]-4.905*eta**2,x[3],x[4],x[5]-9.81*eta]
        archive_body.append(dict(phase=int(t-tick),future=state,q=q.tolist(),dq=dq.tolist(),lag=lag.tolist()))
    return dict(samples=archive_body,phases=phases.tolist(),refs=refs.tolist(),contact_y=hit["location"][1])


class Finish:
    def __init__(self, doc=None):
        self.moves = []
        for r in [] if doc is None else doc["rows"]:
            for label in ("partial", "coach"):
                if r.get(label, {}).get("skill") is not None:
                    s = r[label]["skill"]
                    self.moves.append((r["id"], s))

    def proposals(self, row, path, context):
        q,dq,lag = context
        near = path[abs(path[:, 2]+1.10) < .042]
        if not len(near):
            return [], []
        v = near[np.argmin(abs(near[:, 2]+1.10))]
        p = np.r_[v[1],v[3:7]]
        shortlist = []
        for ident, s in self.moves:
            contact = path[abs(path[:, 2]-s["contact_y"]) < .043]
            if not len(contact):
                continue
            r = contact[np.argmin(abs(contact[:, 2]-s["contact_y"]))]
            dt = r[0] / mpc.DT
            if not 4. <= dt < 23.:
                continue
            best = None
            for b in s["samples"]:
                body = np.mean(((q-b["q"])/body_scale)**2)
                motion = np.mean(((dq-b["dq"])/2.6)**2)+np.mean(((lag-b["lag"])/.20)**2)
                scored = np.mean(((p-b["future"])/future_scale)**2) + .62*body + .17*motion + ((dt+b["phase"])/8)**2
                if best is None or scored < best[0]:
                    best = (scored,b,body)
            if best is not None and best[0] < 2.55 and best[2]< 2.2:
                shortlist.append((best[0], ident, s, dt, best[1]))
        out, note = [], []
        for score, ident, s, dt, b in sorted(shortlist, key=lambda c:c[0])[:3]:
            lag_correct = np.clip(.24*(q-b["q"]), -.075, .075)
            for off in (-1.,0.,1.):
                phase = np.arange(mpc.HORIZON)+1-dt+off
                g = np.column_stack([np.interp(phase, s["phases"], arr) for arr in np.array(s["refs"]).T])
                g[:,:5] += lag_correct * np.maximum(0.,1.-np.arange(mpc.HORIZON)[:,None]/12.)
                out.append(g);note.append(ident)
        return out, note


class Player(prior.season.Player):
    def __init__(self,*a, mode, finish, checker=None, predictor=None, **kw):
        super().__init__(*a, **kw)
        self.mode = mode
        self.finish = finish
        self.ready = Readiness(predictor)
        self.checker = checker

    def begin(self):
        super().begin()
        self.ready.reset()
        self.pick = []
        self.last_try = -999
        self.learned_branch = False
        self.native_readiness_ticks = 0

    def observe_only(self,tick,frames,eye,x,q,ref):
        self.ready.observe(tick,x,q,ref,frames)
        self.engine.observe(tick,eye,q,ref)

    def choose(self,tick,frames,eye,x,q,ref,base):
        self.ready.observe(tick,x,q,ref,frames)
        if x is not None and x[4]>.9:
            self.learned_branch=False
        plain = super().step(tick,eye,x,q,ref,base)
        if self.mode == "stock":
            return plain
        correction = self.ready.correction(tick,x,ref)
        if np.any(correction):
            want = plain + correction
            if self.checker is not None and x[4]>.9:
                lo,hi=self.checker.ik.low,self.checker.ik.high
                # Only check the reachable next commanded increment.
                command=np.clip(ref[:5]+np.clip(want[:5]-ref[:5],-4.16/60,4.16/60),lo,hi)
                if not self.checker.route(q[:5],command,steps=3):
                    return plain
            self.native_readiness_ticks += 1
            return want
        live_lease=self.best is not None and 0<=tick-self.tick<self.lease
        stock_confirmed=live_lease and self.notes and self.notes[-1]["tick"]==self.tick and self.notes[-1].get("robust",0)>0
        if (self.mode != "partial-finish" or self.ready.last_inbound is None or x is None or
                x[4]>=-.9 or not -.97<x[1]<-.31 or tick-self.last_try<3 or
                stock_confirmed or (live_lease and self.learned_branch)):
            return plain
        self.last_try=tick
        if not self.engine.sync():
            return plain
        a=self.engine.twin
        path=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=self.engine.bounced,horizon=.60)
        options,faces=self.finish.proposals(x,path,self.ready.feature_body())
        if not len(options):
            return plain
        a0,_,tr,_,_=self.trial(options,traces=True)
        for k in np.argsort(a0)[:2]:
            pos=self.transport(tr[k],face=True)
            if pos is not None:
                options.append(options[k]+.62*pos);faces.append(faces[k])
        loss,legal,_,scores,_=self.trial(options,long=True)
        keep=np.flatnonzero(legal & (scores[:,5]>.018))
        keep=keep[np.argsort(loss[keep])[:3]]
        if not len(keep):
            self.pick.append(dict(tick=tick,trials=len(options),model_legal=0))
            return plain
        ix,support,_=self.robust([options[k] for k in keep],keep,loss[keep])
        self.pick.append(dict(tick=tick,trials=len(options),model_legal=len(keep),model_support=support,teacher=faces[ix]))
        if support<1:
            return plain
        self.best=np.array(options[ix]);self.tick=tick
        self.lease=min(len(self.best),max(5,round((scores[ix,1]+.08)/mpc.DT)))
        self.next=max(self.next,tick+self.lease)
        self.learned_branch=True
        return self.best[0]


def episode(a, res, actor, eye, item, native, *, forced=None, full=False):
    camera,_,stride,lo,hi=res
    start,k=item["anchor"],item["leg"]
    actor.ready.seen_inbound=True
    axes=a._arm_ids[1][2]
    seen=[]; tape=[]; prep=[]; first_in=None
    for tick in range(start,start+190):
        q=a.joints.copy(); ref=a.data.ctrl[axes].copy()
        frames=camera.capture() if tick%2==0 else None
        x=mpc.observed(eye,tick,frames,q)
        if forced is not None or tick<=item["keep_through"]:
            actor.observe_only(tick,frames,eye.eye,x,q,ref)
            u=(native["actions"][tick] if forced is None else forced[tick])
        else:
            home=actor.priors.home
            base=home
            if x is not None and x[4]<-.9:
                wish,w=actor.raw.get(x,q)
                if wish is not None:base=home+w*(wish-home)
            goal=actor.choose(tick,frames,eye.eye,x,q,ref,base)
            u=np.clip((np.clip(goal,lo,hi)-ref)/stride,-.8,.8).tolist()
        if x is not None:seen.append([tick,*x,*q,*ref,*u])
        if first_in is None and actor.ready.last_inbound is not None:first_in=actor.ready.last_inbound
        if actor.ready.last_ready and x is not None:
            prep.append([tick,actor.ready.last_ready,*q,*ref])
        tape.append(u)
        _,_,ended,capped,_=a.exchange(u,mpc.FAR_COMMAND)
        if ended or capped or (a.rival_returns>=k+2 if full else a.legal_returns>=k+2):
            break
        if forced is not None and tick+1>=len(forced):break
    p=a.point_placements()
    source=native["placements"]["delivered"][k]
    o=p["delivered"][k] if len(p["delivered"])>k else None
    if o is None or not all(prior.same_trace(o[z],source[z]) for z in ("far_contact","near_bounce")):
        raise ValueError("Reversible prep did not preserve the exact native court offer")
    if p["requested"][k]["target"]!=native["placements"]["requested"][k]["target"]:
        raise ValueError("TT37 changed the original target stream")
    return dict(offer=o,legal=bool(o["near_landed"]),answered=a.rival_returns>=k+2,
                same_offer=True,sensed=seen,tape=tape,prep=prep,selected=actor.pick,
                readiness_ticks=actor.native_readiness_ticks,first_rgb_in=first_in,
                event_sha256=prior.digest_json(a.events),clearance=a.clearance_result(),
                reason=next((e.get("reason") for e in a.events[::-1] if e["kind"]=="point"),None))


def matched(phase, output, n, predictor_path=None, *, coach=True):
    reg=prior._read(plan_path); src=prior._read(archive)
    if reg["source_sha256"]!=prior._hash(archive):raise ValueError("Archived physical states changed")
    rows=[r for r in reg["rows"] if r["role"]==phase][:n]
    skill=None if phase=="school" else prior._read(school_path)
    if skill is not None and ({r["id"] for r in rows}&{r["id"] for r in skill["rows"]} or not skill["complete"]):
        raise ValueError("A held release entered the motor school")
    modes=("stock","partial") if phase=="school" else ("stock","partial","partial-finish")
    optional=None
    if predictor_path is not None:
        from .tt51_flight_risk import RiskForecast
        optional=RiskForecast(predictor_path)
    meta=dict(schema="tt51-partial-finish-matched-1",phase=phase,plan_sha256=prior._hash(plan_path),
              actor_sha256=prior._hash(Path(__file__)),modes=modes,
              school_sha256=None if skill is None else prior._hash(school_path),
              prediction_sha256=None if predictor_path is None else prior._hash(predictor_path),
              technique="Bounded reversible elbow/wrist prep, then observation-conditioned control.")
    doc=dict(meta,rows=[],complete=False)
    if output.exists():
        doc=prior._read(output)
        if any(doc[key]!=v for key,v in {**meta,"modes":list(modes)}.items()):
            raise ValueError("Attempting to resume unlike matched course; use a new output")
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    with mpc.centered(src["ready"]):
        a=mpc.arena();cad=mpc.arena(detail="cad")
        res=mpc.resources(a); camera,rig,*_=res
        axes=a._arm_ids[1][2]
        collision=CourtClearance(cad.model,articulated=True)
        checker=prior.Reach(cad,src["ready"],src["ready"])
        with patch.object(mpc,"Fragments",lambda f,e:prior.Carried(f,e,practice=[prior.course.PRACTICE],held={r["id"] for r in rows})):
            actor=Player(mpc.BANK,raw,mode="stock",finish=Finish(skill),checker=checker,predictor=optional,
                         coverage=prior.fresh.shot.Coverage(prior._read(prior.fresh.RISK)),budget=src["budget"],variant=src["variant"])
        try:
            base=prior._read(prior.ARCHIVE)
            prev={**base,"rows":[None]*len(prior._read(prior.season.CARRY)["rows"])+base["rows"]}
            prior.course.resume(actor,prior.BANK,prev)
            probe=Probe(a) if phase=="school" and coach else None
            for item in rows[len(doc["rows"]):]:
                i=item["index"];card=src["cards"][i];native=src["rows"][i];k=item["leg"]
                history=prior.prefix(a,card,native,item["anchor"],camera,axes,window=36)
                snap=prior.State(a); warm=Observer(a,rig,actor,history)
                q={**item,"base_memory":actor.priors.digest(),"original_prefix_sha256":prior.digest_json(a.events)}
                for m in modes:
                    snap.restore(a); guarded(a,collision)
                    eye=warm.restore(actor);actor.mode=m
                    result=episode(a,res,actor,eye,item,native)
                    if phase=="school" and m=="partial":
                        result["skill"]=None if result["clearance"]["penetration_samples"] else receive_labels(result)
                    q[m]=result
                part=q["partial"]
                if probe is not None and not part["legal"] and part["first_rgb_in"] is not None:
                    cue=part["first_rgb_in"]
                    prefix=part["tape"][:cue-item["anchor"]]
                    def reset(steps):
                        snap.restore(a)
                        a.sample_clearance(False)
                        for c in steps:a.exchange(c,mpc.FAR_COMMAND)
                    reset(prefix)
                    if a.velocity[1]<-.9:
                        a.clearance_margin=.0006
                        try:
                            options,diag=probe.search([515921,item["id"],k],48,2)
                        finally:
                            a.clearance_margin=0.
                        q["private_teacher_screen"]=diag
                        for option in options[:2]:
                            reset(prefix)
                            record=trial(a,dict(native,actions=native["actions"][:item["anchor"]]+part["tape"]),card,option,cue,k,already=True)
                            if not record["answered"]:continue
                            snap.restore(a); guarded(a,collision)
                            a.clearance_margin=0.
                            eye=warm.restore(actor); actor.mode="stock"
                            r=episode(a,res,actor,eye,item,native,forced=record["actions"],full=True)
                            if r["answered"] and not r["clearance"]["penetration_samples"]:
                                r["skill"]=receive_labels(r)
                                if r["skill"] is not None:
                                    q["coach"]=r;break
                doc["rows"].append(q);prior._write(output,doc)
                print(item["id"],item["height"],[(m,q[m]["legal"],q[m]["readiness_ticks"],len(q[m]["selected"]))
                      for m in modes],"earned",[(m,bool(q.get(m,{}).get("skill"))) for m in ("partial","coach")],flush=True)
        finally:
            actor.close();camera.close();a.close();cad.close()
    doc["complete"]=True;prior._write(output,doc)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("plan","school","held"))
    p.add_argument("--output",type=Path)
    p.add_argument("--predictor",type=Path)
    p.add_argument("--no-coach",action="store_true")
    p.add_argument("--limit",type=int,default=6)
    a=p.parse_args();root.mkdir(parents=True,exist_ok=True)
    if a.phase=="plan":freeze(a.output or plan_path)
    else:matched(a.phase,a.output or (school_path if a.phase=="school" else held_path),a.limit,a.predictor,coach=not a.no_coach)
