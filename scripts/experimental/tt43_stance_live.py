"""Exercise one distinct ready posture in causal RGB/encoder games on the stock TT37 court."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt41_early_optics_carry import PRACTICE, resume, save
from .tt41_shallow_model import Chart
from .tt43_stance import OUT, ROOT
from .tt43_stance_chart import path_options
from .tt43_stance_native import dynamic_options


class OneHub(mpc.MPC):
    def __init__(self, bank, raw, hub, atlas, memory=(), handoff=False):
        super().__init__(bank, raw, variant="elastic", budget=28)
        self.hub = np.array(hub)
        self.cad = mpc.arena(detail="cad")
        self.chart = Chart(self.cad, self.priors.home, atlas["seeds"][0], atlas=atlas)
        self.earned = deque(memory,maxlen=64)
        self.handoff = handoff

    def begin(self):
        super().begin()
        self.home_started = False
        self.epoch = 0
        self.last_out = -200
        self.cue = None
        self.next_hub = 0
        self.tape = None
        self.tape_start = None
        self.pending = None
        self.rejected = False
        self.provisional = None
        self.deep_supervisor = False

    def close(self):
        self.cad.close()
        super().close()

    def reward(self, archive):
        # Credit only a motor correction whose selected sequence remained
        # committed through the estimated near impact, not a dry-run proposal.
        p = self.pending
        if p is None:
            return
        motion=[(int(s[0]),s[1]) for s in archive if s[1] is not None
                and p["contact_tick"]-15<int(s[0])<p["contact_tick"]+12]
        separated = any(a[1][4]<-.9 and b[1][4]>.9 and b[1][1]<-.45 and b[0]-a[0]<10
                        for a,b in zip(motion,motion[1:]))
        if separated:
            row = p["seed"]
            self.earned.append({k:row[k] for k in ("p","v","face")}
                                | {k:p[k] for k in ("parameters","drive_mode")})
            self.diag["hub_earned"] += 1
        self.pending = None

    def step(self,tick,eye,x,q,reference,base):
        self.engine.observe(tick,eye,q,reference.copy())
        if x is not None and x[4] > .9:
            self.best = None
            if self.cue is not None or tick-self.last_out > 26:
                self.epoch += 1
                self.cue = None
                self.tape = None
                self.provisional = None
                self.deep_supervisor = False
                self.rejected = not self.chart.guard.route(q[:5], self.hub[:5], steps=10)
                self.home_started = not self.rejected
                self.notes.append(dict(tick=tick,mode="hub_route",available=self.home_started,
                                       initial_travel_rad=float(np.max(abs(self.hub-q)))))
                self.diag["hub_preloads"] += int(self.home_started)
            self.last_out = tick
            if self.home_started:
                return self.hub
        if x is None and self.home_started and self.cue is None and tick-self.last_out < 45:
            return self.hub
        if self.tape is not None and 0 <= tick-self.tape_start < len(self.tape):
            ordinary = self.pending is not None and self.pending["seed"]["p"][1]<-1.38
            if self.handoff and ordinary and self.engine.bounced and x is not None and -.49>x[1]>-1.53:
                before=len(self.notes)
                with patch.object(self.engine,"observe",lambda *a:None):
                    follow=super().step(tick,eye,x,q,reference,base)
                if len(self.notes)>before and self.notes[-1].get("chosen") and self.notes[-1].get("robust",0):
                    self.diag["hub_deep_handoffs"] += 1
                    self.deep_supervisor = True
                    self.notes[-1]["hub_handoff"] = True
                    self.tape=None
                    self.pending=None
                    return follow
            self.diag["hub_sequence_ticks"] += 1
            return self.tape[tick-self.tape_start]
        if x is not None and x[4] < -.90 and self.home_started and not self.rejected and not self.deep_supervisor:
            if self.cue is None:
                self.cue = tick
                self.notes.append(dict(tick=tick,mode="hub_first_inbound",y=float(x[1]),
                                       achieved_rad=float(np.max(abs(self.hub-q)))))
            if -.98 < x[1] < .70 and tick >= self.next_hub and self.engine.sync():
                self.next_hub = tick+6
                twin=self.engine.twin
                if twin.launches or twin.card is not None:
                    raise RuntimeError("Need an isolated RGB initialized predictor")
                path=twin.partner.flight.path(twin.ball,twin.velocity,twin.spin,1,
                                            bounced=self.engine.bounced,horizon=.63)
                poses=path_options(self.chart,path,quick=True)
                poses=[r for r in poses if .055<r["eta"]<.62
                       and np.max(abs(q[:5]-r["q"]))/4.16<r["eta"]-.017
                       and self.chart.guard.route(q[:5],np.asarray(r["q"]),steps=8)]
                pick, trials=dynamic_options(twin,self.engine.predict,self.chart.ik,poses,None,40,2,
                                            [433097,tick,self.epoch],memories=self.earned)
                self.diag["hub_trials"] += sum(t["tried"] for t in trials)
                note=dict(tick=tick,mode="hub_optical_search",y=float(x[1]),bounced=bool(self.engine.bounced),
                          poses=len(poses),trials=trials,issued=pick is not None)
                self.notes.append(note)
                if pick is not None:
                    self.tape=np.array(pick["goals"])
                    self.tape_start=tick
                    self.pending={**pick,"contact_tick":tick+round(pick["quality"][1]/twin.control_dt)}
                    note["contact_depth"] = pick["seed"]["p"][1]
                    note["contact_face"] = pick["seed"]["face"]
                    self.diag["hub_commits"] += 1
                    return self.tape[0]
                if poses and x[1]>-.61:
                    # A provisional geometrically clear roll-in is revoked at
                    # the next optical recheck. No landing is presumed.
                    p=max(poses,key=lambda r:r["eta"]-np.max(abs(q[:5]-r["q"]))/4.16)
                    self.provisional=np.r_[p["q"], self.hub[-1]]
                    self.diag["hub_provisional_ticks"] += 1
                    return self.provisional
            elif self.provisional is not None and tick<self.next_hub and x[1]>-.61:
                self.diag["hub_provisional_ticks"] += 1
                return self.provisional
        with patch.object(self.engine,"observe",lambda *a:None):
            return super().step(tick,eye,x,q,reference,base)


def load_memories(file,card_ids):
    if file is None:
        return []
    data=_read(file)
    if not data.get("complete") or {r["id"] for r in data["rows"]}&card_ids:
        raise ValueError("Teacher must be complete and release-disjoint")
    memory=[]
    for row in data["rows"]:
        result=row.get("physical")
        if not result or not (result["answered"] and row["clearance"]["penetration_samples"]==0
                              and row["independent_event_and_placement_replay"]):
            continue
        p=result["chosen"]
        if "drive_mode" not in p or "v" not in p["seed"]:
            continue
        memory.append({k:p["seed"][k] for k in ("p","v","face")}
                      | {k:p[k] for k in ("parameters","drive_mode")})
    return memory


def games(output,chart,teacher,carry,feeds,deck,start,limit,cap,mode,handoff):
    base=_read(chart)
    if not base.get("complete"):
        raise ValueError("Incomplete geometric calibration")
    atlas_file=ROOT/"tt41/shallow-atlas.json.gz"
    atlas=_read(atlas_file)
    home=np.array(_read(mpc.BANK.with_suffix(".json"))["ready"])
    hub=np.r_[base["routes"]["either_far"][1]["q"],home[-1]]
    cards=_read(course.PLAN)[deck][start:start+limit]
    known={c["id"] for c in cards}
    memory=load_memories(teacher,known)
    if any(c["profile"]!="reach" or c["opening"]!="random" for c in cards):
        raise ValueError("Use only the unchanged continuous random TT37 opponent")
    feeds_state=mpc.measured_launches(feeds)
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    d=dict(schema="tt43-single-hub-optical-1",mode=mode,deck=deck,cap=cap,
           cards=cards,handoff=handoff,ready=home.tolist(),hub=hub.tolist(),rows=[],chart_sha256=_hash(chart),
           teacher_sha256=None if teacher is None else _hash(teacher),initial_hub_examples=len(memory),
           carry_sha256=_hash(carry),optics_sha256=_hash(course.OPTIC),plan_sha256=_hash(course.PLAN),
           inputs="Optical ball estimates, own encoders and issued references. No opponent goals or live solver.")
    with mpc.centered(home):
        a=mpc.arena()
        camera,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda file,engine:Carried(file,engine,practice=[PRACTICE],held=known)):
            actor=OneHub(mpc.BANK,raw,hub,atlas,memory,handoff=handoff)
        source=_read(carry)
        if {c["id"] for c in source["cards"][:len(source["rows"])]}&known:
            raise ValueError("Practice checkpoint overlaps test releases")
        resume(actor,carry.with_suffix("").with_suffix(".npz"),source)
        keep=actor.priors.keep

        def credit(archive,leg):
            actor.reward(archive)
            return keep(archive,leg)

        try:
            with patch.object(actor.priors,"keep",credit):
                for card in cards:
                    r=mpc.play(a,camera,rig,stride,lo,hi,home,card,feeds_state[card["id"]],raw,actor,course.OPTIC,mode,cap)
                    for leg, entry in enumerate(r["placements"]["requested"]):
                        if list(draw(card,leg,"random",None)[0]) != entry["target"]:
                            raise ValueError("Far target draw changed")
                    d["rows"].append(r)
                    d["hub_memory"]=list(actor.earned)
                    if mode=="live":
                        d["standard_memory_sha256"]=save(actor,output.with_suffix("").with_suffix(".npz"))
                    _write(output,d)
                    print(card["id"],r["score"]["learner"],r["score"]["opponent"],r["planner"],flush=True)
        finally:
            actor.close();camera.close();a.close()
    d["complete"]=True
    _write(output,d)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUT/"held.json.gz")
    p.add_argument("--chart",type=Path,default=OUT/"chart.json.gz")
    p.add_argument("--teacher",type=Path)
    p.add_argument("--carry",type=Path,default=course.ROOT/"learning-12.json.gz")
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--deck",choices=("practice","assessment"),default="assessment")
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--limit",type=int,default=2)
    p.add_argument("--handoff",action="store_true")
    p.add_argument("--cap",type=int,default=240)
    p.add_argument("--mode",choices=("frozen","live"),default="frozen")
    a=p.parse_args()
    games(a.output,a.chart,a.teacher,a.carry,a.feeds,a.deck,a.start,a.limit,a.cap,a.mode,a.handoff)
