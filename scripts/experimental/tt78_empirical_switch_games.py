"""Prospective unmodified TT37 points for an independently qualified empirical switch."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt78_empirical_switch as empirical
from . import tt59_online_corrections as runner
from .tt37_opponent_audit import CourtClearance, guarded
from .tt39_carry_joints import Carried
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json

mpc,course,season,school=empirical.mpc,empirical.course,empirical.season,empirical.school
PROTOCOL=empirical.ROOT/"game-protocol.json"
OUT=empirical.ROOT/"games.json.gz"
COUNT=10


class Player(season.Player):
    def __init__(self,*args,generator,weights,threshold,limits,random_stream,**kwargs):
        super().__init__(*args,**kwargs)
        self.generator=generator
        self.weights=weights
        self.threshold=threshold
        self.limits=limits
        self.random_stream=random_stream

    def begin(self):
        super().begin()
        self.examined=False
        self.commit=None
        self.repairs=[]

    def search(self,tick,x,q,ref,base):
        super().search(tick,x,q,ref,base)
        if self.examined or not self.engine.bounced or not empirical.gate(x):
            return
        if not self.notes or self.notes[-1]["tick"]!=tick:
            return
        self.examined=True
        opts=self.generator.suggest(self.engine,self.raw,self.priors.home,x,q,ref,base,self.notes[-1],self.random_stream)
        scores=[float(empirical.basis(v["feature"])@self.weights) for v in opts]
        top=int(np.argmax(scores)) if scores else None
        r=dict(tick=tick,issued=False,scores=scores)
        if top is not None and scores[top]>self.threshold:
            a=opts[top]
            action=school.atlas.reference_path(ref,a["rates"],*self.limits)[1:]
            self.commit=(tick,action)
            self.best,self.tick,self.lease=action,tick,empirical.H
            r.update(issued=True,mode=a["mode"],nominal_legal=a["nominal_legal"],feature=a["feature"])
        self.repairs.append(r)

    def step(self,tick,eye,x,q,reference,base):
        if x is not None and x[4]>.90:
            self.examined=False
            self.commit=None
        if self.commit is not None:
            at,goals=self.commit
            self.engine.observe(tick,eye,q,reference.copy())
            return goals[min(tick-at,len(goals)-1)]
        return super().step(tick,eye,x,q,reference,base)


def freeze(feeds):
    if PROTOCOL.exists():
        return
    result=_read(empirical.REPORT)
    if not result["whole_games_admissible"]:
        return
    old=empirical.design()
    banned=set(old["banned"])
    banned.update(r["id"] for rows in old["assignments"].values() for r in rows)
    carry=_read(season.CARRY)
    banned.update(c["id"] for c in carry["cards"][:len(carry["rows"])])
    for f in (mpc.BANK,course.PRACTICE):
        banned.update(r["card"]["id"] for r in _read(f.with_suffix(".json"))["practice"])
    if empirical.sensitivity.OUT.exists():
        banned.update(r["card"]["id"] for r in _read(empirical.sensitivity.OUT)["rows"])
    standard=_read(course.SOURCE)
    playable={}
    for path in empirical.SOURCES:
        d=_read(path)
        playable.update({c["id"]:c for c,r in zip(d["cards"],d["rows"]) if r["score"]["learner"]>0})
    available={c["id"]:c for stage in ("practice","assessment") for c in standard[stage]
               if c["id"] not in banned and c["opening"]=="random" and c["profile"]=="reach"}
    rng=np.random.default_rng(780319)
    shots=mpc.measured_launches(feeds)
    cards=sorted(rng.permutation(sorted(available)),key=lambda j:j not in playable)
    if len(cards)<COUNT:
        raise ValueError("Not enough declared measured independent games")
    rows=[dict(card=dict(available[i],seed=int(rng.integers(2**30)),reply_seed=int(rng.integers(2**30))),
               shot=shots[i],randomization=int(rng.integers(2**30))) for i in cards[:COUNT]]
    fixed=(empirical.PLAN,empirical.MODEL,empirical.REPORT,season.CARRY,
           season.CARRY.with_suffix("").with_suffix(".npz"),season.fresh.RISK,course.OPTIC,mpc.BANK)
    _write(PROTOCOL,dict(ready=old["ready"],rows=rows,sources={str(p):_hash(p) for p in fixed},
                         rule="Single ordinary continuous game, same initial RGB and encoder policy and measured launch, stock versus frozen paired-outcome switch, TT37 legal accepted actual capacity untouched. Independently drawn seeds different from any fitted fork, no point retries or artificial targets. Plant returns are never consulted by the selector. Raw CAD outcomes enforce witness evidence after play."))
    print("TT78 games roster",[x["card"]["id"] for x in rows],flush=True)


def audit(a,guard,item,tape):
    a.launch(item["shot"],card=item["card"])
    a.partner.reset(seed=item["card"]["seed"])
    guarded(a,guard)
    for u in tape["actions"]:
        if max(abs(np.asarray(u)))>.8+1.e-10:
            raise ValueError("Changed signed command rate")
        a.exchange(u,mpc.FAR_COMMAND)
    if digest_json(a.events)!=digest_json(tape["events"]) or digest_json(a.point_placements())!=digest_json(tape["placements"]):
        raise ValueError("Whole-game command tape failed independent exact replay")
    clearance=a.clearance_result()
    if clearance["samples"]!=round(a.data.time/a.model.opt.timestep):
        raise ValueError("Incomplete CAD checking")
    return clearance


def games():
    if not PROTOCOL.exists():
        return
    p=_read(PROTOCOL)
    if any(_hash(Path(f))!=h for f,h in p["sources"].items()):
        raise ValueError("Frozen game inputs changed")
    m=_read(empirical.MODEL)
    init=dict(protocol=_hash(PROTOCOL),model=_hash(empirical.MODEL),rows=[],complete=False)
    d=_read(OUT) if OUT.exists() else init
    if d["protocol"]!=init["protocol"] or d["model"]!=init["model"]:
        raise ValueError("Resume needs the same fitted choice")
    if d["complete"]:
        return
    home=np.asarray(p["ready"])
    keep={r["card"]["id"] for r in p["rows"]}
    cp=_read(season.CARRY)
    rank=season.fresh.shot.Coverage(_read(season.fresh.RISK))
    generator=empirical.Candidates()
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    with mpc.centered(home),school.enabled():
        live=mpc.arena()
        check=mpc.arena(width=64,height=48)
        cad=mpc.arena(detail="cad",width=64,height=48)
        guard=CourtClearance(cad.model,articulated=True)
        cam,rig,stride,lo,hi=mpc.resources(live)
        try:
            for item in p["rows"][len(d["rows"]):]:
                arms={}
                for tag in ("stock","empirical"):
                    with patch.object(mpc,"Fragments",lambda f,e:Carried(f,e,practice=[course.PRACTICE],held=keep)):
                        kwargs=dict(coverage=rank,variant="elastic",budget=28)
                        actor=season.Player(mpc.BANK,raw,**kwargs) if tag=="stock" else Player(mpc.BANK,raw,
                            generator=generator,weights=np.asarray(m["weights"]),threshold=m["threshold"],limits=(stride,lo,hi),
                            random_stream=np.random.default_rng(item["randomization"]),**kwargs)
                    try:
                        course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),cp)
                        tape=runner.play(live,cam,rig,(stride,lo,hi),home,item["card"],item["shot"],raw,actor,item["randomization"])
                        tape["cad"]=audit(check,guard,item,tape)
                        if tag!="stock":
                            tape["same_offer_changes"]=runner.game.compare(arms["stock"],tape)
                        arms[tag]=tape
                        print(item["card"]["id"],tag,tape["score"]["learner"],tape["complete_point"],
                              tape["cad"]["penetration_samples"],len(tape["repairs"]),flush=True)
                    finally:
                        actor.close()
                d["rows"].append(dict(item,arms=arms));_write(OUT,d)
        finally:
            for a in (live,check,cad):a.close()
            cam.close()
    d["complete"]=True
    _write(OUT,d)


if __name__=="__main__":
    i=argparse.ArgumentParser(description=__doc__)
    i.add_argument("mode",choices=("plan","games"))
    i.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p=i.parse_args()
    freeze(p.feeds) if p.mode=="plan" else games()
