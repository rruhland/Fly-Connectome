"""Attempt an opt-in dense stereo correction on genuinely fresh physical offers."""

import argparse
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_camera_precision as dense
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt40_long_low import same_trace
from .tt48_contact_patch import ROOT, ease
from .tt48_contact_patch_report import replay
from .tt48_precise_crop import CroppedStereo
from .tt48_precise_receive import Arc, aperture, camera_pixels
from .tt48_rim_recovery import DIRECT, PLAN as DONORS, STOCK, Rim, alignment, remap

PLAN=ROOT/"precise-live.protocol.json"


class DenseRim(Rim):
    def __init__(self,*args,lens,**kw):
        super().__init__(*args,**kw)
        self.lens=lens
        self.cal=dense.pack(lens.rig)
        self.active=False

    def begin(self):
        super().begin()
        self.sensor=Arc(self.cal)
        self.observed=None
        self.attempts=[]
        self.last_attempt=-99

    def step(self,tick,eye,x,q,ref,base):
        if self.active:
            now=tick*mpc.DT
            self.observed=dense.stock_at(now,eye.latest)
            if aperture(self.observed):
                items=self.lens.capture(self.observed[:3])
                self.sensor.add(now,camera_pixels(items,self.lens.rig,self.observed)[1])
                self.diag["dense_rim_exposures"]+=1
        return super().step(tick,eye,x,q,ref,base)

    def search(self,tick,x,q,ref,base):
        take=(self.active and -.99<x[1]<-.48 and x[4]<-.92 and tick-self.last_attempt>=6)
        mark=len(self.notes)
        self.capture=[] if take else None
        try:
            season.Player.search(self,tick,x,q,ref,base)
            pool=self.capture
        finally:
            self.capture=None
        if not pool or len(self.notes)==mark or self.notes[-1].get("legal_trials",0):
            return
        est=self.sensor.fit(tick*mpc.DT,self.observed,spin_weight=1.,accurate=True)
        if (est is None or est["n"]<8 or est["rms_base_px"]>.15
                or est["pv"][5]>-.38):
            return
        self.last_attempt=tick
        engine=self.engine
        ordinary=engine.latest
        twin=engine.twin
        w=twin.data.qvel[twin._ball_dof+3:twin._ball_dof+6]
        def sync(pv,spin):
            engine.latest=replace(ordinary,time=tick*mpc.DT,position=pv[:3],velocity=pv[3:6])
            assert engine.sync()
            w[:]=spin
            mujoco.mj_forward(twin.model,twin.data)
        pv,spin=est["pv"],est["spin"]
        def key(z):
            _,f=z
            return 8.*abs(f[7]-1.002)+1.4*abs(f[6]+1.55)+max(0.,1.30-f[3])*.14
        heads=[]
        for g,f in sorted(pool,key=key):
            if any(np.max(abs(g-v))<.039 for v,_ in heads):
                continue
            heads.append((g,f))
            if len(heads)==4:
                break
        if not heads:
            return
        note=dict(tick=tick, issued=False, pre_bounce_rms=est["rms_base_px"], fit_pairs=est["n"],
                  corrections=0, nominal=0, robust=0)
        self.attempts.append(note)
        self.diag["dense_rim_fits"]+=1
        try:
            sync(pv,spin)
            # Preserve any head or already-issued lease made legal by the
            # higher-quality state even before trying a learned correction.
            initials=[g for g,_ in heads]
            if self.best is not None and 0<=tick-self.tick<mpc.HORIZON-2:
                initials=[mpc.tail(self.best[tick-self.tick:],mpc.HORIZON)]+initials
            _,existing,_,_,_=DIRECT.trial(self,initials,long=True)
            if np.any(existing):
                return
            tracks=[]
            tags=[]
            time=(1+np.arange(mpc.HORIZON))*mpc.DT
            for move,fb in heads:
                eta=float(np.clip((x[1]-fb[6])/-x[4],.067,.30))
                for earned in self.corrections:
                    shift=remap(earned["delta"],earned["flank"],fb[5])
                    for g in (.72,1.03):
                        tracks.append(move+ease(time/max(.040,.87*eta))[:,None]*(g*shift))
                        tags.append([earned["release"],g])
            note["corrections"]=len(tracks)
            loss,ok,_,y,f=DIRECT.trial(self,tracks,long=True)
            note["nominal"]=int(np.sum(ok))
            choices=np.flatnonzero(ok&(f[:,1]>.5))
            if not len(choices):
                return
            tested=choices[np.argsort(loss[choices])[:4]]
            candidate=[tracks[i] for i in tested]
            angles=alignment(engine,candidate)
            supported=np.zeros(len(tested),dtype=int)
            # A narrow camera-scale stress check guards against new glancing
            # contacts. Values are fixed before any fresh native offer is seen.
            for side in [-1,1]:
                sync(pv+side*np.array([.00035,-.00065,.00038,.005,.010,.011]),spin+side*np.array([8.,0.,0.]))
                _,ok,_,_,_=DIRECT.trial(self,candidate,long=True)
                supported+=ok
            keep=np.flatnonzero((angles>.81)&(supported>=1))
            note["robust"]=int(len(keep))
            if len(keep):
                sel=int(keep[np.argmin(loss[tested[keep]]-.30*supported[keep])])
                i=int(tested[sel])
                eta=float(y[i,1])
                if not .04<eta<.35:
                    return
                self.best=tracks[i]
                self.tick=tick
                self.next=tick+int(np.ceil(eta/mpc.DT))+3
                self.lease=min(mpc.HORIZON,max(mpc.LEASE,self.next-tick))
                note.update(issued=True, alignment=float(angles[sel]), train=tags[i],
                            eta=eta, approved_perturbations=int(supported[sel]))
                self.diag["dense_rim_issued"]+=1
        finally:
            engine.latest=ordinary
            engine.sync()


def schedule(out):
    reference=_read(STOCK)
    # Unique release order is decided by the sealed randomized assessment
    # plan only. Neither opening nor leg outcomes select fresh cards.
    release={}
    for i,r in enumerate(reference["cards"]):
        release.setdefault(r["id"],(i,r))
    rng=np.random.default_rng(480522)
    cards=[dict(c,seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30)))
           for _,c in release.values()]
    if any(c["opening"]!="random" or c["profile"]!="reach" for c in cards):
        raise ValueError("Benchmark opponent envelope changed")
    _write(out,dict(cards=cards, shot_index=[i for i,_ in release.values()], cap=2,
                    source_sha256=_hash(Path(__file__)), calibration_sha256=_hash(ROOT/"precise-held-accuracy.json.gz"),
                    carry_sha256=_hash(season.CARRY), donor_sha256=_hash(DONORS), reference_sha256=_hash(STOCK),
                    decision="Fresh open-loop seed and broad per-leg target stream reserved for each of the eight existing measured assessment IDs. The two physical arms run to first learner return plus next reply, or a natural miss, with frozen carried memory. This is a matched diagnostic, not the 240-alternation benchmark."))


def run(plan,out):
    protocol,stock,donors=_read(plan),_read(STOCK),_read(DONORS)
    if (protocol["carry_sha256"]!=_hash(season.CARRY) or protocol["donor_sha256"]!=_hash(DONORS)
            or protocol["reference_sha256"]!=_hash(STOCK) or protocol["source_sha256"]!=_hash(Path(__file__))
            or protocol["calibration_sha256"]!=_hash(ROOT/"precise-held-accuracy.json.gz")
            or not _read(ROOT/"precise-held-accuracy.json.gz")["complete"]):
        raise ValueError("Changed sealed fresh trial plan")
    result=dict(cards=protocol["cards"],rows=[],complete=False,plan_sha256=_hash(plan))
    if out.exists():
        older=_read(out)
        if older["plan_sha256"]!=_hash(plan):
            raise ValueError("Cannot resume a different fresh trial schedule")
        result=older
    train=_read(season.CARRY)
    train_ids={c["id"] for c in train["cards"][:len(train["rows"])]}
    if any(c["id"] in train_ids or c["id"] in {r["release"] for r in donors["corrections"]} for c in result["cards"]):
        raise ValueError("A precise live ID leaked into learned movement preparation")
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,"local")
    ready=stock["ready"]
    with mpc.centered(ready):
        a=mpc.arena()
        lens=CroppedStereo(a,4)
        camera,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda p,e: Carried(
                p,e,practice=[course.PRACTICE],held={c["id"] for c in result["cards"]})):
            model=DenseRim(mpc.BANK,raw,variant="elastic",budget=stock["budget"],
                          coverage=fresh.shot.Coverage(_read(fresh.RISK)),
                          corrections=donors["corrections"],lens=lens)
        try:
            for i in range(len(result["rows"]),len(result["cards"])):
                c=result["cards"][i]
                shot=stock["rows"][protocol["shot_index"][i]]["shot"]
                arms={}
                for tag in ("control","dense"):
                    course.resume(model,season.CARRY.with_suffix("").with_suffix(".npz"),train)
                    model.active=tag=="dense"
                    r=mpc.play(a,camera,rig,stride,lo,hi,ready,c,shot,raw,model,
                               course.OPTIC,"frozen",protocol["cap"])
                    for leg,target in enumerate(r["placements"]["requested"]):
                        if target["target"]!=list(draw(c,leg,"random",None)[0]):
                            raise ValueError("Fresh stream did not run the unchanged broad opponent")
                    r["amendments"]=model.attempts
                    arms[tag]=r
                control,changed=arms["control"],arms["dense"]
                d=next((k for k,(u,v) in enumerate(zip(control["actions"],changed["actions"])) if u!=v),
                       None if len(control["actions"])==len(changed["actions"]) else min(len(control["actions"]),len(changed["actions"])))
                coded=[r for r in changed["amendments"] if r["issued"]]
                match=None
                if d is not None:
                    if not coded or d<coded[0]["tick"]:
                        raise ValueError("Dense absent before commands diverged")
                    if digest_json([e for e in control["events"] if e["step"]<=d])!=digest_json(
                           [e for e in changed["events"] if e["step"]<=d]):
                        raise ValueError("A different native event arrived before issuance")
                    offers=[r["placements"]["delivered"] for r in (control,changed)]
                    match=bool(offers[0] and offers[1] and all(same_trace(offers[0][0][k],offers[1][0][k])
                             for k in ("far_contact","near_bounce")))
                elif digest_json(control["events"])!=digest_json(changed["events"]):
                    raise ValueError("Camera-only observation perturbed physical outcomes")
                win=match and changed["score"]["learner"]>control["score"]["learner"]
                row=dict(id=c["id"],first_action_diff=d,same_first_offer=match,same_ball_improves=bool(win),arms=arms)
                if win:
                    row["strict_cad"]={tag:replay(ready,c,r) for tag,r in arms.items()}
                result["rows"].append(row)
                result["complete"]=len(result["rows"])==len(result["cards"])
                _write(out,result)
                print(c["id"],control["score"]["learner"],changed["score"]["learner"],len(changed["amendments"]),len(coded),match,flush=True)
        finally:
            model.close();lens.close();camera.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--out",type=Path,default=ROOT/"precise-live.json.gz")
    a=p.parse_args()
    schedule(a.plan) if a.mode=="plan" else run(a.plan,a.out)
