"""Continue one reward-trained controller across more varied physical source arms."""
from dataclasses import dataclass, asdict
import gzip
import json
import numpy as np
import mujoco

from .tt24_bounce import BounceArena
from .tt24_donor_trial import Design, DonorBounce, Native, World, ROOT, _paired, _summary
from .tt24_donor_assess import CHECKPOINT, ShiftedBounce
from .tt24_donor_adapt import historical_basis

@dataclass(frozen=True)
class Family:
    design: Design
    link: float
    gravity: float
    drag: float
    ball_mass: float

# Link multiplier is relative to the already-extended donor geometry.
TRAIN=(
    Family(Design((.10,-.08),1.02,.041,1.,.96),1.,1.,1.,1.),
    Family(Design((-.10,.08),.90,.060,.85,.91),.92,.76,1.45,1.12),
    Family(Design((.15,.05),1.20,.027,1.15,1.04),1.06,.92,.72,.88),
    Family(Design((-.04,-.15),1.10,.072,1.02,.84),.96,1.14,1.20,1.28),
    Family(Design((.04,.09),.84,.034,.95,1.10),1.11,1.23,.90,.79))
NOVEL=Family(Design((-.08,.16),1.08,.048,.91,.98),.943,1.08,1.31,.94)
OUT=ROOT/"broad-donors.npz"


def stretch(position,ratio):
    xyz=np.fromstring(position,sep=" ")
    xyz[0]*=ratio
    return " ".join(str(v) for v in xyz)


class FamilyArena(DonorBounce):
    def __init__(self,spec,view="none"):
        self.spec=spec
        super().__init__(spec.design,view=view,width=192,height=144)
        self._model.opt.gravity[2]*=spec.gravity
        self._model.body_mass[self._ball_body]*=spec.ball_mass
        self._model.body_inertia[self._ball_body]*=spec.ball_mass
        mujoco.mj_setConst(self._model,self._data)

    def _document(self,width,height,detail,arm):
        doc=super()._document(width,height,detail,arm)
        upper=doc.find(".//body[@name='learner_upper_arm']")
        ratio=self.spec.link
        for item in [upper.find("body[@name='learner_elbow']"),upper.find("inertial"),*upper.findall("geom")]:
            if item.get("pos"):
                item.set("pos",stretch(item.get("pos"),ratio))
        inertia=upper.find("inertial")
        tensor=np.fromstring(inertia.get("fullinertia"),sep=" ")
        tensor[1:3]+=(ratio**2-1)*(tensor[1]+tensor[2]-tensor[0])*.5
        tensor[3:5]*=ratio
        inertia.set("fullinertia"," ".join(str(v) for v in tensor))
        mesh=next(m for m in doc.findall("asset/mesh") if "upper_arm_so101_v1" in m.get("file",""))
        mesh.set("scale",stretch(mesh.get("scale","1 1 1"),ratio))
        return doc

    def reset(self,**kwargs):
        obs=super().reset(**kwargs)
        self._drag_force_scale*=self.spec.drag
        return obs


class FamilyWorld(World):
    def __init__(self,native,spec):
        self.native=native
        self.arena=FamilyArena(spec)
        self.handle=native.open(self.arena._model._address)


def utility(scores):
    # Every donor contributes; the worst-performing body gets extra weight.
    each=(.8*scores[...,1]+.2*scores[...,0]+.15*scores[...,2]/300).mean(axis=-1)
    return .70*each.mean(axis=0)+.30*each.min(axis=0)


def fixed_seeds(n,seed):
    return np.random.default_rng(seed).integers(1,2**31-1,n).tolist()


def train(native,anchor):
    rng=np.random.default_rng(169819432)
    worlds=[FamilyWorld(native,spec) for spec in TRAIN]
    best=anchor.copy()
    historical,_=historical_basis(anchor)
    dev_seeds=[fixed_seeds(6,int(s)) for s in rng.integers(2**30,size=len(worlds))]
    checks=[world.prepare(seeds,"held") for world,seeds in zip(worlds,dev_seeds,strict=True)]
    log=[]
    try:
        def check(options):
            return np.asarray([w.play(options,b,cap=270) for w,b in zip(worlds,checks,strict=True)])
        initial=check(best)
        champ=float(utility(initial)[0])
        training_frames=validation_frames=0
        validation_frames+=int(initial[...,2].sum())
        for gen in range(24):
            # The first two candidates use old learned directions. The rest
            # span all motor/visual dimensions with no movement sequences.
            directions=np.stack([np.outer(rng.normal(size=5),rng.normal(size=20)) + .25*rng.normal(size=(5,20))
                                 for _ in range(7)])
            directions/=np.linalg.norm(directions,axis=(1,2))[:,None,None]
            radii=rng.uniform(.054,.17,7)
            dirs=np.concatenate([historical,directions*radii[:,None,None]])
            options=np.concatenate([best[None],best+dirs,best-dirs])
            chosen=rng.choice(len(worlds),4,replace=False)
            seeds=rng.integers(1,2**31-1,size=(4,2)).tolist()
            scores=np.asarray([worlds[j].play(options,worlds[j].prepare(starts,"held"),cap=220)
                               for j,starts in zip(chosen,seeds,strict=True)])
            training_frames+=int(scores[...,2].sum())
            leaders=1+np.argsort(-utility(scores)[1:],kind="stable")[:2]
            tested=check(options[leaders])
            validation_frames+=int(tested[...,2].sum())
            selected=np.argmax(utility(tested))
            value=float(utility(tested)[selected])
            if value>champ:
                best=options[leaders[selected]].copy()
                champ=value
            log.append(dict(generation=gen+1,worlds=chosen.tolist(),seeds=seeds,value=value,best=champ,
                            current_mean=tested[:,selected,:,1].mean(axis=1).tolist(),weights=best.tolist()))
            print("source",gen+1,round(champ,3),flush=True)
        np.savez_compressed(OUT,weights=best)
        with gzip.open(ROOT/"broad-donors-training.json.gz","wt") as f:
            json.dump(dict(domains=[asdict(d) for d in TRAIN],development=dev_seeds,unseen=asdict(NOVEL),
                           initialization="tt24_donor-refined.npz",training_episodes=24*19*4*2,
                           training_frames=training_frames,validation_episodes=30*(1+2*24),
                           validation_frames=validation_frames,initial_donor_means=initial[:,0,:,1].mean(axis=1).tolist(),
                           log=log),f,separators=(",",":"))
    finally:
        for w in worlds:
            w.close()
    return best


def main():
    with np.load(CHECKPOINT,allow_pickle=False) as data:
        anchor=data["weights"]
    native=Native()
    try:
        best=train(native,anchor)
        for i,name in enumerate(("stock","previously_shifted","new_body")):
            arena=(BounceArena(view="wrist",width=192,height=144) if i==0 else
                   ShiftedBounce() if i==1 else FamilyArena(NOVEL,view="wrist"))
            try:
                seeds=fixed_seeds(12,19348133+i)
                groups={}
                for key,w in (("broader",best),("original",anchor)):
                    modes=("public","hold") if key=="broader" else ("public",)
                    rows=_paired(arena,native,w,seeds,modes,"standard" if i==0 else "changed")
                    groups[key]=dict(rows=rows,summary=_summary(rows))
                with gzip.open(ROOT/f"broad-donors-{name}.json.gz","wt") as f:
                    json.dump(dict(name=name,seeds=seeds,groups=groups,all_motor_tapes_verified=True),f,separators=(",",":"))
                print(name,{k:v["summary"]["public"]["mean"] for k,v in groups.items()},flush=True)
            finally:
                arena.close()
    finally:
        native.working.cleanup()


if __name__=="__main__":
    main()
