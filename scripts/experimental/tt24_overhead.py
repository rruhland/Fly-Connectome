"""Can repeated volleys against a physical overhead partner teach solo bouncing?"""
import argparse
import gzip
import json
from pathlib import Path
from xml.etree.ElementTree import SubElement

import numpy as np
import mujoco

from .tt24_bounce import BounceArena
from .tt24_broad_donors import fixed_seeds
from .tt24_donor_assess import ShiftedBounce
from .tt24_donor_trial import DESIGNS, ROOT, DonorBounce, Native, World, _paired, _refine, _summary


class OpposedArm(DonorBounce):
    precise=False

    def _document(self,width,height,detail,arm):
        doc=super()._document(width,height,detail,arm)
        body=SubElement(doc.find("worldbody"),"body",name="overhead_partner",mocap="true",pos="0 0 2")
        SubElement(body,"geom",name="opponent_racket",type="cylinder",size=".048 .004" if self.precise else ".10 .004",mass=".16",
                   rgba=".22 .55 .81 1",contype="0",conaffinity="1",solref="-100000 -24",priority="2")
        return doc

    def reset(self,**kwargs):
        obs=super().reset(**kwargs)
        home=self._data.geom_xpos[self._blade]
        ceiling=.15 + .07*(self._drop.offset[2]-.16)/.16
        if self.precise:
            ceiling=.115+.135*(self._drop.offset[2]-.16)/.16
        offset=(.012,-.008) if self.precise else (.04,-.03)
        self._data.mocap_pos[0]=home+[*offset,ceiling]
        self._ball_position[2]=home[2]+.075
        self._ball_velocity[2]=-.85-5*(self._drop.offset[2]-.16)
        mujoco.mj_forward(self._model,self._data)
        return obs


class PrecisionArm(OpposedArm):
    precise=True


class PartnerWorld(World):
    def __init__(self,native,design,precise=False):
        self.native=native
        self.arena=(PrecisionArm if precise else OpposedArm)(design)
        self.handle=native.open(self.arena._model._address)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds",type=int,default=48)
    parser.add_argument("--directions",type=int,default=10)
    parser.add_argument("--examples",type=int,default=2)
    parser.add_argument("--noise",type=float,default=.22)
    parser.add_argument("--train-cap",type=int,default=200)
    parser.add_argument("--output",type=Path,default=ROOT/"overhead-source-training.json.gz")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"overhead-source.npz")
    parser.add_argument("--precision",action="store_true")
    args=parser.parse_args()
    prefix="overhead-precision" if args.precision else "overhead-source"
    if args.precision:
        args.output=ROOT/(prefix+"-training.json.gz")
        args.checkpoint=ROOT/(prefix+".npz")
    native=Native(Path(__file__).with_name("tt24_overhead_native.cpp"))
    worlds=[PartnerWorld(native,d,args.precision) for d in DESIGNS]
    try:
        weights=_refine(worlds,{"updates":[],"source_reward":"completed physical two-racket pairs",
            "initialization":"all zero"},args)
        checks=[]
        for i,w in enumerate(worlds):
            seeds=fixed_seeds(24,311809721+i+int(args.precision)*100)
            opt=np.stack([weights,np.zeros_like(weights)])
            batch=w.prepare(seeds,"held")
            scores=w.play(opt,batch)
            if not np.array_equal(scores,w.play(opt,batch)):
                raise ValueError("opposed arena changed on replay")
            checks.append(dict(seeds=seeds,scores=scores.tolist()))
        pairs=np.concatenate([np.array(c["scores"])[:,:,1] for c in checks],axis=1)
        print("independent source pairs",pairs.mean(axis=1),pairs.max(axis=1),flush=True)
    finally:
        for w in worlds:
            w.close()
        native.working.cleanup()
    doc=dict(source=checks,source_columns=["completed_pairs","completed_pairs","ticks","near_returns"],
             source_gate="mean one pair ahead of stationary and at least 4 starts with three pairs")
    if pairs[0].mean()>pairs[1].mean()+1. and (pairs[0]>=3).sum()>=4:
        native=Native()
        try:
            for name,arena,regime,seed in (("stock",BounceArena(view="wrist",width=192,height=144,max_steps=300),"standard",45821091),
                   ("tilted",ShiftedBounce(),"changed",45821092)):
                try:
                    rows=_paired(arena,native,weights,fixed_seeds(16,seed+int(args.precision)*100),("public","clean","hold"),regime)
                    doc[name]=dict(cases=rows,summary=_summary(rows))
                finally:
                    arena.close()
                print(name,doc[name]["summary"],flush=True)
        finally:
            native.working.cleanup()
    with gzip.open(ROOT/(prefix+"-held.json.gz"),"wt") as f:
        json.dump(doc,f,separators=(",",":"))


if __name__=="__main__":
    main()
