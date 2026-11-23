"""Load a carried five-joint policy and drive it from wrist RGB and encoder observations."""
import argparse
import gzip
import json
from math import tanh
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristState
from .tt24_compound_carry import templates
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import ROOT, FUSED_OUTPUT
from .tt24_public_body import PublicDecoder, PublicMotion, GatedWrist, ConsensusWrist
from .tt24_public_morph import OUTPUT as MORPH, OpticalLinks, MorphDecoder
from .tt24_view_recall import descriptor

BUNDLE=ROOT/"carried-wrist.npz"


def commands(state,weights,issued):
    """Five continuously recomputed joint rates. The final joint closes the gripper separately."""
    bx,by=np.clip((state[:2]-state[6:8])/.065,-3.,3.)
    bz=np.clip((state[2]-state[8]-.13)/.13,-3.,3.)
    vx,vy=np.clip(state[3:5]/.38,-3.,3.)
    vz=np.clip(state[5]/1.50,-3.,3.)
    features=np.array([1.,bx,by,bz,vx,vy,vz,*np.clip(state[6:9]/.04,-3.,3.),
                       *np.clip(state[9:11]/.14,-3.,3.),*np.clip([bz*vz,vz*vz-.5,bz*bz-.5,
                       bx*vz,by*vz,vx*vz,vy*vz],-3.,3.),tanh(2.*vz)])
    # Preserve the original C++ readout's addition order in deterministic physical rollouts.
    values=np.zeros(5)
    for column,feature in enumerate(features):
        values+=feature*weights[:,column]
    targets=.245*np.array([tanh(value) for value in values])
    action=np.clip((targets-issued)/(5.2/60.),-.65,.65)
    issued+=action*(5.2/60.)
    return action


def build(path=BUNDLE):
    arenas=[BounceArena(view="wrist",width=192,height=144,arm=b) for b in ("stock","alternate")]
    try:
        images=templates(arenas,archived=True)
    finally:
        for arena in arenas:
            arena.close()
    params={"schema":np.array("tt24-carried-wrist-1"),"reference_images":images}
    for prefix,source in (("stock",FUSED_OUTPUT),("changed",MORPH)):
        with np.load(source,allow_pickle=False) as data:
            params.update({f"{prefix}_{k}":data[k] for k in data})
    with np.load(CHECKPOINT,allow_pickle=False) as data:
        params["weights"]=data["weights"]
    with np.load(ROOT/"posture-source.npz",allow_pickle=False) as data:
        params["equilibrium"]=data["center"]
    with gzip.open(ROOT/"tt24_pong_origin-lock.json.gz","rt") as f:
        params["reuse"]=np.array(json.load(f)["selected"]["native"])
    np.savez_compressed(path,**params)


class CarriedWristPolicy:
    """One set of motor weights, two calibrated wrist models, and a learned reference preference."""
    def __init__(self,path=BUNDLE):
        with np.load(path,allow_pickle=False) as archive:
            if archive["schema"].item()!="tt24-carried-wrist-1":
                raise ValueError("unsupported carried wrist checkpoint")
            self.weights=archive["weights"]
            self.face_normals={}
            if "wrist_rim_indices" in archive:
                for index, normal in zip(archive["wrist_rim_indices"], archive["wrist_rim_normals"], strict=True):
                    if int(index) not in (0, 1) or normal.shape!=(3,) or not np.isclose(np.linalg.norm(normal), 1.):
                        raise ValueError("invalid wrist tool normal")
                    self.face_normals[int(index)]=normal
            self.equilibrium=archive["equilibrium"]
            self.reuse=float(archive["reuse"])
            self.references=archive["reference_images"]
            self.calibration=[{k[len(prefix)+1:]:archive[k] for k in archive if k.startswith(prefix+"_")}
                              for prefix in ("stock","changed")]
        self.sensor=None
        self.issued=None
        self.calibration_index=None

    def begin_release(self,observation):
        """Start the visual history at a new release. All learned parameters carry across releases."""
        q=np.asarray(observation["joints"])
        self.calibration_index=int(np.argmax((self.references@descriptor(observation["rgb"])).max(axis=1)))
        p=self.calibration[self.calibration_index]
        optical=(p["offset"],p["normal"],p["registration"],p["default"])
        motion=PublicMotion(p["motion_mean"],p["motion_scale"],p["motion_weights"])
        if self.calibration_index:
            decoder=MorphDecoder(OpticalLinks(p["shifts"],p["tilts"]),*optical)
            self.sensor=ConsensusWrist(decoder,motion)
        else:
            decoder=PublicDecoder(*optical)
            kind=int(p["observer"]) if "observer" in p else int(bool(p.get("shared",False)))
            self.sensor=(WristState,GatedWrist,ConsensusWrist)[kind](decoder,motion)
        target=q+self.reuse*(self.equilibrium-q)
        self.sensor.home=decoder.body(target)[12:15].copy()
        self.issued=(q-target).astype(float)

    def act(self,observation):
        if self.sensor is None:
            raise RuntimeError("call begin_release with the first wrist observation")
        state=self.sensor.observe(observation)
        if self.calibration_index in self.face_normals:
            _, rotations=self.sensor.decoder.links.read(np.asarray(observation["joints"])[None,:5])
            normal=rotations[0]@self.face_normals[self.calibration_index]
            state[9:11]=normal[:2] if normal[2]>=0 else -normal[:2]
        return commands(state,self.weights,self.issued)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=BUNDLE)
    build(parser.parse_args().output)
