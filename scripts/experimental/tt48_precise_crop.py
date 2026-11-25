"""Render ordinary RGB crops at the exact fixed-camera sampling grid."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .tt19_view_trial import CAMERAS
from .tt29_vision import Rig
from .tt42_camera_precision import Dimensions


@dataclass(frozen=True)
class Crop:
    rgb: np.ndarray
    origin: np.ndarray


def rectangle(camera, position):
    projected=camera.project(position)
    if not np.isfinite(projected).all() or projected[2]<.4:
        return None
    ij=np.floor(projected[:2]).astype(int)
    r=int(round(.055*camera.width))
    lim=np.array([camera.width,camera.height])
    lo,hi=np.maximum(ij-r,0),np.minimum(ij+r+1,lim)
    if np.any(hi-lo<.04*camera.width):
        return None
    return lo,hi


class CroppedStereo:
    """Restrict the raster destination, retaining the full sensor projection."""
    def __init__(self, arena, factor):
        self.arena=arena
        scale=Dimensions(arena,factor)
        self.rig=Rig.from_arena(scale)
        self.width,self.height=scale.width,scale.height
        side=2*round(.055*self.width)+1
        model=arena.model
        lens,quality=model.vis.global_,model.vis.quality
        old=lens.offwidth,lens.offheight,quality.offsamples
        try:
            lens.offwidth=lens.offheight=side
            if arena.detail=="preview":
                quality.offsamples=0
            self.renderer=mujoco.Renderer(model,width=side,height=side)
        finally:
            lens.offwidth,lens.offheight,quality.offsamples=old
        self.cameras={}
        for k in ("left","right"):
            cam=mujoco.MjvCamera()
            cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid=model.camera(CAMERAS[k][0]).id
            self.cameras[k]=cam

    def capture(self, center):
        a=self.arena
        r=self.renderer
        r._gl_context.make_current()
        a._color_roles()
        mujoco.mj_kinematics(a.model,a.data)
        mujoco.mj_camlight(a.model,a.data)
        out={}
        for k,cam in self.cameras.items():
            box=rectangle(self.rig.fixed[k],center)
            if box is None:
                continue
            lo,hi=box
            x0,y0=lo
            x1,y1=hi
            if not out:
                mujoco.mjv_updateScene(a.model,a.data,r._scene_option,None,cam,
                                       mujoco.mjtCatBit.mjCAT_ALL,r.scene)
            else:
                mujoco.mjv_updateCamera(a.model,a.data,cam,r.scene)
            # The conventional stereo renderer averages the pair to retain
            # the surveyed pinhole of each camera with no interpupillary shift.
            middle=mujoco.mjv_averageCamera(*r.scene.camera)
            for dest in r.scene.camera:
                for attr in ("pos","forward","up"):
                    getattr(dest,attr)[:]=getattr(middle,attr)
                for attr in ("frustum_bottom","frustum_center","frustum_far","frustum_near",
                             "frustum_top","frustum_width","orthographic"):
                    setattr(dest,attr,getattr(middle,attr))
            r.scene.stereo=mujoco.mjtStereo.mjSTEREO_NONE
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
            # A negative viewport origin leaves the world projection at
            # the full scaled sensor coordinates. Only this subrectangle
            # lies inside the smaller native color/depth buffer.
            view=mujoco.MjrRect(-int(x0),-int(self.height-y1),self.width,self.height)
            read=mujoco.MjrRect(0,0,int(x1-x0),int(y1-y0))
            mujoco.mjr_render(view,r.scene,r._mjr_context)
            pix=np.empty((read.height,read.width,3),np.uint8)
            mujoco.mjr_readPixels(pix,None,read,r._mjr_context)
            out[k]=Crop(pix[::-1],lo)
        return out

    def close(self):
        self.renderer.close()
