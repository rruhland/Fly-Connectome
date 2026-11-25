"""Optional packed off-axis binocular crops, preserving the actual TT75 camera."""
from time import process_time
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import lru_cache
import ctypes
import json
import subprocess

import mujoco
import numpy as np

from .tt29_fast_camera import StereoRenderer,CAMERAS
from .tt75_retina import Retina


class PackedCrop(StereoRenderer):
    """Two real dynamically aimed views in one GL submission and readback."""

    views=2

    def __init__(self, arena, rig, *, size=192, scale=6.):
        self.arena=arena
        self.size, self.scale, self.rig = size, scale, rig
        self.renderer = mujoco.Renderer(arena.model, height=size, width=self.views*size)
        self.cameras=[]
        for stream in ("left","right"):
            c=mujoco.MjvCamera()
            c.type=mujoco.mjtCamera.mjCAMERA_FIXED
            c.fixedcamid=arena.model.camera(CAMERAS[stream][0]).id
            self.cameras.append(c)
        self.prepare_cpu = 0.
        self.render_cpu = 0.
        self.submitted = 0
        self.removed = 0

    def restrict(self,scene):
        pass

    def aim(self, centers):
        self._prepare(mujoco.mjtCatBit.mjCAT_ALL)
        scene=self.renderer.scene
        for camera, name, center in zip(scene.camera, ("left","right"), centers, strict=True):
            fixed=self.rig.fixed[name]
            near=float(camera.frustum_near)
            half=near*self.size/(2*fixed.focal*self.scale)
            x,y=near*(np.asarray(center)-fixed.center)/fixed.focal
            camera.frustum_top=-y+half
            camera.frustum_bottom=-y-half
            camera.frustum_center=x
            camera.frustum_width=2*half
        self.restrict(scene)
        self.submitted+=scene.ngeom
        return scene

    def capture_at(self,centers):
        tick=process_time()
        self.aim(centers)
        self.prepare_cpu+=process_time()-tick
        tick=process_time()
        out=self.renderer.render()
        self.render_cpu+=process_time()-tick
        return {"left":out[:,:self.size],"right":out[:,self.size:]}

    def close(self):
        if self.renderer._gl_context:
            self.renderer._gl_context.make_current()
        super().close()


class PackedRetina(Retina):
    """Exact scheduler and estimator; replace only the binocular raster host."""
    crop_factory=PackedCrop

    def __init__(self,arena,rig,wide,*,rate=120,size=192,scale=6.):
        self.rig,self.wide=rig,wide
        self.rate,self.period,self.scale,self.size=rate,1/rate,scale,size
        self.crop=self.crop_factory(arena,rig,size=size,scale=scale)
        path=Path("artifacts/tt75/sensor-calibration.json")
        self.pixel_maps=[np.asarray(A) for A in json.loads(path.read_text())["pixel_maps"]] if path.exists() else None
        self.reset()


@lru_cache(maxsize=1)
def cull_kernel():
    work=TemporaryDirectory(prefix="tt91-raster-cull-")
    target=Path(work.name)/"cull.so"
    source=Path(__file__).with_name("tt91_retina_cull.cpp")
    headers=Path(mujoco.__file__).parent/"include"
    subprocess.run(["c++","-std=c++17","-O3","-fPIC","-shared","-I",str(headers),str(source),"-o",str(target)],
                   check=True,capture_output=True)
    f=ctypes.CDLL(str(target)).conservative_retina_cull
    f.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
                ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p)
    f.restype=ctypes.c_int
    return f,work


class CulledCrop(PackedCrop):
    """Conservatively skip geometry outside both real off-axis views."""

    def __init__(self,arena,rig,*,size=192,scale=6.):
        super().__init__(arena,rig,size=size,scale=scale)
        self.cull=cull_kernel()[0]
        scene=self.renderer.scene
        self._first=scene.geoms[0]
        self._left,self._right=scene.camera
        self._scene_fields=(self._first.pos.ctypes.data,self._left.pos.ctypes.data,self._right.pos.ctypes.data)

    def restrict(self,scene):
        if scene.ngeom<1:
            return
        skipped=self.cull(*self._scene_fields,scene.ngeom,self._first.type,self._first.objid,
                          self._left.frustum_width,self.arena.model._address)
        if not 0<=skipped<=scene.ngeom:
            raise ValueError("Renderer returned invalid scene compaction")
        scene.ngeom-=skipped
        self.removed+=skipped


class CulledRetina(PackedRetina):
    """Verified scene-local packing and conservative cull, no sensing law change."""
    crop_factory=CulledCrop

    def usable(self,image,sight,A):
        if sight is None:
            return False
        if min(*sight.center,self.size-1-sight.center[0],self.size-1-sight.center[1])<2*sight.radius:
            return False
        # All gate products are at most 25500 and signed 16-bit exact.
        # np.nonzero and thereafter the original moment computation match.
        r,g,b=np.moveaxis(image.astype(np.int16),-1,0)
        chroma=r-b
        mask=(chroma>=64)&(100*(g-b)>25*chroma)&(100*(g-b)<62*chroma)&(100*b<24*r)
        y,x=np.nonzero(mask)
        if len(x)<8:
            return False
        u=(np.column_stack((x,y))-sight.center)@np.linalg.inv(A).T
        cov=u.T@u/len(u)
        e=np.linalg.eigvalsh(cov)
        return bool(e[0]>.60*e[1])


class CulledMonoCrop(CulledCrop):
    """Keep the original two x=0 viewports while culling their union."""
    views=1

    def capture_at(self,centers):
        tick=process_time()
        scene=self.aim(centers)
        vector=("pos","forward","up")
        scalar=("frustum_bottom","frustum_center","frustum_far","frustum_near",
                "frustum_top","frustum_width","orthographic")
        cameras=[([getattr(c,k).copy() for k in vector],[getattr(c,k) for k in scalar]) for c in scene.camera]
        scene.stereo=mujoco.mjtStereo.mjSTEREO_NONE
        self.prepare_cpu+=process_time()-tick
        out={}
        started=process_time()
        for n,(v,s) in zip(("left","right"),cameras,strict=True):
            for c in scene.camera:
                for k,a in zip(vector,v,strict=True):
                    getattr(c,k)[:]=a
                for k,a in zip(scalar,s,strict=True):
                    setattr(c,k,a)
            out[n]=self.renderer.render()
        self.render_cpu+=process_time()-started
        return out


class CulledMonoRetina(CulledRetina):
    """Original TT75 eye viewports, dense timing, and observation law."""
    crop_factory=CulledMonoCrop
