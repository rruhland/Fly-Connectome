"""Actual small-FOV binocular RGB, aimed solely by previous RGB detections."""
from collections import deque
from time import process_time
from pathlib import Path
import json
import numpy as np
import mujoco
from .tt29_fast_camera import StereoRenderer,native_ball_pixels
from .tt35_error import triangulate


class CroppedStereo(StereoRenderer):
    def __init__(self,arena,rig,*,size=192,scale=6.):
        super().__init__(arena)
        self.renderer.close();self.size=size;self.scale=scale;self.rig=rig
        self.renderer=mujoco.Renderer(arena.model,height=size,width=size)

    def capture_at(self,centers):
        self._prepare(mujoco.mjtCatBit.mjCAT_ALL)
        scene=self.renderer.scene
        fields=('pos','forward','up','frustum_bottom','frustum_center','frustum_far',
                'frustum_near','frustum_top','frustum_width','orthographic')
        originals=[{k:np.asarray(getattr(gl,k)).copy() for k in fields} for gl in scene.camera]
        scene.stereo=mujoco.mjtStereo.mjSTEREO_NONE
        output={}
        for original,name,center in zip(originals,('left','right'),centers,strict=True):
            camera=self.rig.fixed[name];near=float(original['frustum_near'])
            half=near*self.size/(2*camera.focal*self.scale)
            shift=near*(np.asarray(center)-camera.center)/camera.focal
            for gl in scene.camera:
                for k,v in original.items():
                    if k in ('pos','forward','up'):getattr(gl,k)[:]=v
                    else:setattr(gl,k,v.item())
                gl.frustum_top=-shift[1]+half;gl.frustum_bottom=-shift[1]-half
                gl.frustum_center=shift[0];gl.frustum_width=2*half
            output[name]=self.renderer.render().copy()
        return output


class Retina:
    """Global RGB reacquisition plus directly rendered high-magnification crops.

    `arena` is used only by the RGB renderer. No dynamic ball/body state is read.
    Camera timestamps and externally supplied own encoders are public sensor data.
    """
    def __init__(self,arena,rig,wide,*,rate=120,size=192,scale=6.):
        self.rig=rig;self.wide=wide;self.crop=CroppedStereo(arena,rig,size=size,scale=scale)
        self.rate=rate;self.period=1/rate;self.scale=scale;self.size=size
        calibration=Path('artifacts/tt75/sensor-calibration.json')
        self.pixel_maps=[np.asarray(A) for A in json.loads(calibration.read_text())['pixel_maps']] if calibration.exists() else None
        self.reset()

    def reset(self):
        self.points=deque(maxlen=6);self.last_wide=-10.;self.force_wide=True
        self.calls=0;self.wide_calls=0;self.valid=0;self.cpu=0.

    def calibrate_pixels(self,centers,original):
        # Measure real crop-axis magnification by moving only the optical ROI.
        # The physical scene is fixed; neither native position nor depth enters.
        shifts=[]
        for axis in range(2):
            moved=[np.asarray(c)+np.eye(2)[axis]*8 for c in centers]
            frames=self.crop.capture_at(moved)
            shifts.append([native_ball_pixels(frames[n]) for n in ('left','right')])
        maps=[]
        for camera in range(2):
            if original[camera] is None or any(s[camera] is None for s in shifts):return False
            A=np.column_stack([(original[camera].center-s[camera].center)/8 for s in shifts])
            if min(np.linalg.svd(A,compute_uv=False))<1:return False
            maps.append(A)
        self.pixel_maps=maps;return True

    def usable(self,image,sight,A):
        if sight is None:return False
        if min(*sight.center,self.size-1-sight.center[0],self.size-1-sight.center[1])<2*sight.radius:return False
        rgb=image.astype(float);r,g,b=np.moveaxis(rgb,-1,0);chroma=r-b
        mask=(chroma>=64)&(100*(g-b)>25*chroma)&(100*(g-b)<62*chroma)&(100*b<24*r)
        y,x=np.nonzero(mask)
        if len(x)<8:return False
        u=(np.column_stack((x,y))-sight.center)@np.linalg.inv(A).T
        cov=u.T@u/len(u);e=np.linalg.eigvalsh(cov)
        return bool(e[0]>.60*e[1])

    def predicted(self,now):
        if not self.points:return None
        t,p=self.points[-1]
        if now-t>.075:return None
        if len(self.points)<2:return p.copy()
        times=np.array([x[0] for x in self.points]);positions=np.array([x[1] for x in self.points])
        dt=times-times.mean();velocity=(dt@positions)/(dt@dt)
        return p+np.clip(velocity,-15,15)*(now-t)+np.array([0.,0.,-4.905])*(now-t)**2

    def capture(self,now):
        start=process_time();self.calls+=1;centers=None;wide_frames=None
        if self.force_wide or now-self.last_wide>=1/30-1e-7:
            wide_frames=self.wide.capture();self.wide_calls+=1;self.last_wide=now
            sights=[native_ball_pixels(wide_frames[name]) for name in ('left','right')]
            if all(s is not None and not s.clipped for s in sights):centers=[s.center for s in sights]
        if centers is None:
            prediction=self.predicted(now)
            if prediction is not None:centers=[self.rig.fixed[name].project(prediction)[:2] for name in ('left','right')]
        if centers is None:
            self.cpu+=process_time()-start;return None,wide_frames,None
        frames=self.crop.capture_at(centers)
        local=[native_ball_pixels(frames[name]) for name in ('left','right')]
        point=None;global_pixels=None
        if self.pixel_maps is None:self.calibrate_pixels(centers,local)
        if self.pixel_maps is not None and all(self.usable(frames[n],s,A) for n,s,A in zip(('left','right'),local,self.pixel_maps,strict=True)):
            middle=(self.size-1)/2
            global_pixels=[np.asarray(c)+np.linalg.solve(A,s.center-middle) for c,s,A in zip(centers,local,self.pixel_maps,strict=True)]
            point=triangulate(global_pixels,self.rig)
        self.force_wide=point is None
        if point is not None:
            self.points.append((float(now),point.copy()));self.valid+=1
        self.cpu+=process_time()-start
        return point,wide_frames,dict(frames=frames,centers=[c.tolist() for c in centers],
            global_pixels=None if global_pixels is None else [p.tolist() for p in global_pixels])

    def close(self):self.crop.close()
