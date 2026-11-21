"""Fast stereo color sensor for simulated practice, checked against native RGB.

Scene truth is confined to the renderer. The player and its outcome estimator
use only RGB images and calibrations fitted from visible landmarks.
"""
import numpy as np
from .tt18_view import ImageColors, ViewRegistry, triangulate_views, OBJECT_HUES
from .tt19_view import LANDMARKS, _blob


class PracticeCamera:
    def __init__(self, arena):
        self.arena = arena
        self.lenses = []
        for stream in ('left','right'):
            camera = arena.model.camera('tt19_'+stream)
            i = camera.id
            self.lenses.append((stream, arena.data.cam_xpos[i].copy(),
                                arena.data.cam_xmat[i].reshape(3,3).copy(), float(camera.fovy[0])))
        self.width, self.height = arena.width, arena.height

    def capture(self):
        frames = {}
        for stream,eye,axes,degrees in self.lenses:
            p = (self.arena.ball - eye) @ axes
            focal = self.height / (2*np.tan(np.deg2rad(degrees)*.5))
            image = np.zeros((self.height,self.width,3),dtype=np.uint8)
            if p[2] < -.02:
                u,v = (self.width-1)*.5-focal*p[0]/p[2], (self.height-1)*.5+focal*p[1]/p[2]
                radius = max(.8, focal*.020/-p[2])
                x,y = int(round(u)), int(round(v))
                span = int(np.ceil(radius))+1
                lx,hx = max(0,x-span), min(self.width,x+span+1)
                ly,hy = max(0,y-span), min(self.height,y+span+1)
                if lx<hx and ly<hy:
                    yy,xx = np.ogrid[ly:hy,lx:hx]
                    image[ly:hy,lx:hx][(xx-u)**2+(yy-v)**2<=radius**2]=[255,112,0]
            frames[stream]=image
        return frames


class BallEyes:
    def __init__(self, calibration):
        self.fits = ViewRegistry(LANDMARKS).update(calibration)
        if not {'left','right'}.issubset(self.fits):
            raise ValueError('fixed cameras could not register from the RGB landmarks')

    def position(self, frames, simple):
        centers={}
        for stream in ('left','right'):
            frame = frames[stream]
            if simple:
                yy,xx=np.nonzero(frame[:,:,0])
                if len(xx):
                    centers[stream]=np.array((xx.mean(),yy.mean()))
            else:
                colors = ImageColors(frame)
                seen = _blob(colors,OBJECT_HUES['ball'],tolerance=.03,minimum=73.,area=1,
                             span=max(7,colors.shape[0]//9))
                if seen is not None:
                    centers[stream]=seen.center
        return triangulate_views(self.fits,centers)
