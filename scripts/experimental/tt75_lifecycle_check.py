"""Actual issued-prefix diagnosis of camera lifetime; no policy tuning."""
from . import tt75_ordinary_play as tt
from .tt75_observer_adapter import ExposureSession
from .tt29_fast_camera import native_ball_pixels
from .tt4_transfer_trial import _read,_write
from time import monotonic,process_time
import numpy as np


def see(frames):
 out={}
 for k,v in frames.items():
  p=native_ball_pixels(v)
  out[k]=dict(sum=int(v.sum()),ball=None if p is None else p.center.tolist())
 return out


def run():
 p=_read(tt.PLAN);old=_read(tt.BASELINE);out={'rows':[],'actual_control_ticks':0};start=monotonic();cpu=process_time()
 with tt.tt80.enabled(),tt.tt80.course.mpc.centered(old['ready']):
  for safe in (False,True):
   a=tt.tt80.course.mpc.arena();wide,rig,*_=tt.tt80.course.mpc.resources(a)
   try:
    for i in range(2):
     pre=wide.capture();s=ExposureSession(a,rig,wide,p['settings']);post=wide.capture()
     a.launch(old['rows'][i]['shot'],card=old['cards'][i]);a.partner.reset(seed=old['cards'][i]['seed']);s.begin()
     for u in old['rows'][i]['actions'][:85]:
      _,_,done,capped,_=a.exchange(u,tt.tt80.course.mpc.FAR_COMMAND);out['actual_control_ticks']+=1
      if done or capped:break
     before=wide.capture();metrics=s.stats()
     if safe:s.close()
     else:a._contacts=s.original;s.retina.close()
     after=wide.capture()
     row=dict(safe=safe,case=i,before_create=see(pre),after_create=see(post),before_close=see(before),after_close=see(after),stats=metrics,close_equal={k:np.array_equal(before[k],after[k]) for k in before},time=float(a.data.time))
     out['rows'].append(row);print(row,flush=True)
   finally:wide.renderer._gl_context.make_current();wide.close();a.close()
 out.update(wall_seconds=monotonic()-start,cpu_seconds=process_time()-cpu)
 _write(tt.ROOT/'lifecycle-check.json',out)


if __name__=='__main__':run()
