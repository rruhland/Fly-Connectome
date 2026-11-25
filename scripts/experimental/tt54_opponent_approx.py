"""Optional exact implementation of the ordinary opponent's approximate score screen."""

import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from . import tt29_partner as stock

BASE=stock._flight_approx


@lru_cache(maxsize=1)
def compiled():
    root=TemporaryDirectory(prefix="tt54-opponent-")
    location=Path(root.name)/"approx.so"
    source=Path(__file__).with_suffix(".cpp")
    try:
        subprocess.run(["c++","-std=c++17","-O3","-ffp-contract=off","-fPIC","-shared",str(source),"-o",str(location)],
                       check=True,capture_output=True)
        f=ctypes.CDLL(str(location)).approximate_flight
        p,i,d=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
        f.argtypes=(p,p,p,p,i,d,d,d,d,p,p,p,p)
        f.restype=i
        return f,root
    except (OSError,subprocess.SubprocessError):
        root.cleanup()
        return None


def approximate(pos,vel,spin,target):
    n=len(pos)
    inputs=pos,vel,spin,target
    if (not 0<n<=10000 or any(type(a) is not np.ndarray or a.dtype!=np.float64 for a in inputs)
            or any(a.shape!=(n,3) for a in inputs[:3]) or target.shape!=(4,)
            or any(not np.isfinite(a).all() for a in inputs)):
        return BASE(*inputs)
    kernel=compiled()
    if kernel is None:
        return BASE(*inputs)
    p,v,w,t=(np.ascontiguousarray(a) for a in inputs)
    records=np.empty((105*n,5))
    index=np.empty(105*n,dtype=np.intc)
    crossing=np.full(n,np.nan)
    land=np.full(n,np.nan)
    count=kernel[0](p.ctypes.data,v.ctypes.data,w.ctypes.data,t.ctypes.data,n,stock.DRAG,stock.LIFT,
                    stock.BALL_RADIUS,stock.TABLE_TOP,records.ctypes.data,index.ctypes.data,
                    crossing.ctypes.data,land.ctypes.data)
    selected=records[:count]
    cost=np.ascontiguousarray(selected[:,:3]) @ [7.,9.,8.] + selected[:,3]
    cost+=selected[:,4]
    best=np.full(n,np.inf)
    np.minimum.at(best,index[:count],cost)
    legal=(crossing>stock.NET_TOP+.012)&(land<-.14)
    best[~legal]=np.inf
    best+=np.where(legal,4.*np.maximum(0.,stock.NET_TOP+.055-crossing),0.)
    return best
