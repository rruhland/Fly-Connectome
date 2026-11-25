"""Read native Mesa BGRA/floating depth and fuse the exact stock stereo composite."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from .tt29_fast_camera import CachedStereoRenderer


class Unsupported(RuntimeError):
    pass


@lru_cache(maxsize=1)
def kernel():
    work=TemporaryDirectory(prefix="tt54-packed-gl-")
    target=Path(work.name)/"pack.so"
    source=Path(__file__).with_suffix(".cpp")
    cmd=["c++","-std=c++17","-O3","-shared","-fPIC",str(source),"-l:libGL.so.1","-o",str(target)]
    try:
        subprocess.run(cmd,check=True,capture_output=True)
        f=ctypes.CDLL(str(target)).packed_capture
    except (subprocess.SubprocessError,OSError) as e:
        work.cleanup()
        raise Unsupported("No optional packed-readback build") from e
    p,i=ctypes.c_void_p,ctypes.c_int
    f.argtypes=(i,i,p,p,p,p,p)
    f.restype=None
    return f,work


class PackedStereo(CachedStereoRenderer):
    """Stock raster and registration, optional readback-only implementation."""

    def __init__(self,arena):
        super().__init__(arena)
        self.hardware=None
        self.fallback=None
        try:
            self.hardware=self.enable()
        except (ImportError,Unsupported) as err:
            self.fallback=str(err)

    def enable(self):
        a,r=self.arena,self.renderer
        if (os.environ.get("MUJOCO_GL","").lower()!="egl"
                or os.environ.get("PYOPENGL_PLATFORM")!="egl" or sys.byteorder!="little"
                or (a.width,a.height)!=(384,288) or a.detail!="preview"
                or tuple(a.streams)!=("left","right")):
            raise Unsupported("Unreviewed stock stereo configuration")
        from OpenGL import GL as gl
        from OpenGL.error import GLError
        r._gl_context.make_current()
        con=r._mjr_context
        if not self.compatible_context():
            raise Unsupported("Unreviewed native depth convention or framebuffer")
        try:
            driver=gl.glGetString(gl.GL_RENDERER)
            if driver is None or not driver.startswith(b"llvmpipe"):
                raise Unsupported("Only Mesa llvmpipe reviewed")
            for att,f in ((gl.GL_COLOR_ATTACHMENT0,gl.GL_RGBA8),(gl.GL_DEPTH_ATTACHMENT,gl.GL_DEPTH32F_STENCIL8)):
                kind=gl.glGetFramebufferAttachmentParameteriv(gl.GL_FRAMEBUFFER,att,gl.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE)
                if int(kind)!=gl.GL_RENDERBUFFER:
                    raise Unsupported("Unreviewed raster attachment")
                obj=gl.glGetFramebufferAttachmentParameteriv(gl.GL_FRAMEBUFFER,att,gl.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME)
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER,obj)
                try:
                    got=gl.glGetRenderbufferParameteriv(gl.GL_RENDERBUFFER,gl.GL_RENDERBUFFER_INTERNAL_FORMAT)
                finally:
                    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER,0)
                if int(got)!=f:
                    raise Unsupported("Unreviewed packed color or depth format")
        except GLError as err:
            raise Unsupported("Packed Mesa attachments unavailable") from err
        self.color=np.empty((con.offHeight,con.offWidth,4),np.uint8)
        self.depth=np.empty((con.offHeight,con.offWidth,2),np.float32)
        self.function=kernel()[0]
        return dict(renderer=str(driver),readback="BGRA8 plus depth32F-stencil8",size=[con.offWidth,con.offHeight])

    def compatible_context(self):
        c=self.renderer._mjr_context
        return (int(c.readDepthMap)==int(mujoco.mjtDepthMap.mjDEPTH_ZEROFAR)
                and c.offSamples==0 and int(c.readPixelFormat)==0x1907
                and (c.offWidth,c.offHeight)==(768,288)
                and int(c.currentBuffer)==int(mujoco.mjtFramebuffer.mjFB_OFFSCREEN))

    def capture(self):
        if self.hardware is None or not self.compatible_context():
            return super().capture()
        r=self.renderer
        self._prepare(mujoco.mjtCatBit.mjCAT_DYNAMIC)
        r._gl_context.make_current()
        mujoco.mjr_render(r._rect,r.scene,r._mjr_context)
        frame=np.empty_like(self.background)
        self.function(768,288,self.color.ctypes.data,self.depth.ctypes.data,self.background.ctypes.data,
                      self.background_depth.ctypes.data,frame.ctypes.data)
        return self._split(frame[::-1])
