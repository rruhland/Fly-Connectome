"""Preserve live floor reflections in cached native hall rendering."""

from dataclasses import dataclass

import mujoco
import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader

from .tt6_render import CachedHallRenderer


_VERTEX = '''#version 400 compatibility
void main() {
    gl_Position = gl_Vertex;
}
'''
_FRAGMENT = '''#version 400 compatibility
uniform sampler2DMS floorColor;
uniform sampler2DMS reflectionColor;
out vec4 color;
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    color = texelFetch(floorColor, pixel, gl_SampleID)
          + texelFetch(reflectionColor, pixel, gl_SampleID);
}
'''


@dataclass
class _Reflection:
    floor: int
    opaque: int
    buffers: np.ndarray
    reflectance: float
    lights: list


class ReflectingHallRenderer(CachedHallRenderer):
    """Shade native moving geometry in a cached hall and its floor reflection."""

    def __init__(self, model, height, width):
        self._resources = {}
        self._reflection = None
        self._mirror_fbo = self._mirror_depth = self._mirror_color = self._program = 0
        super().__init__(model, height, width)
        if not self._context.offSamples:
            raise ValueError('reflection cache requires native multisampling')
        self._mirror_fbo, self._mirror_color, self._mirror_depth = self._sample_texture(depth=True)
        self._program = compileProgram(
            compileShader(_VERTEX, GL.GL_VERTEX_SHADER), compileShader(_FRAGMENT, GL.GL_FRAGMENT_SHADER),
            validate=False)
        GL.glUseProgram(self._program)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, 'floorColor'), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, 'reflectionColor'), 1)
        GL.glUseProgram(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._context.offFBO)

    def _sample_texture(self, *, depth=False):
        framebuffer = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        texture = GL.glGenTextures(1)
        target = GL.GL_TEXTURE_2D_MULTISAMPLE
        GL.glBindTexture(target, texture)
        GL.glTexImage2DMultisample(target, self._context.offSamples, GL.GL_RGBA8,
                                   self.width, self.height, True)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, target, texture, 0)
        GL.glBindTexture(target, 0)
        buffer = 0
        if depth:
            buffer = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, buffer)
            GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self._context.offSamples,
                                                self._depth_format, self.width, self.height)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                        GL.GL_RENDERBUFFER, buffer)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('could not allocate hall reflection samples')
        return framebuffer, texture, buffer

    @staticmethod
    def _restore_view(view, projection):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(projection)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixf(view)

    @staticmethod
    def _enable_lights(lights):
        for i, (kind, position, direction) in enumerate(lights):
            slot = GL.GL_LIGHT0 + i
            if kind == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                GL.glLightfv(slot, GL.GL_POSITION, (*(-direction), 0))
            else:
                GL.glLightfv(slot, GL.GL_POSITION, (*position, 1))
                GL.glLightfv(slot, GL.GL_SPOT_DIRECTION, direction)
            GL.glEnable(slot)

    def _begin_reflection(self, view, projection, lights):
        self._restore_view(view, projection)
        GL.glClipPlane(GL.GL_CLIP_PLANE0, (0., 0., -1., 0.))
        GL.glEnable(GL.GL_CLIP_PLANE0)
        GL.glFrontFace(GL.GL_CW)
        GL.glScalef(1., 1., -1.)
        self._enable_lights(lights)

    @staticmethod
    def _end_reflection():
        GL.glDisable(GL.GL_CLIP_PLANE0)
        GL.glFrontFace(GL.GL_CCW)

    def _cache_eye(self, data, camera):
        base = super()._cache_eye(data, camera)
        background, _, view, projection, _ = base
        context = self._context
        scene = self.scene
        floor = scene.geoms[0]
        if (floor.objid != self.model.geom('floor').id or np.any(floor.pos != 0)
                or not np.array_equal(floor.mat.reshape(3, 3), np.eye(3))):
            raise ValueError('the reflecting hall floor must lie in the world XY plane')
        reflectance = floor.reflectance
        lights = [(light.type, light.pos.copy(), light.dir.copy()) for light in scene.lights[:scene.nlight]]
        static = [self._parameters(geom) for geom in scene.geoms[1:scene.ngeom]]

        # Mark exactly the native samples where the floor is the foreground surface.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, background)
        GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
        GL.glDepthFunc(GL.GL_EQUAL)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glColorMask(False, False, False, False)
        GL.glEnable(GL.GL_STENCIL_TEST)
        GL.glStencilFunc(GL.GL_ALWAYS, 1, 0xFF)
        GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_REPLACE)
        GL.glCallList(context.basePlane + floor.dataid + 1)
        GL.glDisable(GL.GL_STENCIL_TEST)
        GL.glColorMask(True, True, True, True)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_GEQUAL)

        # Bake native floor lighting, texture, and coverage into multisample color.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, context.offFBO)
        scene.ngeom = 1
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        mujoco.mjr_render(self._rect, scene, context)
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        floor_fbo, floor_color, _ = self._sample_texture()
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, context.offFBO)
        GL.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height,
                             GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glDeleteFramebuffers(1, [floor_fbo])

        # Cache the color and depth of opaque, stationary geometry seen in the mirror.
        opaque, buffers = self._framebuffer()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)
        self._begin_reflection(view, projection, lights)
        for params in static:
            self._draw(params, reflectance)
        self._end_reflection()
        for i in range(len(lights)):
            GL.glDisable(GL.GL_LIGHT0 + i)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, context.offFBO)
        self._resources[camera.fixedcamid] = _Reflection(floor_color, opaque, buffers, reflectance, lights)
        return base

    def update_scene(self, data, camera=-1, scene_option=None):
        super().update_scene(data, camera, scene_option)
        self._reflection = self._resources[self.model.camera(camera).id]

    def _composite_floor(self, floor_color):
        GL.glEnable(GL.GL_STENCIL_TEST)
        GL.glStencilFunc(GL.GL_EQUAL, 1, 0xFF)
        GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_KEEP)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self._program)
        target = GL.GL_TEXTURE_2D_MULTISAMPLE
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(target, floor_color)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(target, self._mirror_color)
        GL.glBegin(GL.GL_TRIANGLE_STRIP)
        GL.glVertex2f(-1., -1.)
        GL.glVertex2f(1., -1.)
        GL.glVertex2f(-1., 1.)
        GL.glVertex2f(1., 1.)
        GL.glEnd()
        GL.glBindTexture(target, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(target, 0)
        GL.glUseProgram(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_STENCIL_TEST)

    def render(self, *, out=None):
        self._gl_context.make_current()
        context = self._context
        background, _, view, projection, _ = self._eye
        reflection = self._reflection
        moving = self._moving_parameters()

        self._copy(reflection.opaque, self._mirror_fbo)
        self._begin_reflection(view, projection, reflection.lights)
        self._draw_foreground(moving, reflection.reflectance)
        self._end_reflection()

        self._copy(background, context.offFBO)
        self._composite_floor(reflection.floor)
        self._restore_view(view, projection)
        self._enable_lights(reflection.lights)
        self._draw_foreground(moving)
        for i in range(len(reflection.lights)):
            GL.glDisable(GL.GL_LIGHT0 + i)
        if out is None:
            out = np.empty((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(out, None, self._rect, context)
        out[:] = out[::-1]
        return out

    def close(self):
        if self._program or self._resources:
            self._gl_context.make_current()
            GL.glDeleteProgram(self._program)
            GL.glDeleteFramebuffers(1, [self._mirror_fbo])
            GL.glDeleteRenderbuffers(1, [self._mirror_depth])
            GL.glDeleteTextures([self._mirror_color])
            self._program = 0
            for reflection in self._resources.values():
                GL.glDeleteFramebuffers(1, [reflection.opaque])
                GL.glDeleteRenderbuffers(2, reflection.buffers)
                GL.glDeleteTextures([reflection.floor])
            self._resources.clear()
        super().close()
