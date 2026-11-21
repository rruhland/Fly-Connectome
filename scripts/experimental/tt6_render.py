"""Reuse the native multisample hall background for fixed learning cameras."""

import mujoco
import numpy as np
from OpenGL import GL


_STATIC = mujoco.mjtCatBit.mjCAT_STATIC.value
_MOVING = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
# Display list offsets from MuJoCo 3.17's classic render_context.h.
_SPHERE_TOP, _SPHERE_BOTTOM, _OPEN_CYLINDER = 1, 2, 4
_PRIMITIVE = {
    mujoco.mjtGeom.mjGEOM_SPHERE.value: 0,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID.value: 0,
    mujoco.mjtGeom.mjGEOM_CYLINDER.value: 3,
    mujoco.mjtGeom.mjGEOM_BOX.value: 6,
}
_CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE.value
_LIGHTS = (mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, mujoco.mjtLightType.mjLIGHT_SPOT)
_BLIT_MASK = GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT


class CachedHallRenderer(mujoco.Renderer):
    """Draw moving primitives and the transparent net over cached native samples."""

    def __init__(self, model, height, width):
        self._eyes = {}
        self._eye = None
        super().__init__(model, height=height, width=width, max_geom=model.ngeom)
        self._context = self._mjr_context
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._context.offDepthStencil)
        self._depth_format = GL.glGetRenderbufferParameteriv(
            GL.GL_RENDERBUFFER, GL.GL_RENDERBUFFER_INTERNAL_FORMAT)
        self._net_id = model.geom('net').id
        self._net = None

    def _framebuffer(self):
        context = self._context
        framebuffer = GL.glGenFramebuffers(1)
        buffers = GL.glGenRenderbuffers(2)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        for buffer, fmt, attachment in zip(
                buffers, (GL.GL_RGBA8, self._depth_format),
                (GL.GL_COLOR_ATTACHMENT0, GL.GL_DEPTH_STENCIL_ATTACHMENT), strict=True):
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, buffer)
            GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, context.offSamples, fmt,
                                                self.width, self.height)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, attachment, GL.GL_RENDERBUFFER, buffer)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('could not allocate a multisample hall background')
        return framebuffer, buffers

    def _copy(self, source, target):
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, source)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, target)
        GL.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height,
                             _BLIT_MASK, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target)

    def _camera(self, name):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = self.model.camera(name).id
        return camera

    def _cache_eye(self, data, camera):
        self._gl_context.make_current()
        context = self._context
        alpha = self.model.geom_rgba[self._net_id, 3].copy()
        try:
            self.model.geom_rgba[self._net_id, 3] = 0
            mujoco.mjv_updateScene(self.model, data, self._scene_option, None, camera, _STATIC, self.scene)
        finally:
            self.model.geom_rgba[self._net_id, 3] = alpha
        if any(light.castshadow or light.type not in _LIGHTS for light in self.scene.lights[:self.scene.nlight]):
            raise ValueError('cached rendering requires fixed classic hall lights without shadows')
        mujoco.mjr_render(self._rect, self.scene, context)
        view = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)
        projection = GL.glGetFloatv(GL.GL_PROJECTION_MATRIX)
        lights = [(GL.glGetLightfv(GL.GL_LIGHT0 + i, GL.GL_POSITION),
                   GL.glGetLightfv(GL.GL_LIGHT0 + i, GL.GL_SPOT_DIRECTION))
                  for i in range(self.scene.nlight)]
        cache, buffers = self._framebuffer()
        self._copy(context.offFBO, cache)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, context.offFBO)
        return cache, buffers, view, projection, lights

    def update_scene(self, data, camera=-1, scene_option=None):
        if scene_option is not None:
            raise ValueError('cached rendering uses the native hall scene options')
        name = camera
        camera = self._camera(name)
        if name not in self._eyes:
            if self._net is None:
                mujoco.mjv_updateScene(self.model, data, self._scene_option, None, camera, _STATIC, self.scene)
                self._net = next(self._parameters(g) for g in self.scene.geoms[:self.scene.ngeom]
                                 if g.objid == self._net_id)
            self._eyes[name] = self._cache_eye(data, camera)
        self._eye = self._eyes[name]
        mujoco.mjv_updateScene(self.model, data, self._scene_option, None, camera, _MOVING, self.scene)

    @staticmethod
    def _parameters(geom):
        if geom.matid >= 0 or (geom.type != _CAPSULE and geom.type not in _PRIMITIVE):
            raise ValueError('cached rendering supports untextured hall primitives')
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = geom.mat.reshape(3, 3).T
        return (geom.type, geom.pos.copy(), geom.size.copy(), matrix,
                geom.rgba.copy(), geom.specular, geom.shininess, geom.emission)

    @staticmethod
    def _material(rgba, specular, shininess, emission, reflectance):
        if reflectance != 1.:
            rgba = rgba.copy()
            rgba[:3] *= reflectance
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_EMISSION, (*rgba[:3] * emission, 1))
        GL.glColor4fv(rgba)
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, shininess * 128)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (specular, specular, specular, 1))

    def _shape(self, kind, size):
        GL.glScalef(*size)
        base = self._context.baseBuiltin
        if kind == _CAPSULE:
            radius = .5 * (size[0] + size[1])
            GL.glCallList(base + _OPEN_CYLINDER)
            GL.glScalef(1, 1, radius / size[2])
            GL.glTranslatef(0, 0, (size[2] - radius) / radius)
            GL.glCallList(base + _SPHERE_TOP)
            GL.glTranslatef(0, 0, 2 * (radius - size[2]) / radius)
            GL.glCallList(base + _SPHERE_BOTTOM)
        else:
            GL.glCallList(base + _PRIMITIVE[kind])

    def _draw(self, params, reflectance=1.):
        kind, pos, size, matrix, rgba, specular, shininess, emission = params
        self._material(rgba, specular, shininess, emission, reflectance)
        GL.glPushMatrix()
        GL.glTranslatef(*pos)
        GL.glMultMatrixf(matrix)
        self._shape(kind, size)
        GL.glPopMatrix()

    def _moving_parameters(self):
        return [self._parameters(geom) for geom in self.scene.geoms[:self.scene.ngeom]]

    def _draw_foreground(self, moving, reflectance=1.):
        for params in moving:
            self._draw(params, reflectance)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        self._draw(self._net, reflectance)
        self._draw(self._net, reflectance)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

    def render(self, *, out=None):
        self._gl_context.make_current()
        context = self._context
        cache, _, view, projection, lights = self._eye
        self._copy(cache, context.offFBO)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(projection)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        for i, (pos, direction) in enumerate(lights):
            GL.glLightfv(GL.GL_LIGHT0 + i, GL.GL_POSITION, pos)
            GL.glLightfv(GL.GL_LIGHT0 + i, GL.GL_SPOT_DIRECTION, direction)
            GL.glEnable(GL.GL_LIGHT0 + i)
        GL.glLoadMatrixf(view)
        self._draw_foreground(self._moving_parameters())
        for i in range(len(lights)):
            GL.glDisable(GL.GL_LIGHT0 + i)
        if out is None:
            out = np.empty((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(out, None, self._rect, context)
        out[:] = out[::-1]
        return out

    def close(self):
        if self._eyes:
            self._gl_context.make_current()
            for framebuffer, buffers, *_ in self._eyes.values():
                GL.glDeleteFramebuffers(1, [framebuffer])
                GL.glDeleteRenderbuffers(2, buffers)
            self._eyes.clear()
        super().close()


def cache_hall_stereo(arena, *, reflections=False, compiled_geometry=False):
    """Install the cache before the first native hall observation.

    With reflections enabled, actors are drawn in both the view and floor mirror.
    The shorter development path retains the static hall reflection.
    """
    if arena.venue != 'hall' or arena.render_mode != 'mujoco' or arena._renderer is not None:
        raise ValueError('expected an uncaptured native hall')
    arena._apply_camera_rig()
    renderer = CachedHallRenderer
    if compiled_geometry and not reflections:
        raise ValueError('compiled geometry requires the reflective renderer')
    if compiled_geometry:
        from .tt8_render import CompiledHallRenderer

        renderer = CompiledHallRenderer
    elif reflections:
        from .tt6_render_reflections import ReflectingHallRenderer

        renderer = ReflectingHallRenderer
    size = arena.model.vis.global_
    native_width, native_height = size.offwidth, size.offheight
    if arena.width > native_width or arena.height > native_height:
        raise ValueError('camera exceeds the native offscreen buffer')
    try:
        size.offwidth, size.offheight = arena.width, arena.height
        arena._renderer = renderer(arena.model, arena.height, arena.width)
    finally:
        size.offwidth, size.offheight = native_width, native_height
