"""Cache native material commands while preserving the original matrix operations."""
from OpenGL import GL

from .tt6_render_reflections import ReflectingHallRenderer


class CompiledHallRenderer(ReflectingHallRenderer):
    def __init__(self, *args):
        self._materials = {}
        super().__init__(*args)

    def _material(self, rgba, specular, shininess, emission, reflectance):
        key = (rgba.tobytes(), specular, shininess, emission, reflectance)
        handle = self._materials.get(key)
        if handle is None:
            if len(self._materials) >= 256:
                oldest = next(iter(self._materials))
                GL.glDeleteLists(self._materials.pop(oldest), 1)
            handle = GL.glGenLists(1)
            if not handle:
                return super()._material(rgba, specular, shininess, emission, reflectance)
            GL.glNewList(handle, GL.GL_COMPILE)
            try:
                super()._material(rgba, specular, shininess, emission, reflectance)
            finally:
                GL.glEndList()
            self._materials[key] = handle
        GL.glCallList(handle)

    def close(self):
        if self._materials:
            self._gl_context.make_current()
            for handle in self._materials.values():
                GL.glDeleteLists(handle, 1)
            self._materials.clear()
        super().close()
