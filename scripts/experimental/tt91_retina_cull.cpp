#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mujoco/mujoco.h>

static bool plausible(const mjvGLCamera &c, const float *p, double radius) {
    if (c.orthographic || c.frustum_near <= 0.) return true;
    const double vx=p[0]-c.pos[0], vy=p[1]-c.pos[1], vz=p[2]-c.pos[2];
    const double depth=vx*c.forward[0]+vy*c.forward[1]+vz*c.forward[2];
    // Leave nearby or potentially clipped geometry to the standard GL path.
    if (depth < 2.*radius) return true;
    const double vert=vx*c.up[0]+vy*c.up[1]+vz*c.up[2];
    const double rx=c.forward[1]*c.up[2]-c.forward[2]*c.up[1];
    const double ry=c.forward[2]*c.up[0]-c.forward[0]*c.up[2];
    const double rz=c.forward[0]*c.up[1]-c.forward[1]*c.up[0];
    const double horiz=vx*rx+vy*ry+vz*rz;
    // Widen beyond the configured native horizontal aperture in both
    // directions. The asymmetric crop's measured horizontal scale is 3x.
    const double half=1.25*std::max(c.frustum_width,c.frustum_top-c.frustum_bottom);
    const double xmin=(c.frustum_center-half)/c.frustum_near;
    const double xmax=(c.frustum_center+half)/c.frustum_near;
    const double ymin=(c.frustum_bottom-.12*(c.frustum_top-c.frustum_bottom))/c.frustum_near;
    const double ymax=(c.frustum_top+.12*(c.frustum_top-c.frustum_bottom))/c.frustum_near;
    return !(horiz > xmax*depth+radius*std::hypot(1.,xmax)
          || horiz < xmin*depth-radius*std::hypot(1.,xmin)
          || vert > ymax*depth+radius*std::hypot(1.,ymax)
          || vert < ymin*depth-radius*std::hypot(1.,ymin));
}

extern "C" int conservative_retina_cull(float* first_pos, const float* left_pos,
    const float* right_pos, int all, int first_type, int first_id, float width,
    const mjModel* model) {
    // Bind through documented array fields when the Python binding does not
    // expose a scene address. Verify the host layout before every mutation.
    auto* geoms=reinterpret_cast<mjvGeom*>(reinterpret_cast<char*>(first_pos)-offsetof(mjvGeom,pos));
    const auto* left=reinterpret_cast<const mjvGLCamera*>(
        reinterpret_cast<const char*>(left_pos)-offsetof(mjvGLCamera,pos));
    const auto* right=reinterpret_cast<const mjvGLCamera*>(
        reinterpret_cast<const char*>(right_pos)-offsetof(mjvGLCamera,pos));
    if (all<=0 || geoms[0].type!=first_type || geoms[0].objid!=first_id ||
        left->frustum_width!=width) return 0;
    // Scene-space clipping for every rendered geometry. Never target a
    // specific label, material, color, collision, arm, or ball object.
    int n=0;
    for (int j=0; j<all; ++j) {
        const mjvGeom &g=geoms[j];
        bool keep=true;
        if (g.objtype==mjOBJ_GEOM && g.objid>=0 && g.objid<model->ngeom &&
            g.type!=mjGEOM_PLANE && g.type!=mjGEOM_HFIELD && g.type<mjNGEOMTYPES) {
            const double r=model->geom_rbound[g.objid];
            if (std::isfinite(r) && r>0.) {
                const double rad=1.11*r+.012;
                keep=plausible(*left,g.pos,rad) || plausible(*right,g.pos,rad);
            }
        }
        if (keep) {
            if (n!=j) geoms[n]=g;
            ++n;
        }
    }
    return all-n;
}
