// Move isolated MuJoCo flight stepping and quaternion batches across the Python boundary once.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>

extern "C" void normal_quaternions(const double* normals, double* output, int count) {
    for (int i = 0; i < count; ++i) mju_quatZ2Vec(output + 4*i, normals + 3*i);
}

extern "C" void departure_spins(const double* states, double* output, int count, int columns) {
    for (int i = 0; i < count; ++i) {
        const double* state = states + i*columns;
        mju_rotVecQuat(output + 3*i, state + 18, state + 4);
    }
}

bool tabletop(const mjContact& contact, const int* tables, int count) {
    if (std::abs(contact.frame[2]) <= .55) return false;
    for (int i = 0; i < count; ++i) {
        if (contact.geom[0] == tables[i] || contact.geom[1] == tables[i]) return true;
    }
    return false;
}

extern "C" int flight_window(
    const mjModel* model, mjData* data, const double* position, const double* velocity,
    const double* spin, const int* tables, int count, int net, int steps, int side,
    bool bounced, bool outgoing, double min_y, double surface_height, double ball_top,
    double mass, double inertia, double decay, double drag, double lift, double* output) {
    mj_resetData(model, data);
    std::copy_n(position, 3, data->qpos);
    std::copy_n(velocity, 3, data->qvel);
    std::copy_n(spin, 3, data->qvel + 3);
    mj_forward(model, data);
    bool touching = false;
    int rows = 0;
    const int stride = outgoing ? 7 : 10;
    for (int step = 0; step < steps; ++step) {
        const double* p = data->qpos;
        const double* v = data->qvel;
        double w[3], cross[3];
        mju_rotVecQuat(w, v + 3, p + 3);
        mju_cross(cross, w, v);
        const double speed = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        double* force = data->xfrc_applied + 6;
        for (int axis = 0; axis < 3; ++axis) {
            force[axis] = outgoing ? (-mass * drag * speed)*v[axis] + (mass * lift)*cross[axis]
                                   : mass * ((-drag * speed)*v[axis] + lift*cross[axis]);
            force[axis + 3] = (-inertia * decay)*w[axis];
        }
        mj_step(model, data);
        bool on_table = false;
        for (int index = 0; index < data->ncon; ++index) {
            const mjContact& contact = data->contact[index];
            if (contact.efc_address < 0) continue;
            if (outgoing && (contact.geom[0] == net || contact.geom[1] == net)) return 0;
            on_table |= tabletop(contact, tables, count);
        }
        if (on_table && !touching) {
            if (bounced) break;
            if (outgoing && p[1] >= -.12) return 0;
            if (!outgoing && side*p[1] >= 0) break;
            bounced = true;
        }
        touching = on_table;
        if (p[2] < surface_height - .04) break;
        if (outgoing && (p[1] < min_y || v[1] > -.25)) break;
        if (!outgoing && (side*v[1] > -.2 || std::abs(p[1]) > 2)) break;
        if (!bounced || on_table) continue;
        if (outgoing && (p[1] >= -.65 || p[2] <= ball_top + .035)) continue;
        if (!outgoing && p[2] <= ball_top + .008) continue;
        double* row = output + rows++ * stride;
        row[0] = data->time;
        std::copy_n(p, 3, row + 1);
        std::copy_n(v, 3, row + 4);
        if (!outgoing) std::copy_n(w, 3, row + 7);
    }
    return rows;
}
