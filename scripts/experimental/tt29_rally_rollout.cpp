// Isolated, exact force and contact trials. The live rally state is read only.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {
struct ClearanceBounds {
    std::array<mjtNum, 3> center{}, half{};
    bool finite = false;
};

// The compiled mesh AABB is in the geometry's local frame. For ordinary
// primitives derive the bound directly from the current model sizes. Keep
// unbounded or plugin-defined shapes on the original narrow-test path.
ClearanceBounds local_clearance_box(const mjModel* m, int geom) {
    ClearanceBounds b;
    const mjtNum* size = m->geom_size + 3*geom;
    switch (m->geom_type[geom]) {
        case mjGEOM_SPHERE:
            b.half = {size[0],size[0],size[0]};
            break;
        case mjGEOM_CAPSULE:
            b.half = {size[0],size[0],size[0]+size[1]};
            break;
        case mjGEOM_CYLINDER:
            b.half = {size[0],size[0],size[1]};
            break;
        case mjGEOM_ELLIPSOID:
        case mjGEOM_BOX:
            b.half = {size[0],size[1],size[2]};
            break;
        case mjGEOM_MESH: {
            const mjtNum* local = m->geom_aabb + 6*geom;
            b.center = {local[0],local[1],local[2]};
            b.half = {local[3],local[4],local[5]};
            // Mesh vertices are stored as floats. Cover the compiler's
            // float extrema and center/half-extent roundoff as well.
            for (int j=0; j<3; ++j) {
                b.half[j] += 4.*std::numeric_limits<float>::epsilon() *
                             (1. + std::abs(b.center[j]) + b.half[j]);
            }
            break;
        }
        default:
            return b;
    }
    b.finite = true;
    for (int j=0; j<3; ++j) {
        if (!std::isfinite(b.center[j]) || !std::isfinite(b.half[j]) || b.half[j] < 0.) {
            b.finite = false;
        }
    }
    return b;
}

struct WorldBox { std::array<mjtNum, 3> low, high; bool finite = true; };

WorldBox world_clearance_box(const mjData* d, int geom, const ClearanceBounds& local) {
    const mjtNum* pos = d->geom_xpos + 3*geom;
    const mjtNum* R = d->geom_xmat + 9*geom;
    WorldBox bounds;
    for (int k=0; k<3; ++k) {
        const auto* r = R + 3*k;
        const auto& c = local.center;
        const auto& h = local.half;
        const mjtNum at = pos[k] + r[0]*c[0] + r[1]*c[1] + r[2]*c[2];
        const mjtNum radius = std::abs(r[0])*h[0] + std::abs(r[1])*h[1] + std::abs(r[2])*h[2];
        // Expand the broad bound, never shrink it, including roundoff around
        // the exact horizon. Distances that may matter still go to MuJoCo.
        const mjtNum pad = 1.e-9 * (1. + std::abs(at) + radius);
        bounds.low[k] = at - radius - pad;
        bounds.high[k] = at + radius + pad;
        bounds.finite = bounds.finite && std::isfinite(bounds.low[k]) && std::isfinite(bounds.high[k]);
    }
    return bounds;
}

bool boxes_separated(const WorldBox& a, const WorldBox& b, mjtNum horizon) {
    if (!a.finite || !b.finite) return false;
    for (int k=0; k<3; ++k) {
        if (a.low[k] - b.high[k] > horizon || b.low[k] - a.high[k] > horizon) return true;
    }
    return false;
}
}  // namespace

extern "C" void trial_strokes_v5(
    const mjModel* nominal, const mjData* source, const double* goals,
    int count, int frames, int micro, int side,
    const int* active, int active_count, const int* other, int other_count,
    const double* other_home, const double* increments, const double* low, const double* high,
    const int* ids, bool received, double cutoff, double drag, double lift,
    const double* target, double* rates, double* scores,
    const int* first, const int* second, int pairs, double rate_scale, double speed_limit,
    const int* active_dofs, double rim_penalty, double clearance_margin,
    double envelope_no_load, const int* near_actuators, const int* near_dofs, int near_count, double* feedback, double* traces, double* approaches) {
    const int ball = ids[0], blade = ids[1], table = ids[2], table2 = ids[3], net = ids[4];
    const int ball_body = ids[5], pos = ids[6], vel = ids[7];
    const double start = source->time;
    const double mass = nominal->body_mass[ball_body];
    constexpr int width = 13;
    std::fill(scores, scores + width * count, std::numeric_limits<double>::quiet_NaN());

    // Both checks are purely broad phase. Keep order, sampling, thresholds,
    // narrow distances and the original sphere bound for every plausible pair.
    std::vector<ClearanceBounds> local_bounds(pairs ? nominal->ngeom : 0);
    std::vector<int> monitored;
    if (pairs) {
        std::vector<bool> seen(nominal->ngeom);
        for (int k=0; k<pairs; ++k) {
            for (const int g : {first[k],second[k]}) {
                if (seen[g]) continue;
                seen[g] = true;
                local_bounds[g] = local_clearance_box(nominal,g);
                if (local_bounds[g].finite) monitored.push_back(g);
            }
        }
    }

    #pragma omp parallel num_threads(2)
    {
        // Each worker owns the actuator-limit table it changes during a trial.
        mjModel* local = envelope_no_load > 0. ? mj_copyModel(nullptr, nominal) : nullptr;
        const mjModel* model = local ? local : nominal;
        mjData* d = mj_makeData(model);
        std::vector<double> step_control(model->nu, 0.0);
        std::vector<WorldBox> world_bounds(pairs ? model->ngeom : 0);
        #pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < count; ++i) {
            mj_copyData(d, model, source);
            bool bounced = received, landed = false, struck = false, touching_blade = false;
            bool table_contact = false, net_crossed = false, fault = false;
            double old_y = side * d->qpos[pos + 1], old_z = d->qpos[pos + 2];
            double clearance = 0.0, contact_time = 0.0, landing_time = 0.0;
            double land_x = 0.0, land_y = 0.0, best = std::numeric_limits<double>::infinity();
            double x_best = 0.0, y_best = 0.0, z_best = 0.0, v_best = 0.0, v_after = 0.0, v_before = 0.0;
            double contact_radius = 0.0;
            // Optional practice observation, including candidates that failed to land.
            double* observation = feedback ? feedback + 9 * i : nullptr;
            if (observation) {
                std::fill(observation, observation + 9, 0.);
                observation[0] = 2.;  // Closest center separation before touching.
            }
            // Optional output from a hypothetical model. Preserve its nearest pre-contact
            // body configuration, replacing it with the first actual virtual contact if one occurs.
            double* approach = approaches ? approaches + 19 * i : nullptr;
            double best_approach = std::numeric_limits<double>::infinity();
            if (approach) std::fill(approach, approach + 19, std::numeric_limits<double>::quiet_NaN());
            for (int t = 0; t < frames && !fault; ++t) {
                for (int j = 0; j < active_count; ++j) {
                    const int motor = active[j];
                    const size_t k = (i * frames + t) * active_count + j;
                    const double target_ctrl = std::clamp(goals[k], low[motor], high[motor]);
                    const double rate = std::clamp((target_ctrl - d->ctrl[motor]) / (micro * increments[motor]),
                                                   -rate_scale, rate_scale);
                    rates[k] = rate;
                    step_control[motor] = rate * increments[motor];
                }
                if (traces) {
                    // Pre-command state and the six references issued for this tick.
                    double* record = traces + (static_cast<size_t>(i) * frames + t) * 17;
                    std::copy(d->qpos + pos, d->qpos + pos + 3, record);
                    std::copy(d->qvel + vel, d->qvel + vel + 3, record + 3);
                    for (int j = 0; j < 6; ++j) {
                        const int motor = active[j];
                        const int joint = model->actuator_trnid[2 * motor];
                        if (j < 5) record[6 + j] = d->qpos[model->jnt_qposadr[joint]];
                        record[11 + j] = std::clamp(d->ctrl[motor] + micro * step_control[motor],
                                                    low[motor], high[motor]);
                    }
                }
                for (int j = 0; j < other_count; ++j) {
                    const int motor = other[j];
                    step_control[motor] = std::clamp((other_home[j] - d->ctrl[motor]) / micro,
                                                    -.8 * increments[motor], .8 * increments[motor]);
                }
                for (int tick = 0; tick < micro; ++tick) {
                    for (int j = 0; j < model->nu; ++j)
                        d->ctrl[j] = std::clamp(d->ctrl[j] + step_control[j], low[j], high[j]);

                    const double* velocity = d->qvel + vel;
                    double spin[3], cross[3];
                    mju_rotVecQuat(spin, velocity + 3, d->qpos + pos + 3);
                    mju_cross(cross, spin, velocity);
                    const double speed = mju_norm3(velocity);
                    const double incoming = std::hypot(velocity[0], velocity[1]);
                    double* applied = d->xfrc_applied + 6 * ball_body;
                    for (int j = 0; j < 3; ++j) {
                        applied[j] = mass * (-drag * speed * velocity[j] + lift * cross[j]);
                        applied[j + 3] = -.18 * model->body_inertia[3 * ball_body + j] * spin[j];
                    }
                    if (local) {
                        for (int j = 0; j < near_count; ++j) {
                            const int cap = 2 * near_actuators[j];
                            const double rate = d->qvel[near_dofs[j]] / envelope_no_load;
                            local->actuator_forcerange[cap] = nominal->actuator_forcerange[cap]
                                                            * std::clamp(1. + rate, 0., 1.);
                            local->actuator_forcerange[cap + 1] = nominal->actuator_forcerange[cap + 1]
                                                                * std::clamp(1. - rate, 0., 1.);
                        }
                    }
                    mj_step(model, d);
                    if (speed_limit > 0.0) {
                        for (int j = 0; j < active_count; ++j) {
                            if (std::abs(d->qvel[active_dofs[j]]) > speed_limit) {
                                fault = true;
                                landed = false;
                                break;
                            }
                        }
                    }
                    if (fault) break;
                    if (pairs && (clearance_margin > 0.0 || tick % 4 == 0)) {
                        const double threshold = clearance_margin > 0.0 ? clearance_margin : -.003;
                        const double horizon = std::max(.001, clearance_margin + .0001);
                        if (clearance_margin > 0.0)
                            mj_kinematics(model, d);
                        for (const int g : monitored) {
                            world_bounds[g] = world_clearance_box(d,g,local_bounds[g]);
                        }
                        for (int k = 0; k < pairs; ++k) {
                            const int first_geom = first[k], second_geom = second[k];
                            if (local_bounds[first_geom].finite && local_bounds[second_geom].finite &&
                                boxes_separated(world_bounds[first_geom],world_bounds[second_geom],horizon)) continue;
                            const double radius = model->geom_rbound[first_geom] + model->geom_rbound[second_geom]
                                                + clearance_margin;
                            const double* p1 = d->geom_xpos + 3 * first_geom;
                            const double* p2 = d->geom_xpos + 3 * second_geom;
                            double delta[3];
                            mju_sub3(delta, p1, p2);
                            if (mju_dot3(delta, delta) > radius * radius) continue;
                            if (mj_geomDistance(model, d, first_geom, second_geom, horizon, nullptr) < threshold) {
                                fault = true;
                                landed = false;
                                break;
                            }
                        }
                    }
                    if (fault) break;
                    const double x = d->qpos[pos], y = side * d->qpos[pos + 1], z = d->qpos[pos + 2];
                    if (observation && !struck) {
                        double offset[3];
                        mju_sub3(offset, d->qpos + pos, d->geom_xpos + 3 * blade);
                        observation[0] = std::min(observation[0], mju_norm3(offset));
                    }
                    const mjContact* blade_contact = nullptr;
                    bool hit_table = false;
                    for (int c = 0; c < d->ncon; ++c) {
                        const mjContact& touch = d->contact[c];
                        if (touch.efc_address < 0 || (touch.geom[0] != ball && touch.geom[1] != ball))
                            continue;
                        const int geom = touch.geom[touch.geom[0] == ball ? 1 : 0];
                        if (geom == blade) blade_contact = &touch;
                        else if ((geom == table || geom == table2) && std::abs(touch.frame[2]) > .55) hit_table = true;
                        else if (landed && geom != net) { fault = true; break; }
                        else { fault = true; landed = false; break; }
                    }
                    if (fault) break;
                    const bool hit = blade_contact != nullptr;
                    if (approach && !struck && bounced) {
                        const double* center = d->geom_xpos + 3 * blade;
                        const double* rotation = d->geom_xmat + 9 * blade;
                        double gap[3];
                        mju_sub3(gap, d->qpos + pos, center);
                        const double squared = mju_dot3(gap, gap);
                        if (squared < best_approach || hit) {
                            best_approach = squared;
                            approach[0] = d->time - start;
                            std::copy(d->qpos + pos, d->qpos + pos + 3, approach + 1);
                            std::copy(center, center + 3, approach + 4);
                            for (int axis = 0; axis < 3; ++axis) approach[7 + axis] = rotation[3 * axis + 2];
                            for (int axis = 0; axis < 5; ++axis) {
                                const int joint = model->actuator_trnid[2 * active[axis]];
                                approach[10 + axis] = d->qpos[model->jnt_qposadr[joint]];
                            }
                            std::copy(d->qvel + vel, d->qvel + vel + 3, approach + 15);
                            approach[18] = hit ? 1. : 0.;
                        }
                    }
                    if (hit && !touching_blade) {
                        if (!bounced || struck || y > -.1) {
                            fault = true;
                            break;
                        }
                        struck = true;
                        contact_time = d->time - start;
                        v_before = incoming;
                        double radial[3];
                        mju_sub3(radial, blade_contact->pos, d->geom_xpos + 3 * blade);
                        const double* axes = d->geom_xmat + 9 * blade;
                        const double height = radial[0] * axes[2] + radial[1] * axes[5] + radial[2] * axes[8];
                        contact_radius = std::sqrt(std::max(0., mju_dot3(radial, radial) - height * height));
                    }
                    if (!hit && touching_blade) {
                        v_after = std::hypot(velocity[0], velocity[1]);
                        if (observation) {
                            observation[1] = 1.;
                            std::copy(velocity, velocity + 3, observation + 2);
                            std::copy(d->qpos + pos, d->qpos + pos + 3, observation + 5);
                            observation[8] = contact_radius;
                        }
                    }
                    touching_blade = hit;
                    if (struck && old_y <= 0. && y > 0.) {
                        net_crossed = true;
                        clearance = (old_z + (z - old_z) * -old_y / (y - old_y)) - .9325;
                    }
                    if (hit_table && !table_contact) {
                        if (!struck) {
                            if (y >= 0. || bounced) {
                                fault = true;
                                break;
                            }
                            bounced = true;
                        } else {
                            if (landed) { fault = true; break; }
                            if (y <= 0. || !net_crossed) {
                                fault = true;
                                break;
                            }
                            landed = true;
                            landing_time = d->time - start;
                            land_x = x;
                            land_y = y;
                        }
                    }
                    table_contact = hit_table;
                    if (!struck && (d->time - start > cutoff || y < -1.82)) {
                        fault = true;
                        break;
                    }
                    if (landed && !hit_table && z > .806) {
                        const double forward = side * velocity[1];
                        const double cost = 5.4 * std::abs(x - target[0]) + 6.0 * std::abs(y - target[1])
                                          + 6.2 * std::abs(z - target[2]) + .70 * std::abs(forward - target[3]);
                        if (cost < best) {
                            best = cost;
                            x_best = x;
                            y_best = y;
                            z_best = z;
                            v_best = forward;
                        }
                        if (y > target[1] + .28) {
                            fault = true;
                            break;
                        }
                    }
                    if (z < .70 || std::abs(x) > 1.4 || (struck && y > 1.9)) {
                        fault = true;
                        break;
                    }
                    old_y = y;
                    old_z = z;
                }
            }
            if (landed && net_crossed && std::isfinite(best) && clearance >= 0.0) {
                const double score = best + 9. * std::max(0., .035 - clearance)
                    + 8. * std::max(0., land_y - target[1] + .065) + 6. * std::max(0., land_y - 1.25)
                    + rim_penalty * std::max(0., contact_radius - .032);
                double* row = scores + width * i;
                row[0] = score; row[1] = contact_time; row[2] = landing_time;
                row[3] = land_x; row[4] = side * land_y; row[5] = clearance;
                row[6] = x_best; row[7] = y_best; row[8] = z_best; row[9] = v_best;
                row[10] = v_before; row[11] = v_after; row[12] = contact_radius;
            }
        }
        mj_deleteData(d);
        if (local) mj_deleteModel(local);
    }
}

extern "C" void trial_strokes_v4(
    const mjModel* nominal, const mjData* source, const double* goals,
    int count, int frames, int micro, int side,
    const int* active, int active_count, const int* other, int other_count,
    const double* other_home, const double* increments, const double* low, const double* high,
    const int* ids, bool received, double cutoff, double drag, double lift,
    const double* target, double* rates, double* scores,
    const int* first, const int* second, int pairs, double rate_scale, double speed_limit,
    const int* active_dofs, double rim_penalty, double clearance_margin,
    double envelope_no_load, const int* near_actuators, const int* near_dofs, int near_count, double* feedback, double* traces) {
    trial_strokes_v5(
        nominal, source, goals, count, frames, micro, side, active,
        active_count, other, other_count, other_home, increments, low, high, ids,
        received, cutoff, drag, lift, target, rates, scores, first,
        second, pairs, rate_scale, speed_limit, active_dofs, rim_penalty, clearance_margin, envelope_no_load,
        near_actuators, near_dofs, near_count, feedback, traces, nullptr);
}
