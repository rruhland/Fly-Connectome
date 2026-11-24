// Isolated, exact force and contact trials. The live rally state is read only.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

extern "C" void trial_strokes(
    const mjModel* model, const mjData* source, const double* goals,
    int count, int frames, int micro, int side,
    const int* active, int active_count, const int* other, int other_count,
    const double* other_home, const double* increments, const double* low, const double* high,
    const int* ids, bool received, double cutoff, double drag, double lift,
    const double* target, double* rates, double* scores,
    const int* first, const int* second, int pairs, double rate_scale, double speed_limit,
    const int* active_dofs, double rim_penalty) {
    const int ball = ids[0], blade = ids[1], table = ids[2], table2 = ids[3], net = ids[4];
    const int ball_body = ids[5], pos = ids[6], vel = ids[7];
    const double start = source->time;
    const double mass = model->body_mass[ball_body];
    constexpr int width = 13;
    std::fill(scores, scores + width * count, std::numeric_limits<double>::quiet_NaN());

    #pragma omp parallel num_threads(2)
    {
        mjData* d = mj_makeData(model);
        std::vector<double> step_control(model->nu, 0.0);
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
                    if (pairs && tick % 4 == 0) {
                        for (int k = 0; k < pairs; ++k) {
                            const int first_geom = first[k], second_geom = second[k];
                            const double radius = model->geom_rbound[first_geom] + model->geom_rbound[second_geom];
                            const double* p1 = d->geom_xpos + 3 * first_geom;
                            const double* p2 = d->geom_xpos + 3 * second_geom;
                            double delta[3];
                            mju_sub3(delta, p1, p2);
                            if (mju_dot3(delta, delta) > radius * radius) continue;
                            if (mj_geomDistance(model, d, first_geom, second_geom, .001, nullptr) < -.003) {
                                fault = true;
                                landed = false;
                                break;
                            }
                        }
                    }
                    if (fault) break;
                    const double x = d->qpos[pos], y = side * d->qpos[pos + 1], z = d->qpos[pos + 2];
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
                    if (!hit && touching_blade)
                        v_after = std::hypot(velocity[0], velocity[1]);
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
    }
}
