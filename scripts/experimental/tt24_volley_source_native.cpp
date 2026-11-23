// A physical receive-and-route game with exactly one judged paddle encounter.
#include "tt24_donor_native.cpp"

namespace {
constexpr int kVolleyStats = 13;
#ifdef TT24_CATCHABLE
constexpr double kTargetZ = .118;
constexpr double kTargetX = .057;
constexpr double kHalfX = .043;
#else
constexpr double kTargetZ = .165;
constexpr double kTargetX = .080;
constexpr double kHalfX = .055;
#endif
constexpr double kTargetY = 0.;
constexpr double kHalfY = .070;
constexpr double kDirection = .045;

void volley(Donor* donor, const double* weights, const double* initial, double drag, double damping,
            int cap, double* output) {
    auto* m = donor->model;
    auto* d = donor->data;
    const int p = donor->position, v = donor->velocity;
    const int ball = donor->ball, blade = donor->blade;
    mj_resetData(m, d);
    m->geom_solref[mjNREF * blade + 1] = -damping;
    mj_setState(m, d, initial, mjSTATE_INTEGRATION);
    mj_forward(m, d);
    const double* blade_pos = d->geom_xpos + 3 * blade;
    const double home[] = {blade_pos[0], blade_pos[1], blade_pos[2]};
    const double plane = home[2] + kTargetZ;
    const double receive_side = d->qpos[p] < home[0] ? -1. : 1.;
    double state[11], issued[kJoints] = {}, action[kJoints] = {};
    double hit_time = 0., paddle_speed = 0., best_height = 0.;
    double finish_x = 0., finish_y = 0., finish_vx = 0.;
#ifdef TT24_CATCHABLE
    double crossing_vz = 0.;
#endif
    bool done = false, contact = false, sent = false, crossing = false, legal = false;
    int decisions = 0, frames = 0, encounters = 0, exit_code = 0;
    for (; frames < cap && !done; ++frames) {
        if (!contact && !sent) {
            mj_kinematics(m, d);
            for (int i = 0; i < 3; ++i) {
                state[i] = d->qpos[p+i] - home[i];
                state[i+3] = d->qvel[v+i];
                state[i+6] = blade_pos[i] - home[i];
            }
            const double* normal = d->geom_xmat + 9 * blade;
            const double sign = normal[8] >= 0. ? 1. : -1.;
            state[9] = sign * normal[2];
            state[10] = sign * normal[5];
            command(state, weights, issued, action);
            ++decisions;
        } else {
            std::fill(action, action + kJoints, 0.);
        }
        for (int sub = 0; sub < kSubsteps; ++sub) {
            for (int i = 0; i < kJoints; ++i) {
                d->ctrl[i] = std::clamp(d->ctrl[i] + action[i] * (kRate * m->opt.timestep),
                                        m->actuator_ctrlrange[2*i], m->actuator_ctrlrange[2*i+1]);
            }
            const double* q = d->qpos + p;
            const double previous[] = {q[0], q[1], q[2]};
            const double* velocity = d->qvel + v;
            const double incoming = velocity[2];
            const double factor = drag * std::sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1] +
                                                   velocity[2]*velocity[2]);
            for (int i = 0; i < 3; ++i) d->xfrc_applied[6*donor->body+i] = factor * velocity[i];
            mj_step(m, d);

            const mjContact* hit = nullptr;
            int sign = 0;
            for (int i = 0; i < d->ncon; ++i) {
                const mjContact& entry = d->contact[i];
                if (entry.efc_address < 0) continue;
                const bool first = entry.geom[0] == ball, second = entry.geom[1] == ball;
                if (!first && !second) continue;
                if (entry.geom[first ? 1 : 0] != blade) {
                    done = true;
                    exit_code = 1;
                    break;
                }
                hit = &entry;
                sign = first ? -1 : 1;
            }
            if (done) break;
            if (hit && sent) {
                // A return that touches again before its apex is a failed route.
                // No second encounter is passed to the pupil or rewarded.
                done = true;
                exit_code = 2;
                legal = false;
                break;
            }
            if (hit && !contact) {
                ++encounters;
                const double* rot = d->geom_xmat + 9 * blade;
                const double alignment = std::abs(hit->frame[0]*rot[2] + hit->frame[1]*rot[5] + hit->frame[2]*rot[8]);
                if (sign * hit->frame[2] <= .55 || alignment <= .95 || incoming >= -.25) {
                    done = true;
                    exit_code = 3;
                    break;
                }
                contact = true;
                hit_time = d->time;
                double twist[6];
                mj_objectVelocity(m, d, mjOBJ_GEOM, blade, twist, 0);
                paddle_speed = twist[5] + twist[0]*(hit->pos[1]-blade_pos[1]) -
                                          twist[1]*(hit->pos[0]-blade_pos[0]);
            }
            if (contact && !hit) {
                contact = false;
                if (d->time - hit_time > .032 || velocity[2] <= .25) {
                    done = true;
                    exit_code = 4;
                    break;
                }
                sent = true;
                legal = true;
            }
            if (sent) {
                finish_x = q[0] - home[0];
                finish_y = q[1] - home[1];
                finish_vx = velocity[0];
                best_height = std::max(best_height, q[2] - home[2]);
                if (previous[2] < plane && q[2] >= plane && velocity[2] > 0.) {
                    const double t = (plane - previous[2]) / (q[2] - previous[2]);
                    finish_x = previous[0] + t * (q[0] - previous[0]) - home[0];
                    finish_y = previous[1] + t * (q[1] - previous[1]) - home[1];
#ifdef TT24_CATCHABLE
                    crossing_vz = velocity[2];
#endif
                    crossing = true;
                    done = true;
                    exit_code = 5;
                    break;
                }
                if (velocity[2] <= 0.) {
                    done = true;
                    exit_code = 6;
                    break;
                }
            }
            if (q[2] < .51 || q[2] > 2.1 || std::abs(q[0]) > .66 || q[1] <= -1.65 || q[1] >= -.33) {
                done = true;
                exit_code = 7;
                break;
            }
        }
    }
#ifdef TT24_TWO_RECEIVERS
    const double route_side = finish_x < 0. ? -1. : 1.;
#else
    const double route_side = receive_side;
#endif
    bool gate = legal && crossing && std::abs(route_side * finish_x - kTargetX) <= kHalfX &&
                std::abs(finish_y - kTargetY) <= kHalfY && route_side * finish_vx >= kDirection;
#ifdef TT24_CATCHABLE
#ifdef TT24_TWO_RECEIVERS
    gate = gate && crossing_vz >= .16 && crossing_vz <= 1.45;
#else
    gate = gate && crossing_vz >= .16 && crossing_vz <= 1.18;
#endif
#endif
    const bool powered = gate && paddle_speed > .055;
    const double dx = (route_side * finish_x - kTargetX) / kHalfX, dy = (finish_y - kTargetY) / kHalfY;
    const double aiming = std::exp(-.35*(dx*dx+dy*dy));
    const double lift = std::clamp(best_height/kTargetZ, 0., 1.);
    const double toward = std::clamp((route_side * finish_vx + .05) / .22, 0., 1.);
    const double power = std::clamp(paddle_speed / .22, 0., 1.);
    double speed_quality = 1.;
#ifdef TT24_CATCHABLE
    if (crossing) speed_quality = std::clamp(1. - std::abs(crossing_vz - .7) / 1.30, 0., 1.);
#endif
    output[0] = legal ? .10 + .33*lift + .45*aiming*lift*speed_quality + .24*toward*lift*speed_quality +
                               .27*power + double(gate) + .5*double(powered) : 0.;
    output[1] = gate;
    output[2] = powered;
    output[3] = decisions;
    output[4] = frames;
    output[5] = encounters;
    output[6] = exit_code;
    output[7] = best_height;
    output[8] = finish_vx;
    output[9] = paddle_speed;
    output[10] = legal;
    output[11] = finish_x;
    output[12] = finish_y;
}
}  // namespace

extern "C" void tt24_volley_evaluate(void* handle, const double* weights, int population,
                                     const double* states, const double* settings, int starts,
                                     int cap, double* outputs) {
    auto* donor = static_cast<Donor*>(handle);
    for (int policy = 0; policy < population; ++policy) {
        for (int start = 0; start < starts; ++start) {
            volley(donor, weights + policy*kJoints*kFeatures, states + start*donor->state_size,
                   settings[2*start], settings[2*start+1], cap, outputs + kVolleyStats*(policy*starts+start));
        }
    }
}
