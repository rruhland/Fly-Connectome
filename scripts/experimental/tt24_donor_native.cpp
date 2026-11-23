// Closed-loop reward rollouts for the installed physical BounceArena fixture.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>

namespace {
constexpr int kJoints = 5;
constexpr int kFeatures = 20;
constexpr int kSubsteps = 8;
constexpr double kRate = 5.2;

void features(const double* state, double* x) {
    const auto clip = [](double v) { return std::clamp(v, -3., 3.); };
    const double bx = clip((state[0] - state[6]) / .065);
    const double by = clip((state[1] - state[7]) / .065);
    const double bz = clip((state[2] - state[8] - .13) / .13);
    const double vx = clip(state[3] / .38);
    const double vy = clip(state[4] / .38);
    const double vz = clip(state[5] / 1.50);
    x[0] = 1.;
    x[1] = bx; x[2] = by; x[3] = bz;
    x[4] = vx; x[5] = vy; x[6] = vz;
    x[7] = clip(state[6] / .04);
    x[8] = clip(state[7] / .04);
    x[9] = clip(state[8] / .04);
    x[10] = clip(state[9] / .14);
    x[11] = clip(state[10] / .14);
    x[12] = clip(bz * vz);
    x[13] = clip(vz * vz - .5);
    x[14] = clip(bz * bz - .5);
    x[15] = clip(bx * vz);
    x[16] = clip(by * vz);
    x[17] = clip(vx * vz);
    x[18] = clip(vy * vz);
    x[19] = std::tanh(2. * vz);
}

void command(const double* state, const double* weights, double* issued, double* output) {
    double x[kFeatures];
    features(state, x);
    for (int joint = 0; joint < kJoints; ++joint) {
        double value = 0.;
        for (int k = 0; k < kFeatures; ++k) value += x[k] * weights[joint * kFeatures + k];
        const double target = .245 * std::tanh(value);
        output[joint] = std::clamp((target - issued[joint]) / (kRate / 60.), -.65, .65);
        issued[joint] += output[joint] * (kRate / 60.);
    }
}

struct Judge {
    int contacts = 0, hits = 0, consecutive = 0, longest = 0, last_scored_impact = 0;
    double impact_t = 0., impact_z = 0., launch_t = 0., apex = 0.;
    bool touching = false, candidate = false, flying = false;

    bool sample(const mjData* d, int ball, int blade, int position, int velocity, double incoming) {
        const mjContact* contact = nullptr;
        int sign = 0;
        for (int i = 0; i < d->ncon; ++i) {
            const mjContact& item = d->contact[i];
            if (item.efc_address < 0) continue;
            if (item.geom[0] == ball) {
                if (item.geom[1] != blade) return true;
                contact = &item;
                sign = -1;
            } else if (item.geom[1] == ball) {
                if (item.geom[0] != blade) return true;
                contact = &item;
                sign = 1;
            }
        }
        const double* q = d->qpos + position;
        const double upward = d->qvel[velocity + 2];
        const double t = d->time;
        if (contact) {
            flying = false;
            if (!touching) {
                ++contacts;
                const double* matrix = d->geom_xmat + 9 * blade;
                const double* n = contact->frame;
                const double alignment = std::abs(n[0] * matrix[2] + n[1] * matrix[5] + n[2] * matrix[8]);
                if (sign * n[2] > .55 && alignment > .95 && incoming < -.25) {
                    candidate = true;
                    impact_t = t;
                    impact_z = q[2];
                }
            }
            touching = true;
        } else if (touching) {
            touching = false;
            if (candidate && t - impact_t <= .032 && upward > .25) {
                flying = true;
                launch_t = t;
                apex = q[2];
            }
            candidate = false;
        }
        if (q[2] < .51 || q[2] > 2.1 || std::abs(q[0]) > .66 || q[1] <= -1.65 || q[1] >= -.33) return true;
        if (flying) {
            apex = std::max(apex, q[2]);
            if (upward <= 0. && t - launch_t >= .065) {
                flying = false;
                if (apex - impact_z >= .045) {
                    ++hits;
                    consecutive = contacts == last_scored_impact + 1 ? consecutive + 1 : 1;
                    last_scored_impact = contacts;
                    longest = std::max(longest, consecutive);
                }
            }
        }
        return false;
    }
};

struct Donor {
    mjModel* model;
    mjData* data;
    int ball, blade, body, position, velocity, state_size;

    explicit Donor(mjModel* m) : model(m), data(mj_makeData(m)) {
        ball = mj_name2id(m, mjOBJ_GEOM, "ball");
        blade = mj_name2id(m, mjOBJ_GEOM, "learner_racket");
        body = mj_name2id(m, mjOBJ_BODY, "ball");
        const int free_joint = mj_name2id(m, mjOBJ_JOINT, "ball_free");
        position = m->jnt_qposadr[free_joint];
        velocity = m->jnt_dofadr[free_joint];
        state_size = mj_stateSize(m, mjSTATE_INTEGRATION);
    }

    ~Donor() { mj_deleteData(data); }

    void play(const double* weights, const double* initial, double drag, double damping, int cap, double* result,
              const double* transient=nullptr) {
        mj_resetData(model, data);
        model->geom_solref[blade * mjNREF + 1] = -damping;
        mj_setState(model, data, initial, mjSTATE_INTEGRATION);
        mj_forward(model, data);
        const double* blade_pos = data->geom_xpos + 3 * blade;
        const double home[] = {blade_pos[0], blade_pos[1], blade_pos[2]};
        double issued[kJoints] = {}, state[11], action[kJoints];
        Judge judge;
        int tick = 0;
        bool done = false;
        for (; tick < cap && !done; ++tick) {
            mj_kinematics(model, data);
            for (int a = 0; a < 3; ++a) {
                state[a] = data->qpos[position + a] - home[a];
                state[a + 3] = data->qvel[velocity + a];
                state[a + 6] = blade_pos[a] - home[a];
            }
            const double* normal = data->geom_xmat + 9 * blade;
            const int side = normal[8] >= 0. ? 1 : -1;
            state[9] = side * normal[2];
            state[10] = side * normal[5];
            if (transient && tick < 120) {
                // The extra readout decays continuously after reset and uses the same live observations.
                const double gain = std::exp(-tick / 16.);
                double mixed[kJoints * kFeatures];
                for (int i = 0; i < kJoints * kFeatures; ++i) mixed[i] = weights[i] + gain * transient[i];
                command(state, mixed, issued, action);
            } else {
                command(state, weights, issued, action);
            }
            for (int sub = 0; sub < kSubsteps; ++sub) {
                for (int a = 0; a < kJoints; ++a) {
                    data->ctrl[a] = std::clamp(data->ctrl[a] + action[a] * (kRate * model->opt.timestep),
                                              model->actuator_ctrlrange[2 * a], model->actuator_ctrlrange[2 * a + 1]);
                }
                const double* v = data->qvel + velocity;
                const double incoming = v[2];
                const double factor = drag * std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
                for (int a = 0; a < 3; ++a) data->xfrc_applied[6 * body + a] = factor * v[a];
                mj_step(model, data);
                done = judge.sample(data, ball, blade, position, velocity, incoming);
                if (done) break;
            }
        }
        result[0] = judge.hits;
        result[1] = judge.longest;
        result[2] = tick;
        result[3] = judge.contacts;
    }
};
}  // namespace

extern "C" void* tt24_donor_open(mjModel* model) { return new Donor(model); }
extern "C" void tt24_donor_close(void* handle) { delete static_cast<Donor*>(handle); }

extern "C" void tt24_donor_evaluate(void* handle, const double* weights, int population,
                                    const double* states, const double* settings, int starts,
                                    int cap, double* outputs) {
    auto* donor = static_cast<Donor*>(handle);
    for (int policy = 0; policy < population; ++policy) {
        for (int start = 0; start < starts; ++start) {
            donor->play(weights + policy * kJoints * kFeatures, states + start * donor->state_size,
                        settings[2 * start], settings[2 * start + 1], cap, outputs + 4 * (policy * starts + start));
        }
    }
}

extern "C" void tt24_donor_command(const double* state, const double* weights, double* issued, double* output) {
    command(state, weights, issued, output);
}

extern "C" void tt24_donor_transient(void* handle, const double* weights, const double* residuals,
                                     int policies, const double* states, const double* physics,
                                     int starts, int cap, double* outputs) {
    auto* donor = static_cast<Donor*>(handle);
    for (int i = 0; i < policies; ++i) {
        for (int s = 0; s < starts; ++s) {
            donor->play(weights, states+s*donor->state_size, physics[2*s], physics[2*s+1], cap,
                        outputs+4*(i*starts+s), residuals+i*kJoints*kFeatures);
        }
    }
}
