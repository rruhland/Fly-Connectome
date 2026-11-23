// Actual MuJoCo donor rollouts with the orientation at reset in the feedback state.
#define tt24_donor_open tt24_original_open
#define tt24_donor_close tt24_original_close
#define tt24_donor_evaluate tt24_original_evaluate
#define tt24_donor_command tt24_original_command
#include "tt24_donor_native.cpp"
#undef tt24_donor_open
#undef tt24_donor_close
#undef tt24_donor_evaluate
#undef tt24_donor_command

namespace {
constexpr int kTiltedFeatures = kFeatures + 2;

void tilt_command(const double* state, const double* weights, double* issued, double* action) {
    double x[kTiltedFeatures];
    features(state, x);
    x[kFeatures] = std::clamp(state[11] / .14, -3., 3.);
    x[kFeatures + 1] = std::clamp(state[12] / .14, -3., 3.);
    for (int joint = 0; joint < kJoints; ++joint) {
        double value = 0.;
        for (int k = 0; k < kTiltedFeatures; ++k) value += x[k] * weights[joint * kTiltedFeatures + k];
        const double target = .245 * std::tanh(value);
        action[joint] = std::clamp((target - issued[joint]) / (kRate / 60.), -.65, .65);
        issued[joint] += action[joint] * (kRate / 60.);
    }
}

struct TiltedDonor : Donor {
    explicit TiltedDonor(mjModel* model) : Donor(model) {}

    void rollout(const double* weights, const double* initial, double drag, double damping, int cap, double* score) {
        mj_resetData(model, data);
        model->geom_solref[blade * mjNREF + 1] = -damping;
        mj_setState(model, data, initial, mjSTATE_INTEGRATION);
        mj_forward(model, data);
        const double* pad = data->geom_xpos + 3 * blade;
        const double* orientation = data->geom_xmat + 9 * blade;
        const double home[] = {pad[0], pad[1], pad[2]};
        const int initial_side = orientation[8] >= 0. ? 1 : -1;
        double state[13] = {};
        state[11] = initial_side * orientation[2];
        state[12] = initial_side * orientation[5];
        double issued[kJoints] = {}, action[kJoints];
        Judge judge;
        int tick = 0;
        bool lost = false;
        for (; tick < cap && !lost; ++tick) {
            mj_kinematics(model, data);
            for (int axis = 0; axis < 3; ++axis) {
                state[axis] = data->qpos[position + axis] - home[axis];
                state[axis + 3] = data->qvel[velocity + axis];
                state[axis + 6] = pad[axis] - home[axis];
            }
            const int side = orientation[8] >= 0. ? 1 : -1;
            state[9] = side * orientation[2];
            state[10] = side * orientation[5];
            tilt_command(state, weights, issued, action);
            for (int sub = 0; sub < kSubsteps; ++sub) {
                for (int axis = 0; axis < kJoints; ++axis) {
                    data->ctrl[axis] = std::clamp(data->ctrl[axis] + action[axis] * (kRate * model->opt.timestep),
                                                model->actuator_ctrlrange[2 * axis], model->actuator_ctrlrange[2 * axis + 1]);
                }
                const double* v = data->qvel + velocity;
                const double incoming = v[2];
                const double factor = drag * std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
                for (int axis = 0; axis < 3; ++axis) data->xfrc_applied[6 * body + axis] = factor * v[axis];
                mj_step(model, data);
                lost = judge.sample(data, ball, blade, position, velocity, incoming);
                if (lost) break;
            }
        }
        score[0] = judge.hits;
        score[1] = judge.longest;
        score[2] = tick;
        score[3] = judge.contacts;
    }
};
}  // namespace

extern "C" void* tt24_donor_open(mjModel* model) { return new TiltedDonor(model); }
extern "C" void tt24_donor_close(void* handle) { delete static_cast<TiltedDonor*>(handle); }

extern "C" void tt24_donor_evaluate(void* handle, const double* weights, int population,
                                    const double* states, const double* settings, int starts,
                                    int cap, double* output) {
    auto* donor = static_cast<TiltedDonor*>(handle);
    for (int actor = 0; actor < population; ++actor) {
        for (int seed = 0; seed < starts; ++seed) {
            donor->rollout(weights + actor * kJoints * kTiltedFeatures, states + seed * donor->state_size,
                           settings[2 * seed], settings[2 * seed + 1], cap, output + 4 * (actor * starts + seed));
        }
    }
}

extern "C" void tt24_donor_command(const double* state, const double* weights, double* issued, double* action) {
    tilt_command(state, weights, issued, action);
}
