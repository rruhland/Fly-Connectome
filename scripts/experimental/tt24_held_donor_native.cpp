// Native physical held-ball donor episodes using the existing referee and full feedback head.
#define tt24_donor_open tt24_plain_open
#define tt24_donor_close tt24_plain_close
#define tt24_donor_evaluate tt24_plain_evaluate
#include "tt24_donor_native.cpp"
#undef tt24_donor_open
#undef tt24_donor_close
#undef tt24_donor_evaluate

namespace {
constexpr int kSettings = 26;

struct HeldDonor : Donor {
    double gravity[3];

    explicit HeldDonor(mjModel* model) : Donor(model) {
        std::copy(model->opt.gravity, model->opt.gravity + 3, gravity);
    }

    void rollout(const double* weights, const double* initial, const double* plan, int cap, double* score) {
        const int lead = static_cast<int>(plan[2]);
        auto set_gravity = [&](double ratio) {
            for (int a = 0; a < 3; ++a) model->opt.gravity[a] = gravity[a] * ratio;
        };
        set_gravity(plan[17]);
        mj_resetData(model, data);
        model->geom_solref[blade * mjNREF + 1] = -plan[1];
        mj_setState(model, data, initial, mjSTATE_INTEGRATION);
        mj_forward(model, data);
        auto* q = data->qpos + position;
        auto* v = data->qvel + velocity;
        const double* pad = data->geom_xpos + 3 * blade;
        const double* normal = data->geom_xmat + 9 * blade;
        const double location[] = {q[0], q[1], q[2]};
        double issued[kJoints], state[11], action[kJoints], observed_velocity[3];
        std::copy(plan + 12, plan + 17, issued);
        std::copy(v, v + 3, observed_velocity);
        Judge judge;
        int ticks = 0;
        bool lost = false;
        for (int t = 0; t < lead + cap && !lost; ++t) {
            const int released = t - lead;
            if (released == static_cast<int>(plan[18])) set_gravity(plan[19]);
            if (released == static_cast<int>(plan[20])) set_gravity(plan[21]);
            mj_kinematics(model, data);
            for (int a = 0; a < 3; ++a) {
                observed_velocity[a] = (1. - plan[25]) * v[a] + plan[25] * observed_velocity[a];
                state[a] = q[a] - plan[3 + a] + plan[22 + a];
                state[a + 3] = observed_velocity[a];
                state[a + 6] = pad[a] - plan[3 + a];
            }
            const int sign = normal[8] >= 0. ? 1 : -1;
            state[9] = sign * normal[2];
            state[10] = sign * normal[5];
            command(state, weights, issued, action);

            for (int sub = 0; sub < kSubsteps && !lost; ++sub) {
                for (int a = 0; a < kJoints; ++a) {
                    data->ctrl[a] = std::clamp(data->ctrl[a] + action[a] * (kRate * model->opt.timestep),
                                              model->actuator_ctrlrange[2 * a], model->actuator_ctrlrange[2 * a + 1]);
                }
                if (t < lead) {
                    std::copy(location, location + 3, q);
                    std::fill(v, v + 6, 0.);
                    mj_step(model, data);
                } else {
                    const double incoming = v[2];
                    const double factor = plan[0] * std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
                    for (int a = 0; a < 3; ++a) data->xfrc_applied[6 * body + a] = factor * v[a];
                    mj_step(model, data);
                    lost = judge.sample(data, ball, blade, position, velocity, incoming);
                }
            }
            if (t < lead) {
                std::copy(location, location + 3, q);
                std::fill(v, v + 3, 0.);
                if (t + 1 == lead) std::copy(plan + 6, plan + 12, v);
                mj_forward(model, data);
            } else {
                ++ticks;
            }
        }
        score[0] = judge.hits;
        score[1] = judge.longest;
        score[2] = ticks;
        score[3] = judge.contacts;
        set_gravity(1.);
    }
};
}  // namespace

extern "C" void* tt24_donor_open(mjModel* model) { return new HeldDonor(model); }
extern "C" void tt24_donor_close(void* handle) { delete static_cast<HeldDonor*>(handle); }

extern "C" void tt24_donor_evaluate(void* handle, const double* weights, int population,
                                    const double* states, const double* plans, int starts,
                                    int cap, double* output) {
    auto* world = static_cast<HeldDonor*>(handle);
    for (int actor = 0; actor < population; ++actor) {
        for (int slot = 0; slot < starts; ++slot) {
            world->rollout(weights + actor * kJoints * kFeatures, states + slot * world->state_size,
                           plans + slot * kSettings, cap, output + 4 * (actor * starts + slot));
        }
    }
}
