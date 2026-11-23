#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <mujoco/mujoco.h>

extern "C" void tt24_world_target(const double*, const double*, double*);

namespace {
constexpr int weights = 42;
constexpr int fields = 6;
constexpr int substeps = 8;
constexpr double height = .60;
constexpr double dt = 1.0 / 480.0;

const char* model_xml = R"xml(
<mujoco model="Cartesian rebound gantry">
  <compiler autolimits="true" angle="radian"/>
  <option timestep=".0020833333333333333" integrator="implicitfast" iterations="14" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 .44" size="1 1 .02" contype="2" conaffinity="1"/>
    <body name="carriage" pos="0 0 .60" gravcomp="1">
      <inertial pos="0 0 0" mass=".110" diaginertia=".00025 .00025 .00048"/>
      <joint name="x" type="slide" axis="1 0 0" range="-.15 .15" damping=".08" armature=".010"/>
      <joint name="y" type="slide" axis="0 1 0" range="-.15 .15" damping=".08" armature=".010"/>
      <joint name="z" type="slide" axis="0 0 1" range="-.08 .08" damping=".08" armature=".010"/>
      <joint name="turn_y" type="hinge" axis="0 1 0" range="-.33 .33" damping=".0004" armature=".0001"/>
      <joint name="turn_x" type="hinge" axis="1 0 0" range="-.33 .33" damping=".0004" armature=".0001"/>
      <geom name="disk" type="cylinder" size=".076 .006" contype="2" conaffinity="1"
            priority="2" condim="3" friction=".26 .003 .00012" solref="-100000 -32"
            solimp=".99 .99 .001 .5 2"/>
    </body>
    <body name="ball_body" pos="0 0 .84">
      <freejoint name="ball_joint"/>
      <geom name="ball" type="sphere" size=".02" mass=".0027" contype="1" conaffinity="2"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ctrl_x" joint="x" kp="170" kv="8.8" forcerange="-3 3"/>
    <position name="ctrl_y" joint="y" kp="170" kv="8.8" forcerange="-3 3"/>
    <position name="ctrl_z" joint="z" kp="205" kv="9.6" forcerange="-3.6 3.6"/>
    <position name="ctrl_turn_y" joint="turn_y" kp=".82" kv=".027" forcerange="-.10 .10"/>
    <position name="ctrl_turn_x" joint="turn_x" kp=".82" kv=".027" forcerange="-.10 .10"/>
  </actuator>
</mujoco>
)xml";

struct Random {
    uint64_t value;
    double between(double lo, double hi) {
        auto next = (value += UINT64_C(0x9e3779b97f4a7c15));
        next = (next ^ (next >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        next = (next ^ (next >> 27)) * UINT64_C(0x94d049bb133111eb);
        return lo + (hi - lo) * (((next ^ (next >> 31)) >> 11) * 0x1.0p-53);
    }
    double around(double amount) { return between(-amount, amount); }
};

struct Gantry {
    mjModel* model = nullptr;
    mjData* data = nullptr;
    int blade = -1, ball = -1, body = -1, qpos = -1, qvel = -1;

    Gantry() {
        char error[1024] = {};
        mjSpec* spec = mj_parseXMLString(model_xml, nullptr, error, sizeof(error));
        if (spec) {
            model = mj_compile(spec, nullptr);
            if (!model) std::fprintf(stderr, "%s\n", mjs_getError(spec));
            mj_deleteSpec(spec);
        } else {
            std::fprintf(stderr, "%s\n", error);
        }
        if (!model) return;
        data = mj_makeData(model);
        blade = mj_name2id(model, mjOBJ_GEOM, "disk");
        ball = mj_name2id(model, mjOBJ_GEOM, "ball");
        body = mj_name2id(model, mjOBJ_BODY, "ball_body");
        int free = mj_name2id(model, mjOBJ_JOINT, "ball_joint");
        qpos = model->jnt_qposadr[free];
        qvel = model->jnt_dofadr[free];
    }

    ~Gantry() {
        if (data) mj_deleteData(data);
        if (model) mj_deleteModel(model);
    }

    void configure(Random& rng, bool stress) {
        mj_resetData(model, data);
        model->opt.gravity[2] = -rng.between(stress ? 8.2 : 9.05, stress ? 11.5 : 10.55);
        model->geom_solref[2*blade] = -rng.between(stress ? 66000. : 84000., stress ? 135000. : 117000.);
        model->geom_solref[2*blade + 1] = -rng.between(stress ? 22. : 25., stress ? 59. : 48.);
        model->geom_friction[3*blade] = rng.between(stress ? .12 : .19, stress ? .54 : .36);
        const double kp[5] = {170., 170., 205., .82, .82};
        const double kv[5] = {8.8, 8.8, 9.6, .027, .027};
        const double force[5] = {3., 3., 3.6, .10, .10};
        for (int i = 0; i < 5; ++i) {
            const double drive = rng.between(stress ? .54 : .78, stress ? 1.41 : 1.24);
            model->actuator_gainprm[mjNGAIN*i] = kp[i]*drive;
            model->actuator_biasprm[mjNBIAS*i + 1] = -kp[i]*drive;
            model->actuator_biasprm[mjNBIAS*i + 2] = -kv[i]*std::sqrt(drive)*rng.between(.86, 1.15);
            const double cap = force[i]*rng.between(stress ? .56 : .80, stress ? 1.33 : 1.19);
            model->actuator_forcerange[2*i] = -cap;
            model->actuator_forcerange[2*i + 1] = cap;
        }
        double* position = data->qpos + qpos;
        double* speed = data->qvel + qvel;
        position[0] = rng.around(stress ? .050 : .036);
        position[1] = rng.around(stress ? .050 : .036);
        position[2] = height + rng.between(stress ? .145 : .177, stress ? .350 : .304);
        speed[0] = rng.around(stress ? .115 : .074);
        speed[1] = rng.around(stress ? .115 : .074);
        speed[2] = rng.between(-.15, .04);
        for (int i = 3; i < 6; ++i) speed[i] = rng.around(stress ? 14. : 8.);
        mj_forward(model, data);
    }

    void flight_force(double drag, double spin_decay, double magnus) {
        const double* v = data->qvel + qvel;
        const double* matrix = data->xmat + 9*body;
        double world_spin[3];
        mju_mulMatVec3(world_spin, matrix, v + 3);
        double spin_force[3];
        mju_cross(spin_force, world_spin, v);
        double* force = data->xfrc_applied + 6*body;
        const double scale = -.0027 * drag * std::sqrt(mju_dot3(v, v));
        for (int i = 0; i < 3; ++i) {
            force[i] = scale*v[i] + .0027*magnus*spin_force[i];
            force[i + 3] = -spin_decay * (.4*.0027*.02*.02) * world_spin[i];
        }
    }

    void play(const double* w, uint64_t seed, int profile, int budget, double* result) {
        Random rng{seed};
        bool stress = profile != 0;
        configure(rng, stress);
        const double drag = rng.between(stress ? .075 : .12, stress ? .31 : .235);
        const double spin_decay = rng.between(.11, .24);
        const double magnus = rng.between(.00012, .00037);
        const int delay = int(rng.between(0., stress ? 3. : 2.));
        std::array<std::array<double, 5>, 3> requests{};
        const double* ball_pos = data->qpos + qpos;
        const double* speed = data->qvel + qvel;
        const double* paddle = data->geom_xpos + 3*blade;
        const double* frame = data->geom_xmat + 9*blade;
        int contacts = 0, rebounds = 0, previous_impact = 0, best = 0, streak = 0, powered = 0, frames = 0;
        double rise_sum = 0., strike_z = 0., takeoff_z = 0., peak = 0., face_speed = 0.;
        double strike_time = 0., takeoff_time = 0.;
        bool stopped = false, touching = false, candidate = false, airborne = false;
        for (int tick = 0; tick < budget && !stopped; ++tick) {
            frames += 1;
            mj_kinematics(model, data);
            const double state[11] = {ball_pos[0], ball_pos[1], ball_pos[2] - height, speed[0], speed[1], speed[2],
                                      paddle[0], paddle[1], paddle[2] - height, frame[2], frame[5]};
            tt24_world_target(w, state, requests[tick % 3].data());
            const auto& request = requests[(tick + 3 - delay) % 3];
            for (int k = 0; k < 3; ++k) data->ctrl[k] = request[k];
            const double rx = -std::asin(std::clamp(request[4], -.32, .32));
            data->ctrl[3] = std::asin(std::clamp(request[3] / std::cos(rx), -.32, .32));
            data->ctrl[4] = rx;
            for (int step = 0; step < substeps; ++step) {
                const double incoming = speed[2];
                const double before_z = ball_pos[2];
                flight_force(drag, spin_decay, magnus);
                mj_step(model, data);
                const mjContact* hit = nullptr;
                int normal_sign = 1;
                for (int k = 0; k < data->ncon; ++k) {
                    const mjContact& contact = data->contact[k];
                    if (contact.efc_address < 0) continue;
                    if (contact.geom[0] != ball && contact.geom[1] != ball) continue;
                    const int other = contact.geom[0] == ball ? contact.geom[1] : contact.geom[0];
                    if (other != blade) { stopped = true; break; }
                    hit = &contact;
                    normal_sign = contact.geom[0] == ball ? -1 : 1;
                }
                if (stopped) break;
                if (hit) {
                    airborne = false;
                    if (!touching) {
                        contacts += 1;
                        const double alignment = hit->frame[0]*frame[2] + hit->frame[1]*frame[5] + hit->frame[2]*frame[8];
                        candidate = hit->frame[2]*normal_sign > .55 && std::abs(alignment) > .95 && incoming < -.25;
                        if (candidate) {
                            strike_z = before_z;
                            strike_time = data->time;
                            double axis_velocity[6];
                            mj_objectVelocity(model, data, mjOBJ_GEOM, blade, axis_velocity, 0);
                            const double offset_x = hit->pos[0] - paddle[0];
                            const double offset_y = hit->pos[1] - paddle[1];
                            face_speed = axis_velocity[5] + axis_velocity[0]*offset_y - axis_velocity[1]*offset_x;
                        }
                    }
                    touching = true;
                } else if (touching) {
                    touching = false;
                    airborne = candidate && data->time - strike_time <= .032 && speed[2] > .25;
                    if (airborne) {
                        peak = takeoff_z = ball_pos[2];
                        takeoff_time = data->time;
                    }
                }
                if (airborne) {
                    peak = std::max(peak, ball_pos[2]);
                    if (speed[2] <= 0. && data->time - takeoff_time >= .065) {
                        airborne = false;
                        const double rise = peak - strike_z;
                        if (rise >= .045 && peak - takeoff_z >= .018) {
                            rebounds += 1;
                            streak = contacts == previous_impact + 1 ? streak + 1 : 1;
                            previous_impact = contacts;
                            best = std::max(best, streak);
                            rise_sum += rise;
                            powered += (face_speed > .055);
                        }
                    }
                }
                if (ball_pos[2] < height - .125 || ball_pos[2] > height + .97 ||
                    std::abs(ball_pos[0]) > .42 || std::abs(ball_pos[1]) > .42) {
                    stopped = true;
                    break;
                }
            }
        }
        result[0] = rebounds;
        result[1] = frames;
        result[2] = best;
        result[3] = contacts;
        result[4] = rise_sum / std::max(rebounds, 1);
        result[5] = powered;
    }
};
} // namespace

extern "C" void* tt24_gantry_open() {
    auto* gantry = new Gantry();
    if (gantry->model) return gantry;
    delete gantry;
    return nullptr;
}

extern "C" void tt24_gantry_close(void* handle) {
    delete static_cast<Gantry*>(handle);
}

extern "C" void tt24_gantry_rollouts(void* handle, const double* population, int count, const uint64_t* seeds,
                                     int worlds, int profile, int frames, double* result) {
    auto& gantry = *static_cast<Gantry*>(handle);
    for (int i = 0; i < count*worlds; ++i) {
        gantry.play(population + weights*(i/worlds), seeds[i % worlds], profile, frames, result + fields*i);
    }
}
