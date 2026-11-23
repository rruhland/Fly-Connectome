#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
constexpr double tick = 1.0 / 60.0;
constexpr int substeps = 2;
constexpr int weight_count = 42;
constexpr int fields = 6;
constexpr int body_width = 96;

struct Vector {
    double x, y, z;
    Vector operator+(Vector b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vector operator-(Vector b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vector operator*(double k) const { return {x * k, y * k, z * k}; }
    Vector operator/(double k) const { return *this * (1.0 / k); }
    double dot(Vector b) const { return x * b.x + y * b.y + z * b.z; }
    Vector cross(Vector b) const { return {y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x}; }
    double mag() const { return std::sqrt(dot(*this)); }
};

struct Random {
    uint64_t state;
    double unit() {
        uint64_t v = (state += UINT64_C(0x9e3779b97f4a7c15));
        v = (v ^ (v >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        v = (v ^ (v >> 27)) * UINT64_C(0x94d049bb133111eb);
        return ((v ^ (v >> 31)) >> 11) * 0x1.0p-53;
    }
    double range(double lo, double hi) { return lo + (hi - lo) * unit(); }
    double around(double amplitude) { return range(-amplitude, amplitude); }
    Vector jitter(double lo, double hi) { return {around(lo), around(lo), around(hi)}; }
};

struct Action { Vector position, tilt; };

Vector flat(Vector a) { return {a.x, a.y, 0.0}; }
Vector unit_normal(Vector xy) {
    return {xy.x, xy.y, std::sqrt(std::max(0.01, 1.0 - xy.dot(xy)))};
}
Vector bounded(Vector a, double scale) { return a / std::max(scale, a.mag()); }
Vector cap(Vector a, double distance) { return a * std::min(1.0, distance / std::max(1.e-12, a.mag())); }
double clamp(double x, double cap_value) { return std::clamp(x, -cap_value, cap_value); }

Action action(const double* w, Vector ball, Vector speed, Vector plate, Vector tilt) {
    const Vector v[4] = {bounded(flat(ball - plate), .074), bounded(flat(speed), .42),
                         bounded(flat(plate), .071), bounded(tilt, .19)};
    const double z = clamp(ball.z / .19, 1.0), dz = clamp(speed.z / 2.1, 1.0), bz = clamp(plate.z / .041, 1.0);
    const double f[10] = {1., z, dz, bz, z*dz, z*z, dz*dz, v[0].dot(v[0]), v[1].dot(v[1]), v[0].dot(v[1])};
    const double h[4] = {1., z, dz, f[7]};
    double lift = 0.;
    for (int i = 0; i < 10; ++i) lift += w[i] * f[i];
    const double limit[2] = {.069, .181};
    Vector result[2] = {};
    for (int group = 0; group < 2; ++group) {
        for (int i = 0; i < 4; ++i) {
            double coefficient = 0.;
            for (int j = 0; j < 4; ++j) coefficient += w[10 + 16*group + 4*i + j] * h[j];
            result[group] = result[group] + v[i] * coefficient;
        }
        result[group] = result[group] * (limit[group] / (1. + result[group].mag()));
    }
    result[0].z = .039 * lift / (1. + std::abs(lift));
    return {result[0], result[1]};
}

void episode(const double* weights, uint64_t seed, int profile, int budget, double* out,
             const double* fitted = nullptr) {
    Random random{seed};
    const bool hard = profile > 0;
    const bool stress = profile == 2;
    // These are varied body/flight parameters. The controller receives no parameter labels.
    const double g = stress ? random.range(6.9, 13.0) : hard ? random.range(7.4, 12.6) : random.range(8.6, 11.1);
    const double drag = hard ? random.range(.07, .32) : random.range(.08, .24);
    const double restitution = stress ? random.range(.54, .78) : hard ? random.range(.59, .88) : random.range(.68, .84);
    const double tangent_loss = random.range(.025, hard ? .24 : .155);
    const double radius = random.range(.018, .0225);
    const double disk_radius = hard ? random.range(.062, .089) : random.range(.069, .083);
    const int native_delay = int(random.unit() * (stress ? 7 : hard ? 5 : 3));
    const int delay = fitted ? int(fitted[95]) : native_delay;
    const double pxo = random.range(hard ? .016 : .009, hard ? .052 : .038);
    const double pzo = random.range(.012, hard ? .045 : .037);
    const double nto = random.range(.014, stress ? .085 : hard ? .065 : .044);
    const double translation_speed = random.range(.39, hard ? .96 : .83);
    const double vertical_speed = random.range(.32, hard ? 1.02 : .88);
    const double tilt_speed = random.range(1.55, hard ? 5.7 : 4.6);
    const double noise = random.unit();
    const Vector position_bias = random.jitter((hard ? .009 : .0055) * noise, (hard ? .017 : .010) * noise);
    const Vector speed_bias = random.jitter(.16 * noise, (stress ? .39 : .22) * noise);
    const Vector resting_tilt = flat(random.jitter(hard ? .045 : .023, 0.));
    Vector plate = random.jitter(.008, .006);
    Vector tilt = resting_tilt + flat(random.jitter(.009, 0.));
    if (fitted) {
        plate = {fitted[85], fitted[86], fitted[87]};
        tilt = {fitted[88], fitted[89], 0.};
    }
    Vector ball = plate + Vector{random.around(hard ? .053 : .040), random.around(hard ? .053 : .040),
                                 random.range(hard ? .137 : .161, hard ? .36 : .320)};
    Vector speed = {random.around(hard ? .123 : .077), random.around(hard ? .123 : .077), random.range(-.15, .04)};
    Vector sensor_speed{};
    Action controls[7]{};
    double previous[5] = {};
    const double scales[5] = {.069, .069, .039, .181, .181};
    const double resolution = tick / substeps;
    const double reach = radius + .0057;
    const double move_gain = 1.0 - std::exp(-tick / pxo), lift_gain = 1.0 - std::exp(-tick / pzo);
    const double turn_gain = 1.0 - std::exp(-tick / nto);
    int frames = 0, rebounds = 0, attacks = 0, current_run = 0, longest_run = 0, previous_impact = 0, powered = 0;
    bool continued = true, flying = false;
    double launch_height = 0., peak = 0., elapsed = 0., gap_peak = 0., rise_sum = 0., launch_paddle_z = 0.;
    for (int frame = 0; frame < budget && continued; ++frame) {
        frames += 1;
        sensor_speed = sensor_speed * .43 + random.jitter(.14 * noise, .44 * noise) * .57;
        Action intended = action(weights, ball + position_bias + random.jitter(.0017*noise, .0036*noise),
                                speed + speed_bias + sensor_speed, plate, tilt);
        controls[frame % 7] = intended;
        Action request = controls[(frame + 7 - delay) % 7];
        Vector movement = (request.position - plate) * move_gain;
        movement.z = (request.position.z - plate.z) * lift_gain;
        movement.z = clamp(movement.z, vertical_speed * tick);
        Vector horizontal = cap(flat(movement), translation_speed * tick);
        movement.x = horizontal.x; movement.y = horizontal.y;
        Vector rotation = cap((resting_tilt + request.tilt - tilt) * turn_gain, tilt_speed * tick);
        if (fitted) {
            const double readings[5] = {plate.x, plate.y, plate.z, tilt.x, tilt.y};
            const double goals[5] = {request.position.x, request.position.y, request.position.z, request.tilt.x, request.tilt.y};
            double features[16] = {1.}, displacement[5] = {};
            for (int i = 0; i < 5; ++i) {
                features[1+i] = readings[i] / scales[i];
                features[6+i] = previous[i];
                features[11+i] = (goals[i] - readings[i]) / scales[i];
            }
            for (int i = 0; i < 5; ++i) {
                double estimate = 0.;
                for (int j = 0; j < 16; ++j) estimate += features[j] * fitted[5*j+i];
                const double move = scales[i] * clamp(estimate, fitted[80+i]);
                displacement[i] = clamp(readings[i] + move, fitted[90+i]) - readings[i];
                previous[i] = displacement[i] / scales[i];
            }
            movement = {displacement[0], displacement[1], displacement[2]};
            rotation = {displacement[3], displacement[4], 0.};
        }
        const Vector plate_velocity = movement / tick;
        const Vector tilt_velocity = rotation / tick;
        for (int j = 0; j < substeps; ++j) {
            const Vector a = speed * (-drag * speed.mag()) + Vector{0., 0., -g};
            Vector next_ball = ball + speed * resolution + a * (.5 * resolution * resolution);
            Vector next_speed = speed + a * resolution;
            const Vector next_plate = plate + plate_velocity * resolution;
            const Vector next_tilt = tilt + tilt_velocity * resolution;
            const Vector n = unit_normal(tilt), next_n = unit_normal(next_tilt);
            const double before = (ball - plate).dot(n) - reach;
            const double after = (next_ball - next_plate).dot(next_n) - reach;
            if (before >= -1.e-9 && after < 0.) {
                const double fraction = std::clamp(before / std::max(1.e-9, before - after), 0., 1.);
                const double hit_time = resolution * fraction;
                const Vector hit_pose = plate + plate_velocity * hit_time;
                const Vector normal_xy = tilt + tilt_velocity * hit_time;
                const Vector normal = unit_normal(normal_xy);
                const Vector point = ball + speed * hit_time + a * (.5 * hit_time * hit_time);
                const Vector offset = point - hit_pose - normal * reach;
                // Angular velocity of the disk frame contributes at the actual contact point.
                const Vector normal_rate = {tilt_velocity.x, tilt_velocity.y, -normal_xy.dot(tilt_velocity) / normal.z};
                const Vector surface = plate_velocity + normal.cross(normal_rate).cross(offset);
                const Vector incoming = speed + a * hit_time - surface;
                const double impact = incoming.dot(normal);
                if (offset.mag() <= disk_radius && impact < 0.) {
                    const Vector tangent = incoming - normal * impact;
                    const Vector outgoing = surface + tangent * (1.0 - tangent_loss) - normal * (restitution * impact);
                    const double remainder = resolution - hit_time;
                    next_ball = point + normal * .00003 + outgoing * remainder + Vector{0., 0., -.5 * g * remainder * remainder};
                    next_speed = outgoing + Vector{0., 0., -g * remainder};
                    attacks += 1;
                    flying = impact < -.25 && outgoing.z > .25;
                    launch_height = point.z;
                    elapsed = -hit_time;
                    peak = point.z;
                    gap_peak = 0.;
                    launch_paddle_z = surface.z;
                }
            }
            ball = next_ball;
            plate = next_plate;
            tilt = next_tilt;
            speed = next_speed;
            if (flying) {
                elapsed += resolution;
                peak = std::max(peak, ball.z);
                gap_peak = std::max(gap_peak, (ball - plate).dot(next_n) - reach);
                if (speed.z <= 0. && elapsed >= .065) {
                    flying = false;
                    const double rise = peak - launch_height;
                    if (rise >= .045 && gap_peak >= .018) {
                        rebounds += 1;
                        current_run = attacks == previous_impact + 1 ? current_run + 1 : 1;
                        previous_impact = attacks;
                        longest_run = std::max(longest_run, current_run);
                        rise_sum += rise;
                        powered += (launch_paddle_z > .055);
                    }
                }
            }
            if (ball.z < -.125 || ball.z > .97 || std::abs(ball.x) > .42 || std::abs(ball.y) > .42) {
                continued = false;
                break;
            }
        }
    }
    out[0] = rebounds;
    out[1] = frames;
    out[2] = longest_run;
    out[3] = attacks;
    out[4] = rise_sum / std::max(rebounds, 1);
    out[5] = powered;
}
} // namespace

extern "C" void tt24_world_target(const double* weights, const double* state, double* out) {
    const Action goal = action(weights, {state[0],state[1],state[2]}, {state[3],state[4],state[5]},
                               {state[6],state[7],state[8]}, {state[9],state[10],0.});
    out[0]=goal.position.x; out[1]=goal.position.y; out[2]=goal.position.z; out[3]=goal.tilt.x; out[4]=goal.tilt.y;
}

extern "C" void tt24_world_rollouts(const double* weights, int n, const uint64_t* seeds, int m,
                                   int profile, int frames, int workers, double* out) {
    #pragma omp parallel for num_threads(workers) schedule(dynamic, 16)
    for (int i = 0; i < n * m; ++i) {
        episode(weights + weight_count * (i / m), seeds[i % m], profile, frames, out + fields * i);
    }
}

extern "C" void tt24_world_aligned_rollouts(const double* weights, int n, const uint64_t* seeds, int m,
                                            int profile, int frames, int workers, const double* plants,
                                            int models, double* out) {
    #pragma omp parallel for num_threads(workers) schedule(dynamic, 16)
    for (int i = 0; i < n * m; ++i) {
        const uint64_t seed = seeds[i % m];
        const double* body = plants + body_width * (seed % models);
        episode(weights + weight_count * (i / m), seed, profile, frames, out + fields * i, body);
    }
}
