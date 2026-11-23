#include "tt24_world_prior_native.cpp"

namespace {
constexpr int margin_fields = 16;
constexpr int command_buffer = 10;

Vector direction(Random& source) {
    const double angle = source.range(0., 6.283185307179586);
    return {std::cos(angle), std::sin(angle), 0.};
}

struct CompliantDisk {
    Vector origin, heading, position{}, tilt{}, velocity{}, turn{};
    double settle_xy, settle_z, settle_tilt, memory, speed_xy, speed_z, speed_tilt;
    double translation_gain, lift_gain, tilt_gain, coupling;

    bool advance(Action goal, Vector& step, Vector& twist) {
        const Vector sideways = {-heading.y, heading.x, 0.};
        Vector adjusted = flat(goal.position) * translation_gain + heading * (coupling * goal.position.z);
        adjusted = adjusted + goal.tilt * (.047 * coupling);
        adjusted.z = lift_gain * goal.position.z + coupling * flat(goal.position).dot(heading)
                     + .055 * coupling * goal.tilt.dot(sideways);
        Vector normal_goal = origin + goal.tilt * tilt_gain + sideways * (coupling * goal.position.z * 1.45)
                             + flat(goal.position) * (1.3 * coupling);
        normal_goal = cap(normal_goal, .29);
        Vector wish = (adjusted - position) / settle_xy;
        wish.z = (adjusted.z - position.z) / settle_z;
        const Vector want_turn = (normal_goal - tilt) / settle_tilt;
        const bool saturated = flat(wish).mag() > speed_xy || std::abs(wish.z) > speed_z || want_turn.mag() > speed_tilt;
        Vector regulated = cap(flat(wish), speed_xy);
        regulated.z = clamp(wish.z, speed_z);
        velocity = velocity * memory + regulated * (1. - memory);
        turn = turn * memory + cap(want_turn, speed_tilt) * (1. - memory);
        step = velocity * tick;
        twist = turn * tick;
        return saturated;
    }
};

void margin_episode(const double* weights, uint64_t seed, int profile, int budget, double* out) {
    Random random{seed};
    const bool hard = profile == 2;
    const bool disturbed = profile > 0;
    const double g = random.range(hard ? 8.0 : 8.7, hard ? 12.3 : 11.15);
    const double drag = random.range(.09, hard ? .30 : .22);
    const double e = random.range(hard ? .66 : .72, hard ? .84 : .87);
    const double traction_loss = random.range(.025, hard ? .19 : .12);
    const double radius = random.range(.019, .022);
    const double disk_radius = random.range(hard ? .062 : .069, .082);
    const double margin = random.range(.075, .115);
    const int lag = int(random.range(hard ? 3. : 1., hard ? 8. : 6.));
    CompliantDisk body;
    body.origin = flat(random.jitter(hard ? .047 : .031, 0.));
    body.heading = direction(random);
    body.position = random.jitter(.008, .006);
    body.tilt = body.origin + flat(random.jitter(.009, 0.));
    body.settle_xy = random.range(.039, hard ? .142 : .104);
    body.settle_z = random.range(.024, hard ? .094 : .065);
    body.settle_tilt = random.range(.049, hard ? .152 : .113);
    body.memory = random.range(.33, hard ? .81 : .71);
    body.speed_xy = random.range(hard ? .195 : .26, .56);
    body.speed_z = random.range(hard ? .16 : .22, .53);
    body.speed_tilt = random.range(hard ? .78 : 1.10, 2.9);
    body.translation_gain = random.range(.67, 1.06);
    body.lift_gain = random.range(hard ? .55 : .67, 1.02);
    body.tilt_gain = random.range(hard ? .62 : .73, 1.07);
    body.coupling = random.range(hard ? .32 : .20, hard ? .54 : .40) * (2. * (random.unit() > .5) - 1.);
    Vector ball = body.position + Vector{random.around(hard ? .045 : .036), random.around(hard ? .045 : .036),
                                         random.range(.183, .36)};
    Vector speed = {random.around(.080), random.around(.080), random.range(-.13, .033)};
    const double uncertainty = random.unit();
    const Vector position_bias = random.jitter((hard ? .0073 : .0045) * uncertainty, .009 * uncertainty);
    const Vector velocity_bias = random.jitter(.120 * uncertainty, (hard ? .27 : .19) * uncertainty);
    Vector noise{};
    Action queue[command_buffer]{}, previous{}, previous_change{};
    bool live = true, flying = false;
    int rebounds = 0, attacks = 0, previous_rebound = 0, run = 0, longest = 0, high = 0, frames = 0;
    int shock_frame = int(random.range(36., 90.)), shocks = 0, recovered = 0, pending_impact = -1;
    double age = 0., release_height = 0., apex = 0., separation = 0., rise_sum = 0., safety_sum = 0.;
    double pos_rate = 0., tilt_rate = 0., pos_acc = 0., tilt_acc = 0., effort = 0.;
    int requested_limits = 0, executed_limits = 0;
    const double dt = tick / substeps;
    const double clearance = radius + .0057;
    for (int frame = 0; frame < budget && live; ++frame) {
        frames += 1;
        if (disturbed && frame == shock_frame) {
            const double size = hard ? random.range(.17, .33) : random.range(.105, .22);
            speed = speed + direction(random) * size;
            shocks += 1;
            pending_impact = attacks;
            shock_frame += int(random.range(52., 105.));
        }
        noise = noise * .63 + random.jitter(.115 * uncertainty, .30 * uncertainty) * .37;
        const Action issued = action(weights, ball + position_bias + random.jitter(.0014*uncertainty, .0028*uncertainty),
                                     speed + velocity_bias + noise, body.position, body.tilt);
        effort += issued.position.dot(issued.position) / (.050*.050) + issued.tilt.dot(issued.tilt) / (.17*.17);
        if (frame > 0) {
            const Action change{(issued.position - previous.position) / tick, (issued.tilt - previous.tilt) / tick};
            pos_rate += change.position.dot(change.position);
            tilt_rate += change.tilt.dot(change.tilt);
            if (frame > 1) {
                const Vector acceleration = (change.position - previous_change.position) / tick;
                const Vector tilt_acceleration = (change.tilt - previous_change.tilt) / tick;
                pos_acc += acceleration.dot(acceleration);
                tilt_acc += tilt_acceleration.dot(tilt_acceleration);
            }
            requested_limits += (flat(change.position).mag() > body.speed_xy || std::abs(change.position.z) > body.speed_z
                                 || change.tilt.mag() > body.speed_tilt);
            previous_change = change;
        }
        previous = issued;
        queue[frame % command_buffer] = issued;
        const Action target = queue[(frame + command_buffer - lag) % command_buffer];
        Vector step{}, twist{};
        executed_limits += body.advance(target, step, twist);
        const Vector plate_v = step / tick;
        const Vector tilt_v = twist / tick;
        for (int sub = 0; sub < substeps; ++sub) {
            const Vector acceleration = speed * (-drag * speed.mag()) + Vector{0., 0., -g};
            Vector next = ball + speed * dt + acceleration * (.5 * dt * dt);
            Vector faster = speed + acceleration * dt;
            const Vector pw = body.position + plate_v * dt;
            const Vector tw = body.tilt + tilt_v * dt;
            const Vector n0 = unit_normal(body.tilt), n1 = unit_normal(tw);
            const double earlier = (ball - body.position).dot(n0) - clearance;
            const double later = (next - pw).dot(n1) - clearance;
            if (earlier >= -1.e-9 && later < 0.) {
                const double crossing = dt * std::clamp(earlier / std::max(1.e-9, earlier - later), 0., 1.);
                const Vector p = body.position + plate_v * crossing;
                const Vector xy = body.tilt + tilt_v * crossing;
                const Vector normal = unit_normal(xy);
                const Vector at = ball + speed * crossing + acceleration * (.5 * crossing * crossing);
                const Vector radius_at_contact = at - p - normal * clearance;
                const Vector normal_derivative{tilt_v.x, tilt_v.y, -xy.dot(tilt_v) / normal.z};
                const Vector contact_velocity = plate_v + normal.cross(normal_derivative).cross(radius_at_contact);
                const Vector incoming = speed + acceleration * crossing - contact_velocity;
                const double impact = incoming.dot(normal);
                if (radius_at_contact.mag() <= disk_radius && impact < 0.) {
                    const Vector tangent = incoming - normal * impact;
                    const Vector outward = contact_velocity + tangent * (1. - traction_loss) - normal * (e * impact);
                    const double rest = dt - crossing;
                    next = at + normal * .00003 + outward * rest + Vector{0., 0., -.5 * g * rest * rest};
                    faster = outward + Vector{0., 0., -g * rest};
                    attacks += 1;
                    flying = impact < -.25 && outward.z > .25;
                    age = -crossing;
                    release_height = apex = at.z;
                    separation = 0.;
                }
            }
            ball = next;
            speed = faster;
            body.position = pw;
            body.tilt = tw;
            if (flying) {
                age += dt;
                apex = std::max(apex, ball.z);
                separation = std::max(separation, (ball - body.position).dot(n1) - clearance);
                if (speed.z <= 0. && age >= .065) {
                    flying = false;
                    const double rise = apex - release_height;
                    if (rise >= .045 && separation >= .018) {
                        rebounds += 1;
                        high += rise >= .100;
                        rise_sum += rise;
                        safety_sum += std::clamp((rise - margin) / .055, 0., 2.);
                        run = attacks == previous_rebound + 1 ? run + 1 : 1;
                        previous_rebound = attacks;
                        longest = std::max(longest, run);
                        if (pending_impact >= 0 && attacks > pending_impact) {
                            recovered += 1;
                            pending_impact = -1;
                        }
                    }
                }
            }
            if (ball.z < -.14 || ball.z > 1.0 || std::abs(ball.x) > .40 || std::abs(ball.y) > .40) {
                live = false;
                break;
            }
        }
    }
    out[0] = rebounds; out[1] = frames; out[2] = longest; out[3] = attacks;
    out[4] = rise_sum; out[5] = safety_sum; out[6] = high; out[7] = shocks; out[8] = recovered;
    out[9] = std::sqrt(pos_rate / std::max(1, frames-1)); out[10] = std::sqrt(tilt_rate / std::max(1, frames-1));
    out[11] = std::sqrt(pos_acc / std::max(1, frames-2)); out[12] = std::sqrt(tilt_acc / std::max(1, frames-2));
    out[13] = double(requested_limits) / std::max(1, frames-1); out[14] = double(executed_limits) / std::max(1, frames);
    out[15] = effort / std::max(1, frames);
}
} // namespace

extern "C" void tt24_world_margin_rollouts(const double* weights, int models, const uint64_t* seeds, int count,
                                           int profile, int frames, int workers, double* results) {
    #pragma omp parallel for num_threads(workers) schedule(dynamic, 16)
    for (int i = 0; i < models * count; ++i) {
        margin_episode(weights + weight_count * (i / count), seeds[i % count], profile, frames, results + margin_fields * i);
    }
}
