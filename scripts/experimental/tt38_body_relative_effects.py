"""Train actual short-horizon encoder response and use it in a direct motor readout."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_deep import DeepField, run
from .tt37_opponent import arena
from .tt38_body_relative import OUT, PLAN, Body, body_inputs, groups, input_motion, regress

BANK = PLAN.with_name("pooled-rollin-rgb.npz")
HORIZON = 3
SPEED_SCALE = np.array([.75, .85, .80])


def feature(q, motion, issued):
    error = (issued - q) / .20
    rate = motion / .05
    return np.concatenate((np.ones((*q.shape[:-1], 1)), error, rate, q, error*q, rate*q), axis=-1)


class EncoderDynamics:
    """Causal autoregression learned from measured encoder differences and issued targets."""

    def __init__(self, weights):
        self.w = weights.copy()

    def motion(self, q, dq, issued):
        return feature(q, dq, issued) @ self.w

    def differential(self, q, dq, issued):
        _, cmd, rate, pose, mixed_cmd, mixed_rate = np.split(self.w, [1, 6, 11, 16, 21], axis=0)
        aq = (-cmd / .20 + pose + ((issued - 2*q) / .20)[:, None] * mixed_cmd
              + (dq / .05)[:, None] * mixed_rate).T
        ar = ((rate + q[:, None] * mixed_rate) / .05).T
        au = ((cmd + q[:, None] * mixed_cmd) / .20).T
        return aq, ar, au

    def predict(self, q, dq, issued, horizon=HORIZON, *, sensitivity=False):
        """Pure encoder forecast, with a derivative for a bounded held actuator command."""
        if sensitivity:
            hq = np.zeros((5, 5))
            hr = np.zeros((5, 5))
        for _ in range(horizon):
            if sensitivity:
                aq, ar, au = self.differential(q, dq, issued)
                hr = aq @ hq + ar @ hr + au
                hq = hq + hr
            dq = self.motion(q, dq, issued)
            q = q + dq
        return (q, hq) if sensitivity else q


def sample_indices(bank, receipt, horizon=HORIZON):
    if horizon < 1:
        raise ValueError("At least one control tick is required to measure a response")
    time = bank["marks"][:, 0]
    point, release = groups(receipt)
    steps = (np.diff(time) == 1) & (np.diff(point) == 0)
    valid = steps[:-horizon].copy()
    for step in range(1, horizon + 1):
        valid &= steps[step:step+len(valid)]
    i = np.flatnonzero(valid) + 1
    capture = bank["marks"][i, 1]
    i = i[(capture > -.17) & (capture < 0.) & (bank["states"][i, 4] < -.9)
          & (bank["marks"][i+horizon, 1] < 0.)]
    return i, point, release


def fit_weights(x, y):
    return np.linalg.solve(x.T @ x + .04 * np.eye(x.shape[1]), x.T @ y)


def stats(values):
    return dict(median=float(np.median(values)), p90=float(np.quantile(values, .9)))


def fit(bank_path, body_path, output):
    meta = _read(bank_path.with_suffix(".json"))
    if meta["bank_sha256"] != _hash(bank_path):
        raise ValueError("Unexpected bank")
    with np.load(bank_path) as z, np.load(body_path) as body_fit:
        bank, body = dict(z), Body(body_fit["weights"])
    q = bank["states"][:, 6:11]
    issued = bank["targets_now"][:, :5]
    i, _, releases = sample_indices(bank, meta, horizon=1)
    long, _, _ = sample_indices(bank, meta, horizon=HORIZON)
    x, y = feature(q[i], q[i] - q[i-1], issued[i]), q[i+1]-q[i]
    short_error = np.empty((len(i), 2))
    long_error = np.empty((len(long), 3))
    for release in np.unique(releases[i]):
        test = releases[i] == release
        model = EncoderDynamics(fit_weights(x[~test], y[~test]))
        short = i[test]
        errors = body.position(q[short]+model.motion(q[short], q[short]-q[short-1], issued[short]))
        short_error[test, 0] = np.linalg.norm(errors - body.position(q[short+1]), axis=1)
        short_error[test, 1] = np.linalg.norm(body.position(2*q[short]-q[short-1])-body.position(q[short+1]), axis=1)
        hold = releases[long] == release
        s = long[hold]
        pose, motion = q[s], q[s] - q[s-1]
        for step in range(HORIZON):
            motion = model.motion(pose, motion, issued[s+step])
            pose = pose + motion
        long_error[hold, 0] = np.linalg.norm(body.position(pose) - body.position(q[s+HORIZON]), axis=1)
        long_error[hold, 1] = np.linalg.norm(body.position(q[s] + HORIZON*(q[s]-q[s-1]))
                                           - body.position(q[s+HORIZON]), axis=1)
        final_v = (body.position(q[s+HORIZON]) - body.position(q[s+HORIZON-1]))*60
        forecast_v = (body.position(pose) - body.position(pose-motion))*60
        long_error[hold, 2] = np.linalg.norm(final_v - forecast_v, axis=1)
    a = arena(protocol="random")
    try:
        axis = a._arm_ids[1][2][:5]
        lower, upper = a._joint_low[axis], a._joint_high[axis]
        step = a._joint_increments[axis] * (a.control_dt / a.model.opt.timestep) * .8
    finally:
        a.close()
    np.savez_compressed(output, weights=fit_weights(x, y), lower=lower, upper=upper, step=step,
                        bank_sha256=_hash(bank_path), body_sha256=_hash(body_path))
    requested = np.linalg.norm(body.position(issued[i]) - body.position(q[i]), axis=1)
    actual = np.linalg.norm(body.position(q[i+1]) - body.position(q[i]), axis=1)
    output_stats = dict(
        one_tick_rows=len(i), three_tick_rows=len(long), releases=len(set(releases[i])),
        nominal_command_movement_m=stats(requested), actual_movement_m=stats(actual),
        one_tick_encoder_model_m=stats(short_error[:, 0]), one_tick_constant_velocity_m=stats(short_error[:, 1]),
        three_tick_encoder_model_m=stats(long_error[:, 0]), three_tick_constant_velocity_m=stats(long_error[:, 1]),
        three_tick_final_velocity_error_mps=stats(long_error[:, 2]),
        validation="Entire release held out. Three-tick diagnostic is given the actual issued commands in order.",
        bank_sha256=_hash(bank_path), body_sha256=_hash(body_path), fit_sha256=_hash(output))
    _write(output.with_suffix(".json"), output_stats)
    print(output_stats, flush=True)


class EffectField:
    """Match observed short-horizon paddle motion with a bounded change to a direct command."""

    def __init__(self, bank, receipt, weights, fit):
        self.base = DeepField(bank, "local")
        self.body = Body(weights)
        self.motor = EncoderDynamics(fit["weights"])
        self.lower, self.upper, self.step = (fit[key] for key in ("lower", "upper", "step"))
        s = bank["states"]
        i, point, _ = sample_indices(bank, receipt)
        p, v = input_motion(s, bank["marks"], point, self.body)
        self.x = np.c_[body_inputs(s[i, :6], s[i, 6:11], p[i], v[i], "position"), v[i] / SPEED_SCALE]
        self.displacement = p[i + HORIZON] - p[i]
        self.tree = cKDTree(self.x)
        self.begin_point()

    def begin_point(self):
        self.prev = None
        self.diagnostics = dict(proposed_ticks=0, corrected_ticks=0, modeled_error_before_sum=0.,
                                modeled_error_after_sum=0., observed_next_position_error_sum=0.,
                                observed_next_positions=0)
        self.forecast = None

    def clock_tick(self, tick):
        self.tick = tick

    def get(self, optical, q, *, reference):
        desired, confidence = self.base.get(optical, q, reference=reference)
        q = q[:5]
        previous = self.prev
        self.prev = self.tick, q.copy()
        if self.forecast is not None and self.forecast[0] == self.tick - 1:
            self.diagnostics["observed_next_position_error_sum"] += float(np.linalg.norm(
                self.body.position(q) - self.forecast[1]))
            self.diagnostics["observed_next_positions"] += 1
        if (previous is None or previous[0] != self.tick-1 or desired is None or confidence < .99
                or optical[4] >= -.9):
            return desired, confidence
        p = self.body.position(q)
        longitudinal = optical[1] - p[1]
        if not (.03 < longitudinal < .51):
            return desired, confidence
        dq = q - previous[1]
        v = np.clip(60. * (p - self.body.position(previous[1])), -3., 3.)
        x = np.r_[body_inputs(optical, q, p, v, "position"), v / SPEED_SCALE]
        distance, near = self.tree.query(x, 12)
        support = float(np.clip((1.45 - distance[2]) / .7, 0., 1.))
        if support <= 0.:
            return desired, confidence
        self.diagnostics["proposed_ticks"] += 1
        goal = regress(self.x[near], self.displacement[near], distance, x)
        # Respect the rate limit before predicting motion. The actor never sees a hypothetical ball.
        lo = np.maximum(self.lower, reference[:5]-self.step)
        hi = np.minimum(self.upper, reference[:5]+self.step)
        base = np.clip(desired[:5], lo, hi)
        following, gain = self.motor.predict(q, dq, base, sensitivity=True)
        weighting = np.array([1., .55, 1.])
        error = weighting * (p + goal - self.body.position(following))
        j = (self.body.jacobian(following) @ gain) * weighting[:, None]
        delta = j.T @ np.linalg.solve(j @ j.T + .025**2 * np.eye(3), error)
        # Keep the direct joint readout as an anchor and reject backwards model changes.
        change = np.clip(.65 * support * delta, -.13, .13)
        chosen = np.clip(base + change, lo, hi)
        corrected = weighting * (p + goal - self.body.position(self.motor.predict(q, dq, chosen)))
        before, after = np.linalg.norm(error), np.linalg.norm(corrected)
        if after >= .9 * before or np.linalg.norm(chosen - base) < 1.e-5:
            return desired, confidence
        self.diagnostics["corrected_ticks"] += 1
        self.diagnostics["modeled_error_before_sum"] += float(before)
        self.diagnostics["modeled_error_after_sum"] += float(after)
        self.forecast = self.tick, self.body.position(q + self.motor.motion(q, dq, chosen))
        return np.r_[chosen, desired[5]], confidence


def physical(bank, fit_path, body, feeds, output, deck):
    receipt = _read(bank.with_suffix(".json"))
    with np.load(bank) as bank_state, np.load(body) as body_state, np.load(fit_path) as model:
        if str(model["bank_sha256"]) != _hash(bank) or str(model["body_sha256"]) != _hash(body):
            raise ValueError("Dynamics calibration came from different encoders or body map")
        field = EffectField(bank_state, receipt, body_state["weights"], model)
    run(feeds, bank, PLAN, deck, "local", output, cap=200, field_hook=lambda _: field,
        details=dict(body_coordinates="actual-servo-effects", dynamics_sha256=_hash(fit_path)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "physical"))
    p.add_argument("output", type=Path)
    p.add_argument("--fit", type=Path, default=OUT / "servo-effects.npz")
    p.add_argument("--bank", type=Path, default=BANK)
    p.add_argument("--body", type=Path, default=OUT / "body.npz")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--deck", choices=("held_openings", "held_replies"), default="held_openings")
    args = p.parse_args()
    if args.phase == "fit":
        fit(args.bank, args.body, args.output)
    else:
        physical(args.bank, args.fit, args.body, args.feeds, args.output, args.deck)
