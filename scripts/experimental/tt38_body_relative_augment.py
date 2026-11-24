"""Challenge the body-relative field with small analytic transports of training labels."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_deep import run
from .tt38_body_relative import BANK, OUT, PLAN, Body, RelativeField, body_inputs, groups, input_motion

SHIFTS = ((-.037, 0., 0.), (.037, 0., 0.), (0., 0., -.027), (0., 0., .027))


def translate(body, q, shift, lower, upper):
    """Restricted local inverse with no look-ahead or ball model."""
    q = q.copy()
    position = body.position(q)
    desired = position + shift
    prior = np.diag([1., .13, .7, .7, .52])
    delta = np.zeros_like(q)
    for _ in range(2):
        moved = q + delta
        j = body.jacobian(moved) @ prior
        solve = np.linalg.solve(j @ j.transpose(0, 2, 1) + .019**2 * np.eye(3),
                                (desired - body.position(moved))[:, :, None])
        step = (prior @ j.transpose(0, 2, 1) @ solve).squeeze(-1)
        delta = np.clip(delta + step, -.19, .19)
        delta = np.clip(q + delta, lower, upper) - q
    applied = q + delta
    residual = np.linalg.norm(body.position(applied) - desired, axis=-1)
    keep = (residual < .006) & (np.linalg.norm(delta, axis=1) < .28)
    return applied, keep


def synthetic(bank, receipt, weights, *, mode):
    """Make counterfactual supervision only, grouped by its real parent release."""
    s = bank["states"]
    q = s[:, 6:11]
    labels = bank["targets_now"]
    point, releases = groups(receipt)
    low, high = labels[:, :5].min(axis=0), labels[:, :5].max(axis=0)
    use_time = (s[:, 4] < -.9) & (s[:, 1] < -.58) & (bank["marks"][:, 0] % 2 == 0)
    body = Body(weights)
    blocks, report = [], []
    for delta in SHIFTS:
        shift = np.asarray(delta)
        changed, safe = translate(body, q, shift, low - .006, high + .006)
        desired, possible = translate(body, labels[:, :5], shift, low, high)
        active = (safe & possible & use_time & (s[:, 2] + shift[2] > .755))
        taught = s.copy()
        taught[:, :3] += shift
        output = labels.copy()
        if mode == "translated":
            taught[:, 6:11] = changed
            output[:, :5] = desired
        p, v = input_motion(taught, bank["marks"], point, body)
        feature = body_inputs(taught[:, :6], taught[:, 6:11], p, v, "motion")
        blocks.append((feature[active], output[active], np.sign(taught[active, 4]), releases[active]))
        report.append(dict(shift_m=delta, rows=int(active.sum()),
                           lesson="translated command" if mode == "translated" else "unchanged command"))
    return tuple(np.concatenate(col) for col in zip(*blocks, strict=True)), report


def compare(bank_path, body_path, output):
    receipt = _read(bank_path.with_suffix(".json"))
    with np.load(bank_path) as doc, np.load(body_path) as calib:
        body, bank = Body(calib["weights"]), dict(doc)
    point, releases = groups(receipt)
    s = bank["states"]
    p, v = input_motion(s, bank["marks"], point, body)
    late = (bank["marks"][:, 1] > -.13) & (bank["marks"][:, 1] < 0)
    target = bank["targets_now"][late]
    good_releases = releases[late]
    metric = {}
    for mode in ("motion", "translated", "visual_only"):
        sample, counts = (None, []) if mode == "motion" else synthetic(bank, receipt, body.weights, mode=mode)
        effective = np.empty_like(target)
        for release in np.unique(good_releases):
            added = () if sample is None else (tuple(column[sample[3] != release] for column in sample[:3]),)
            field = RelativeField(bank, receipt, body.weights, "motion", use=releases != release, additional=added)
            ix = np.flatnonzero(good_releases == release)
            for k, row in zip(ix, np.flatnonzero(late)[ix], strict=True):
                guess, conf = field.at(s[row, :6], s[row, 6:11], p[row], v[row])
                ready = np.asarray(receipt["ready"])
                effective[k] = ready if guess is None else ready + conf * (guess - ready)
        error = np.linalg.norm(body.position(effective) - body.position(target), axis=1)
        metric[mode] = dict(command_abs_rad=float(np.abs(effective[:, :5] - target[:, :5]).mean()),
                            nominal_blade_error_p50_m=float(np.median(error)),
                            nominal_blade_error_p90_m=float(np.quantile(error, .9)), synthetic=counts)
        _write(output, dict(approach_last_ms=130, test_frames=len(target), no_release_overlap=True,
                            bank_sha256=_hash(bank_path), body_sha256=_hash(body_path), modes=metric))
        print(metric[mode], flush=True)


def physical(bank_path, body_path, feeds, deck, mode, output):
    receipt = _read(bank_path.with_suffix(".json"))
    with np.load(bank_path) as bank, np.load(body_path) as calib:
        weights = calib["weights"]
        sample, counts = synthetic(bank, receipt, weights, mode=mode)
        fit = RelativeField(bank, receipt, weights, "motion", additional=(sample[:3],))
    run(feeds, bank_path, PLAN, deck, "local", output, cap=200, field_hook=lambda _: fit,
        details=dict(body_coordinates="motion", augmentation=mode, analytic_labels=counts,
                     body_sha256=_hash(body_path)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("compare", "physical"))
    p.add_argument("output", type=Path)
    p.add_argument("--bank", type=Path, default=BANK)
    p.add_argument("--body", type=Path, default=OUT / "body.npz")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--deck", choices=("held_openings", "held_replies"), default="held_openings")
    p.add_argument("--mode", choices=("translated", "visual_only"), default="translated")
    a = p.parse_args()
    if a.phase == "compare":
        compare(a.bank, a.body, a.output)
    else:
        physical(a.bank, a.body, a.feeds, a.deck, a.mode, a.output)
