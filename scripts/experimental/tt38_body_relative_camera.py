"""Check body-only calibration against synchronized images and compliant paddle motion."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_source_rally import blade_sighting
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, resources
from .tt35_error import triangulate
from .tt37_opponent import arena
from .tt38_body_relative import BANK, OUT, Body

EYES = ("left", "right")


def blade_center(frames, rig):
    readings = [blade_sighting(frames[name]) for name in EYES]
    if any(r is None for r in readings):
        return None
    return triangulate([r[0] * frames[name].shape[0] / 360.
                        for name, r in zip(EYES, readings, strict=True)], rig)


def capture(bank, body_path, out):
    manifest = _read(bank.with_suffix(".json"))
    if _hash(bank) != manifest["bank_sha256"]:
        raise ValueError("The observation bank has changed")
    with np.load(bank) as data, np.load(body_path) as fitted:
        marker = data["marks"]
        observed_q = data["states"][:, 6:11]
        body = Body(fitted["weights"])
    cursor = 0
    sources = {}
    records, poses, observations, official, predicted = [], [], [], [], []
    with centered(manifest["ready"]):
        a = arena(protocol="random")
        stereo, rig, *_ = resources(a)
        try:
            for i, sample in enumerate(manifest["practice"]):
                card = sample["card"]
                indices = np.arange(cursor, cursor + sample["frames"])
                cursor += sample["frames"]
                if i % 4 != 0:
                    continue
                # Provenance is independent of camera visibility and only uses a training tape.
                use = indices[((marker[indices, 0].astype(int) % 10) == 0)
                              & (np.abs(marker[indices, 1]) < .21)]
                ticks = {int(marker[j, 0]): j for j in use}
                source = sample["source"]
                if source not in sources:
                    path = bank.parent / source
                    if _hash(path) != manifest["sources"][source]:
                        raise ValueError("Teacher source has changed")
                    sources[source] = _read(path)
                s = sources[source]
                row = s["rows"][sample["source_row"]]
                if s["cards"][sample["source_row"]] != card:
                    raise ValueError("Mismatched camera capture receipt")
                a.launch(row["shot"], card=card)
                a.partner.reset(seed=card["seed"])
                for tick, command in enumerate(row["actions"]):
                    if tick in ticks:
                        q = a.joints[:5]
                        if not np.allclose(q, observed_q[ticks[tick]], atol=1e-9):
                            raise ValueError("Paddle sighting not aligned with the encoder tick")
                        measured = blade_center(stereo.capture(), rig)
                        poses.append(q)
                        observations.append(measured if measured is not None else np.full(3, np.nan))
                        predicted.append(body.position(q))
                        # Only this diagnosis receives native paddle location. No actor uses it.
                        official.append(a.paddle.copy())
                        records.append((i, tick, card["id"]))
                    a.exchange(command, FAR_COMMAND)
                if digest_json(a.events) != digest_json(row["events"]):
                    raise ValueError("Training observation tape no longer replays")
                print(card["id"], len(records), flush=True)
        finally:
            stereo.close()
            a.close()
    q, rgb, nominal, witness, info = map(np.asarray, (poses, observations, predicted, official, records))
    use = np.isfinite(rgb).all(axis=1)
    qualified = use & (np.linalg.norm(rgb - nominal, axis=1) < .055)
    err = lambda a, b: np.linalg.norm(a - b, axis=1)
    def describe(x):
        return dict(n=len(x), median_m=float(np.median(x)), p90_m=float(np.quantile(x, .90))) if len(x) else dict(n=0)
    stats = dict(
        images=len(rgb), two_eyes=int(use.sum()), within_55_mm_of_nominal=int(qualified.sum()),
        nominal_vs_native=describe(err(nominal, witness)), rgb_vs_native=describe(err(rgb[use], witness[use])),
        gate_rgb_vs_native=describe(err(rgb[qualified], witness[qualified])),
        gate_nominal_vs_native=describe(err(nominal[qualified], witness[qualified])),
        bank_sha256=_hash(bank), body_sha256=_hash(body_path),
        note="Native witness for scoring only. Images are from training tapes, not reserved openings.",
    )
    np.savez_compressed(out, q=q, observed=rgb, nominal=nominal, native_diagnostic=witness, group_tick_release=info)
    _write(out.with_suffix(".json"), stats)
    print(stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--bank", type=Path, default=BANK)
    p.add_argument("--body", type=Path, default=OUT / "body.npz")
    a = p.parse_args()
    capture(a.bank, a.body, a.output)
