"""On each matched game, rotate the state-to-command pairing inside the reward-credit window."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt40_within_rally import replay_memory
from .tt41_early_optics_carry import resume

SOURCE = course.ROOT / "learning-24.json.gz"
PREVIOUS = course.ROOT / "learning-12.json.gz"


def rotate(bank, count):
    if count > 1:
        old = list(bank)[-count:]
        for i, (state, _, tag) in enumerate(old):
            bank[len(bank) - count + i] = (
                state,
                old[(i + max(1, count // 2)) % count][1],
                tag,
            )


def run(out):
    data, prev = _read(SOURCE), _read(PREVIOUS)
    m = course.mpc
    initial = len(prev["rows"])
    receipt = {
        "source": _hash(SOURCE),
        "boundary": _hash(PREVIOUS),
        "cards": data["cards"][initial : len(data["rows"])],
        "rows": [],
        "pairs": [],
    }
    if out.exists():
        receipt = _read(out)
    if receipt["source"] != _hash(SOURCE) or receipt["boundary"] != _hash(PREVIOUS):
        raise ValueError("Incompatible cached audit")
    with np.load(m.BANK) as b:
        raw = m.DeepField(b, "local")
    with m.centered(data["ready"]):
        a = m.arena()
        sensors = m.resources(a)
        model = m.MPC(m.BANK, raw, variant="elastic", budget=data["budget"])
        try:
            model.priors = course.Carried(
                m.BANK,
                model.engine,
                practice=[course.PRACTICE],
                held={c["id"] for c in data["cards"]},
            )
            resume(model, PREVIOUS.with_suffix("").with_suffix(".npz"), prev)
            original = model.priors.keep
            for j, c in enumerate(receipt["cards"]):
                i = initial + j
                control = data["rows"][i]
                if model.priors.digest() != control["memory"]["before"]:
                    raise ValueError("Mismatched starting memory")
                if j >= len(receipt["rows"]):
                    prior = model.priors.online.copy()
                    rotations = []

                    def miscredit(archives, leg, records=rotations):
                        count = original(archives, leg)
                        rotate(model.priors.online, count)
                        records.append(count)
                        return count

                    with patch.object(model.priors, "keep", miscredit):
                        sham = m.play(
                            a,
                            *sensors,
                            data["ready"],
                            c,
                            control["shot"],
                            raw,
                            model,
                            course.OPTIC,
                            "live",
                            data["cap"],
                        )
                    model.priors.online = prior
                    receipt["rows"].append(sham)
                    receipt["pairs"].append(
                        {
                            "id": c["id"],
                            "aligned": control["score"]["learner"],
                            "misaligned": sham["score"]["learner"],
                            "windows": rotations,
                        }
                    )
                    _write(out, receipt)
                    print(
                        c["id"],
                        control["score"]["learner"],
                        sham["score"]["learner"],
                        flush=True,
                    )
                else:
                    model.serial += 1
                replay_memory(
                    a,
                    sensors,
                    model.priors,
                    c,
                    control,
                    data["ready"],
                    course.OPTIC,
                    i + 1,
                )
        finally:
            model.close()
            sensors[0].close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    run(p.parse_args().out)
