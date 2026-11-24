"""Transfer a reward readout to a different direct-joint controller on the same physical task."""

import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt38_motor_value import MotorValue


def run(model_path, deck, out, limit, bank, feeds, optics, manifest):
    dataset = _read(model_path)
    roster = _read(manifest)
    cards = roster["held"][deck]
    if {r["id"] for r in cards}.intersection(dataset["release"]):
        raise ValueError("Test release was used to fit the transferred readout")
    judge = MotorValue(dataset)
    source_cost = mpc.cost

    def ranked(prediction, response):
        cost, legal = source_cost(prediction, response)
        if legal.any():
            cost[legal] = 0.40 * cost[legal] + 0.60 * (
                judge.cost(prediction[legal]) - 5.0
            )
        return cost, legal

    with patch.object(mpc, "cost", ranked):
        mpc.run(
            feeds,
            bank,
            mpc.PLAN,
            deck,
            "live",
            out,
            optic=optics,
            cap=200,
            limit=limit,
            variant="elastic",
            protocol_path=manifest,
        )
    result = _read(out)
    result["transferred_reward_model"] = {
        "source": str(model_path),
        "sha256": _hash(model_path),
        "source_controller": "geometric",
        "cases": dataset["n"],
        "model_weight": 0.60,
        "release_overlap": 0,
    }
    _write(out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deck", choices=("held_openings", "held_replies"))
    p.add_argument("output", type=Path)
    p.add_argument(
        "--model", type=Path, default=Path("artifacts/tt37/motor-value.json.gz")
    )
    p.add_argument("--bank", type=Path, default=mpc.BANK)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--optics", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    p.add_argument(
        "--manifest", type=Path, default=Path("artifacts/tt38/joint-mpc-protocol.json")
    )
    a = p.parse_args()
    run(
        a.model,
        a.deck,
        a.output,
        mpc.ASSESSMENT_LIMITS[a.deck],
        a.bank,
        a.feeds,
        a.optics,
        a.manifest,
    )
