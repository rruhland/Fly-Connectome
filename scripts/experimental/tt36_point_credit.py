"""Replay the two changed retention points, switching feedback at the third opponent stroke."""

import argparse
from pathlib import Path

import numpy as np

from . import tt36_reach as reach
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import resources
from .tt35_replies import varied_opponent
from .tt36_course import OUT


def run(feeds):
    paths = {
        "fixed": OUT.parent / "tt35/replies-nominal.json.gz",
        "aligned": OUT / "reach-retained-aligned.json.gz",
    }
    tapes = {mode: _read(path) for mode, path in paths.items()}
    model = _read(reach.PUPIL)
    authority = _read(OUT / "reach-checkpoint.json")["authority_rad"]
    with np.load(reach.BODY_BANK) as bank:
        body = reach.Body(bank)
    with np.load(OUT / "reach-observations.npz") as obs:
        relation = reach.Relation(obs)
    launches = measured_launches(feeds)
    rows = []
    with centered(model["ready"]):
        arena = replay_arena([])
        stereo, rig, stride, lo, hi = resources(arena)
        try:
            for i, (old, changed) in enumerate(
                zip(tapes["fixed"]["rows"], tapes["aligned"]["rows"], strict=True)
            ):
                if old["score"]["learner"] == changed["score"]["learner"]:
                    continue
                card = tapes["fixed"]["cards"][i]
                seed = tapes["fixed"]["reply_seeds"][i]
                for prefix, source in [("aligned", changed), ("fixed", old)]:
                    switch = source["placements"]["delivered"][2]["far_contact"]["step"]

                    class Switch(reach.Reach):
                        def bias(
                            self,
                            tick,
                            optical,
                            joints,
                            frames,
                            cut=switch,
                            start=prefix,
                        ):
                            if (tick < cut) == (start == "aligned"):
                                return super().bias(tick, optical, joints, frames)
                            return None

                    with varied_opponent(card["index"], seed):
                        row = reach.play(
                            arena,
                            stereo,
                            rig,
                            model,
                            stride,
                            lo,
                            hi,
                            launches[card["id"]],
                            card,
                            body,
                            relation,
                            authority,
                            1,
                            feedback=Switch,
                        )
                    np.testing.assert_array_equal(
                        row["actions"][:switch], source["actions"][:switch]
                    )
                    np.testing.assert_allclose(
                        row["placements"]["delivered"][2]["far_contact"]["ball"],
                        source["placements"]["delivered"][2]["far_contact"]["ball"],
                        atol=1e-10,
                        rtol=0,
                    )
                    rows.append(
                        {
                            "index": i,
                            "reply_seed": seed,
                            "prefix": prefix,
                            "switch_tick": switch,
                            "original_own": old["score"]["learner"],
                            "aligned_own": changed["score"]["learner"],
                            "switched_own": row["score"]["learner"],
                            "row": row,
                        }
                    )
        finally:
            stereo.close()
            arena.close()
    _write(
        OUT / "retention-switch.json.gz",
        {
            "sources": {k: _hash(v) for k, v in paths.items()},
            "selection": "Every changed-score point from the eight archived 5530 continuations",
            "rows": rows,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    run(parser.parse_args().feeds)
