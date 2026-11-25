"""Compare independent games and diagnose the first command that changed after one long rally."""

import numpy as np

from . import tt51_transfer_after_point as boundary
from .tt4_transfer_trial import _read, _write, _hash


def compare():
    paths = [boundary.ROOT / (nm + "-held.json.gz") for nm in ("before", "after")]
    before, after = map(_read, paths)
    if before["cards"] != after["cards"] or not len(before["cards"]) == len(
        before["rows"]
    ) == len(after["rows"]):
        raise ValueError("Incomplete paired season")
    result = []
    for c, l, r in zip(before["cards"], before["rows"], after["rows"], strict=True):
        i = next(
            (t for t, (x, y) in enumerate(zip(l["actions"], r["actions"])) if x != y),
            None,
        )
        row = dict(
            id=c["id"],
            before=l["score"]["learner"],
            after=r["score"]["learner"],
            first_command=i,
        )
        if i is None:
            if l["actions"] != r["actions"] or l["events"] != r["events"]:
                raise ValueError("Identical tape disagrees")
        else:
            if [e for e in l["events"] if e["step"] <= i] != [
                e for e in r["events"] if e["step"] <= i
            ]:
                raise ValueError("Physics diverged before a changed action")
            deliveries = l["placements"]["delivered"]
            leg = next(
                (
                    k
                    for k in range(len(deliveries) - 1, -1, -1)
                    if deliveries[k].get("far_contact")
                    and deliveries[k]["far_contact"]["time"] * 60 <= i
                ),
                None,
            )
            if leg is not None:
                u, v = deliveries[leg], r["placements"]["delivered"][leg]
                same = (
                    u["far_contact"] == v["far_contact"]
                    and u["near_bounce"] == v["near_bounce"]
                )
                row.update(
                    first_offer=leg,
                    same_ball=same,
                    first_legal=[bool(u["near_landed"]), bool(v["near_landed"])],
                )
        result.append(row)
    rates = np.array([[z["before"], z["after"]] for z in result])
    delta = rates[:, 1] - rates[:, 0]
    rng = np.random.default_rng(57111)
    ci = np.percentile(
        delta[rng.integers(len(delta), size=(20000, len(delta)))].mean(axis=1),
        [2.5, 97.5],
    )
    _write(
        boundary.ROOT / "transfer.json",
        dict(
            sources=[_hash(p) for p in paths],
            points=result,
            total=rates.sum(0).tolist(),
            median=np.median(rates, axis=0).tolist(),
            paired_mean_ci=ci.tolist(),
        ),
    )


if __name__ == "__main__":
    compare()
