"""Replay every learned low-central score on the registered-body course."""

from . import tt68_clamp_transfer as transfer
from .tt39_rally_witness import witness
from .tt39_opportunity_map import projected_height
from .tt4_transfer_trial import _read, _write


def verify():
    source = transfer.ROOT / "registered.json.gz"
    course = _read(source)
    counts = []
    with transfer.fixture.slid(course["fixture_dx_m"]):
        for index, (card, row) in enumerate(
            zip(course["cards"], course["rows"], strict=True)
        ):
            deliveries = row["placements"]["delivered"]
            for leg, offer in enumerate(deliveries):
                line = offer.get("receive_line")
                if not line or not offer["near_landed"]:
                    continue
                x, z = line["xyz"][0], projected_height(line)
                if abs(x) >= 0.10 or z >= 1:
                    continue
                target = transfer.ROOT / f"random-low-{index}-{leg}.json"
                witness(source, card["id"], target, False, index)
                record = _read(target)
                clearance = record["clearance"]
                if clearance["nearest"]["signed_m"] <= 0 or record["launches"] != 1:
                    raise ValueError("Native witness failed physical qualification")
                counts.append(
                    dict(
                        game=index + 1,
                        id=card["id"],
                        own_ordinal=leg + 1,
                        answered=leg + 1 < len(deliveries)
                        and bool(deliveries[leg + 1]["near_bounce"]),
                        x_m=x,
                        projected_height_m=z,
                        replay=target.name,
                    )
                )
    _write(transfer.ROOT / "low-witnesses.json", dict(scores=counts))


if __name__ == "__main__":
    verify()
