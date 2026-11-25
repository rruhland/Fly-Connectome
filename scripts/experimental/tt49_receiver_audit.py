"""Reproduce each discordant next-return result from issued commands, then audit full CAD."""
import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt37_opponent as live
from . import tt49_tactical_receive as receive
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent_audit import CourtClearance, guarded, verify_clearance


def verify(tape, output):
    plan, data = _read(receive.PLAN), _read(tape)
    if not data["complete"] or data["plan_sha256"] != _hash(receive.PLAN):
        raise ValueError("Expected the complete paired physical receiver comparison")
    selected = {c["id"]: c for c in plan["cases"]}
    rows = []
    for r in data["rows"]:
        pair = selected[r["release"]]
        for a,b in zip(*(r["branches"][n] for n in ("low_risk","high_risk")),strict=True):
            if a["own_next"] == b["own_next"]:
                continue
            both = []
            for name, trace in zip(("low_risk","high_risk"),(a,b),strict=True):
                match = pair[name]
                path = Path("artifacts") / (match["source"]+".json.gz")
                if _hash(path)!=plan["source_receipts"][match["source"]]:
                    raise ValueError("Archive changed")
                original = _read(path)
                card = match["card"]
                seed, leg = trace["seed"], match["leg"]+1
                standard = live.draw

                def draw(c, j, protocol, rng):
                    return standard(dict(c,reply_seed=seed) if j==leg else c,j,protocol,rng)

                row = original["rows"][match["row"]]
                with centered(original["ready"]):
                    env, cad = live.arena(protocol="random"), live.arena(protocol="random",detail="cad")
                    try:
                        env.launch(row["shot"],card=card)
                        env.partner.reset(seed=card["seed"])
                        guarded(env, CourtClearance(cad.model,articulated=True))
                        with patch.object(live,"draw",draw):
                            for u in row["actions"][:trace["start_tick"]] + trace["commands"]:
                                env.exchange(u,FAR_COMMAND)
                        if digest_json(env.events)!=trace["native_sha256"]:
                            raise ValueError("Action-only reproduction changed referee events")
                        if env.point_placements()["requested"][leg]["target"] != trace["target"]:
                            raise ValueError("Independent draw target changed")
                        verify_clearance(env)
                        both.append(dict(arm=name, own_next=trace["own_next"], event_sha256=trace["native_sha256"],
                                         clearance=env.clearance_result()))
                    finally:
                        cad.close()
                        env.close()
            rows.append(dict(release=r["release"], seed=a["seed"], matched_target=a["target"], verified=both))
    _write(output, dict(source_sha256=_hash(tape), proof="Complete action-only replay of all discordant worlds",
                        rows=rows))
    print([(r["release"], [(p["own_next"],p["clearance"]["samples"],p["clearance"]["penetration_samples"])
                           for p in r["verified"]]) for r in rows],flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tape",type=Path,default=receive.ROOT/"receiver-live.json.gz")
    p.add_argument("--output",type=Path,default=receive.ROOT/"receiver-cad.json")
    a=p.parse_args()
    verify(a.tape,a.output)
