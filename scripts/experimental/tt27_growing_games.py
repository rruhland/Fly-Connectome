"""Play regulation points selected prospectively from a separate recorded roster, end to end."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_trial import COACHED
from .tt26_grip_course import Grip, ReferenceReceiver, load_match
from .tt26_match_continuation import archived_step
from .tt27_flight_feedback import OUT
from .tt27_full_points import one
from .tt27_outcome_receiver import OutcomeModel, independent
from .tt27_reply import DEV_DONORS, Flight, PoseMap, examples
from .tt27_reply_recorded import transported


def run_group(args):
    path, cohort = args
    lookup, initial = load_match(path)
    model = OutcomeModel()
    src = [e for e in examples(_read(OUT / "reply-traces.json.gz")["traces"]) if e["id"] in DEV_DONORS]
    arena = ReferenceReceiver(stationary_opponent=False)
    replay = ReferenceReceiver()
    arena.max_steps = replay.max_steps = 450
    install_rally_opponent(arena, target_profile=_read(COACHED)["metadata"]["opening"]["far_profile"])
    tape = FarTape(arena)
    poses, flight = PoseMap(arena), Flight()
    bank = transported(src, Grip(arena.model), poses, 0.0)
    rows = []
    # Alternate two near recordings from each of 19 source neighborhoods, plus 32 raw recordings.
    roster = [s for g, s in independent(lookup) if g == cohort]
    roster = [s for i, s in enumerate(roster) if i % 6 in (0, 3)] if cohort == "nearby" else roster[:32]
    try:
        for shot in roster:
            by = {}
            for name in ("original", "blend"):
                r = one(
                    arena,
                    tape,
                    shot,
                    initial,
                    bank,
                    poses,
                    flight,
                    None,
                    "repeat",
                    opener=lambda k: model.act(k, name),
                )
                replay.launch(shot)
                for t, a in enumerate(r["actions"]):
                    _, _, done, cap, _ = archived_step(replay, r["opponent"], t, a)
                if not (done or cap) or digest_json(replay.events) != digest_json(r["events"]):
                    raise ValueError("Regulation game diverged on replay")
                by[name] = r
            rows.append(dict(id=shot["id"], by=by))
        print(
            cohort,
            len(rows),
            {k: [sum(r["by"][k]["near"] >= n for r in rows) for n in (1, 2, 3)] for k in by},
            flush=True,
        )
    finally:
        arena.close()
        replay.close()
    return dict(cohort=cohort, rows=rows)


def run(path):
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        groups = list(pool.map(run_group, [(path, g) for g in ("nearby", "unrestricted")]))
    _write(OUT / "growing-games.json.gz", dict(groups=groups))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    run(parser.parse_args().feeds)
