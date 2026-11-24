"""Replay the coupled 100-return point between identical wrist-only bounce assessments."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_action_memory import Experience
from .tt29_bounce import OUTPUT, held_score, load_checkpoint
from .tt29_bounce_continued import RESULT as CONTINUATION
from .tt29_bounce_drift import RESULT as DRIFT
from .tt29_bounce_promotion import PROMOTION, RETAINED
from .tt29_vision_replay import replay

SOURCE = OUTPUT / "rally-objective-envelope.json.gz"
TABLE_AUDIT = OUTPUT / "coupled-fresh100-audit.json"
RESULT = OUTPUT / "tt29_coupled_retention.json"
SHOT_ID = 4187


def restore(prefix):
    """Load the same retained motion examples used by the actual table entry point."""
    memory = Experience()
    for example in prefix:
        memory.add(np.asarray(example["flight"]), np.asarray(example["targets"]), example["time_to_contact"])
    return memory


def fingerprint(memory):
    return digest_json([dict(flight=row.flight.tolist(), targets=row.targets.tolist(),
                             time_to_contact=row.time_to_contact) for row in memory.examples])


def scores(rows, label):
    outcomes = [row["results"][label] for row in rows]
    streaks = [row["streak"] for row in outcomes]
    return dict(rebounds=sum(row["rebounds"] for row in outcomes),
                longest=max(streaks), median=float(np.median(streaks)), count=len(outcomes))


def run(feeds, output, *, scratch=None):
    source_hash = _hash(SOURCE)
    source = _read(SOURCE)
    route = _read(TABLE_AUDIT)
    course = _read(PROMOTION)
    if (route["source_sha256"] != source_hash or route["shot_id"] != SHOT_ID
            or not route["exact_events"] or route["score"]["returns"] != 100):
        raise ValueError("The coupled 100-return archive does not match its exact replay")
    solo_hash = _hash(RETAINED)
    if solo_hash != course["published_checkpoint_sha256"]:
        raise ValueError("The retained wrist checkpoint is different from the promotion receipt")
    table = next(row for row in source["rows"] if row["shot"]["id"] == SHOT_ID)
    first, last = table["initial_examples"], table["final_examples"]
    memory_before = restore(source["checkpoint"][:first])
    digest_before = fingerprint(memory_before)
    kept = load_checkpoint(RETAINED)
    label = str(kept["drops"])
    original = [{**row, "results": {label: row["results"][label]}} for row in course["held"]]
    kwargs = dict(held_count=len(original), seed=course["held_seed"])
    workspace = TemporaryDirectory(prefix="tt29-coupled-retention-") if scratch is None else nullcontext(scratch)
    with workspace as directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        baseline_path, replay_path = directory / "baseline.json.gz", directory / "table.json"
        if baseline_path.exists():
            before = _read(baseline_path)
        else:
            before = held_score({label: kept}, **kwargs)
            _write(baseline_path, before)
        if before != original:
            raise ValueError("The pre-table wrist command tapes differ from the held promotion deck")
        print("held wrist baseline exact", len(before), flush=True)
        if replay_path.exists():
            observed = _read(replay_path)
            if observed["source_sha256"] != source_hash or observed["feed_sha256"] != _hash(feeds):
                raise ValueError("The staged replay used different recorded inputs")
        else:
            observed = replay(feeds, SOURCE, SHOT_ID, replay_path)
    for field in ("events_sha256", "state_sha256", "score", "learning"):
        if digest_json(observed[field]) != digest_json(route[field]):
            raise ValueError(f"Coupled replay differs in {field}")
    if digest_before != fingerprint(memory_before):
        raise ValueError("The archived tape mutated the pre-table motor memory")
    memory_after = restore(source["checkpoint"][:last])
    restored = fingerprint(memory_after)
    after = held_score({label: load_checkpoint(RETAINED)}, **kwargs)
    if before != after or solo_hash != _hash(RETAINED) or restored != fingerprint(memory_after):
        raise ValueError("A retained state or wrist command tape changed across the table round trip")
    later = {}
    for kind, path in (("new_motion_fit", CONTINUATION), ("rebound_drift", DRIFT)):
        receipt = _read(path)
        later[kind] = dict(sha256=_hash(path), promoted=receipt["promoted"],
                           difficult_held_rebound_change=receipt["held_summary"]["difficult"]["qualified_rebound_change"])
    result = dict(schema="tt29-coupled-bounce-retention-1", archive_sha256=source_hash,
                  official_table_audit_sha256=_hash(TABLE_AUDIT), camera_streams=source["view"],
                  table_wrist_checkpoint=source["bounce_checkpoint"], table_solo_timing_checkpoint=source["timing_checkpoint"],
                  replay=dict(shot_id=SHOT_ID, returns=100, exact_events=True, exact_state=True,
                              events_sha256=observed["events_sha256"], state_sha256=observed["state_sha256"],
                              physics_states=observed["clearance"]["samples"], motor=observed["motor"]),
                  table_examples=dict(before=first, after=last, original_live_update_count=last-first,
                                      before_sha256=digest_before, restored_after_sha256=restored),
                  solo=dict(checkpoint_sha256=solo_hash, source_promotion_sha256=_hash(PROMOTION),
                            held_seed=kwargs["seed"], held_before=scores(before, label),
                            held_after=scores(after, label), all_actions_identical=True, all_referees_identical=True,
                            verified_command_replays=2*len(before),
                            rows=[dict(seed=row["seed"], regime=row["regime"],
                                       action_sha256=digest_json(row["results"][label]["actions"]),
                                       referee_sha256=row["results"][label]["referee"],
                                       rebounds=row["results"][label]["rebounds"],
                                       streak=row["results"][label]["streak"]) for row in after]),
                  previously_measured_promotion=dict(retained=course["retained"], later_challengers=later))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, result)
    print("verified", len(after), "matching wrist releases around 100 returns", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--scratch", type=Path, help="preserve staged assessments for interrupted runs")
    args = parser.parse_args()
    run(args.feeds, args.output, scratch=args.scratch)
