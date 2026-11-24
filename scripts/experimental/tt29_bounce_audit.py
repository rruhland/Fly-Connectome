"""Check the bounce and receiving command streams against the compiled CAD geometry."""

from collections import Counter

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_public_body import stock
from .tt25_reference_receiver import ReferenceReceiver
from .tt29_arm_clearance import ArmClearance
from .tt29_bounce import COURSE, HELD_CAP, OUTPUT, PRACTICE_CAP
from .tt29_bounce_receive import CAP, DEMONSTRATION, RECEIPT


def track_summary(checker, data, step, commands, finish):
    pairs = Counter()
    failing = 0
    worst = None
    shadow = mujoco.MjData(checker.model)
    for tick in range(len(commands) + 1):
        shadow.qpos[:] = data.qpos
        violations = checker.violations(shadow)
        failing += bool(violations)
        for a, b, interval in violations:
            pairs[(a, b)] += 1
            if worst is None or interval < worst["distance_m"]:
                worst = dict(first=a, second=b, distance_m=interval, tick=tick)
        if tick < len(commands):
            step(commands[tick])
    return dict(frames=len(commands) + 1, failing_frames=failing,
                pairs=[dict(first=a, second=b, frames=n) for (a, b), n in pairs.most_common()],
                worst=worst, referee=finish())


def run():
    solo = _read(COURSE)
    table = _read(RECEIPT)
    demo = _read(DEMONSTRATION)
    on_board = stock(fixture="table-edge")
    regulating = ReferenceReceiver(fixture="table-edge", width=640, height=480,
                                   frameskip=4, max_steps=CAP, view_mode="wrist")
    cad_solo = BounceArena(fixture="table-edge", detail="cad", view="none")
    cad_table = ReferenceReceiver(fixture="table-edge", detail="cad")
    bounce, receiving = [], []
    try:
        solo_checker = ArmClearance(cad_solo._model)
        table_checker = ArmClearance(cad_table.model)

        def bounce_replay(group, row, result, dose):
            on_board.max_steps = PRACTICE_CAP if group == "practice" else HELD_CAP
            on_board.reset(seed=row["seed"], split="practice" if group == "practice" else "held",
                           regime=row["regime"], render=False)
            trail = track_summary(solo_checker, on_board._data, lambda u: on_board.step(u, render=False),
                                  result["actions"], lambda: digest_json(on_board.judge_metrics()))
            if trail["referee"] != result["referee"]:
                raise ValueError(f"The solo audit changed referee for {row['seed']}/{dose}")
            bounce.append(dict(split=group, seed=row["seed"], dose=dose, **trail))

        for row in solo["practice"]:
            bounce_replay("practice", row, row, None)
        for row in solo["held"]:
            for dose, result in row["results"].items():
                bounce_replay("held", row, result, dose)
        for row in table["post_table_held"]:
            for dose, result in row["results"].items():
                bounce_replay("post_table", row, result, dose)

        def receive_replay(group, attempt, shot, result, label):
            regulating.launch(shot)
            trail = track_summary(table_checker, regulating.data, lambda u: regulating.step(u, render=False),
                                  result["actions"], lambda: digest_json(regulating.events))
            if trail["referee"] != result["referee"]:
                raise ValueError(f"The table audit changed referee for {group}/{attempt}/{label}")
            receiving.append(dict(split=group, attempt=attempt, mode=label, legal=result.get("legal", 1), **trail))

        receive_replay("demonstration", 0, demo["shot"],
                       dict(actions=demo["actions"], referee=digest_json(demo["events"])), "coach")
        for split in ("practice", "held"):
            for row in table[split]:
                for label, result in row["results"].items():
                    receive_replay(split, row["attempt"], row["shot"], result, label)
    finally:
        on_board.close()
        regulating.close()
        cad_solo.close()
        cad_table.close()
    summary = {}
    for name, rows in (("bounce", bounce), ("receiving", receiving)):
        summary[name] = {}
        for split in sorted({row["split"] for row in rows}):
            group = [row for row in rows if row["split"] == split]
            summary[name][split] = dict(tapes=len(group), frames=sum(row["frames"] for row in group),
                                        clear=sum(not row["failing_frames"] for row in group),
                                        failing_frames=sum(row["failing_frames"] for row in group))
        print(name, summary[name], flush=True)
    _write(OUTPUT / "tt29_bounce_clearance.json.gz", dict(schema="tt29-bounce-clearance-1",
           bounce_sha256=_hash(COURSE), receiving_sha256=_hash(RECEIPT), demonstration_sha256=_hash(DEMONSTRATION),
           clock_hz=60, checked_tolerance_m=solo_checker.tolerance, geometry="full CAD shadow", summary=summary, bounce=bounce,
           receiving=receiving))


if __name__ == "__main__":
    run()
