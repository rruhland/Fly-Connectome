"""Practice short direct joint corrections around a player's own observed contacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt25_reference_receiver import ReferenceReceiver
from .tt25_recorded_flight import OUTPUT


def deviations():
    draws = qmc.Sobol(6, scramble=True, seed=81501).random_base2(7)
    # All five axes are eligible. The sixth draw selects the starting lead.
    return [(3 + int(x[5]*7), (2*x[:5]-1)*.9) for x in draws]


def trial(arena, shot, original, trigger=None, delta=None):
    arena.launch(shot)
    contacts, actions = [], []
    for tick in range(arena.max_steps):
        command = np.array(original[min(tick,len(original)-1)]).copy()
        if trigger is not None and trigger <= tick < trigger+8:
            command[:5] = np.clip(command[:5]+delta, -1, 1)
        actions.append(command.tolist())
        _, _, done, cap, info = arena.step(command, render=False)
        contacts.extend(e for e in info["events"] if e["kind"]=="contact" and e["side"]==1)
        if done or cap or arena.legal_returns:
            break
    return dict(legal=arena.legal_returns, contact=bool(contacts), steps=[e["step"] for e in contacts],
                event_hash=digest_json(arena.events), actions=actions)


def run(path):
    import hashlib
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != json.loads((OUTPUT/"recorded-secondary-reference.json").read_text())["sha256"]:
        raise ValueError("Wrong DeepMind data")
    shots = {r["id"]:r for r in json.loads(raw)}
    baseline = _read(OUTPUT/"recorded-moderate-receive.json.gz")
    sources = [c["results"]["trained"] for c in baseline["cases"] if c["results"]["trained"]["contact"]]
    roster = json.loads((OUTPUT/"recorded-reach.json").read_text())["cases"][64:96]
    if {r["id"] for r in sources} & {r["id"] for r in roster}:
        raise ValueError("Practice and assessment overlap")
    arena = ReferenceReceiver()
    histories = []
    try:
        for source in sources:
            shot, baseline_commands = shots[source["id"]], source["actions"]
            original = trial(arena, shot, baseline_commands)
            if original["event_hash"] != source["events"] or not original["contact"]:
                raise ValueError("Unmodified originating contact did not reproduce")
            stamp = original["steps"][0] - 1
            scores = []
            for lead, delta in deviations():
                trigger = max(0,stamp-lead)
                row = trial(arena, shot, baseline_commands, trigger, delta)
                scores.append(dict(lead=lead, delta=delta.tolist(), legal=row["legal"], contact=row["contact"]))
                if row["legal"]:
                    again = trial(arena, shot, baseline_commands, trigger, delta)
                    if again != row:
                        raise ValueError("Practice success was not reproducible")
                    break
            histories.append(dict(id=source["id"], original_contact_tick=stamp, trials=scores))
            print("practice",source["id"],len(scores),scores[-1]["legal"],flush=True)
        _write(OUTPUT / "recorded-contact-practice.json.gz", dict(histories=histories,
               held_roster=[r["id"] for r in roster], qualifying_trajectories=sum(
                   any(t["legal"] for t in h["trials"]) for h in histories)))
    finally:
        arena.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind",type=Path)
    run(p.parse_args().deepmind)
